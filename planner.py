from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from baseline import plan_action as heuristic_plan_action
from models import GridAction, GridObservation
from tasks import TaskSpec

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")


@dataclass
class PolicyDecision:
    action: GridAction
    planner_mode: str
    planner_error: str | None = None


class HybridPlanner:
    def __init__(
        self,
        api_base_url: str = API_BASE_URL,
        model_name: str = MODEL_NAME,
        hf_token: str | None = HF_TOKEN,
    ) -> None:
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.hf_token = hf_token
        self._client = None

        if OpenAI is not None and hf_token:
            self._client = OpenAI(base_url=api_base_url, api_key=hf_token, timeout=20.0)

    @property
    def llm_enabled(self) -> bool:
        return self._client is not None

    def choose_action(
        self,
        observation: GridObservation,
        task: TaskSpec,
        base_action: GridAction | None = None,
        candidate_actions: list[GridAction] | None = None,
    ) -> PolicyDecision:
        chosen_action = base_action or heuristic_plan_action(observation, task)
        candidates = candidate_actions or [chosen_action]
        if not self._client:
            return PolicyDecision(
                action=chosen_action.model_copy(update={"operator_note": "Heuristic fallback policy"}),
                planner_mode="heuristic",
            )

        candidate_index, note, planner_error = self._select_candidate(
            observation,
            task,
            candidates,
        )
        if candidate_index is None:
            return PolicyDecision(
                action=chosen_action.model_copy(update={"operator_note": "Heuristic fallback policy"}),
                planner_mode="heuristic",
                planner_error=planner_error,
            )

        note = note or f"LLM selected candidate {candidate_index}"
        return PolicyDecision(
            action=candidates[candidate_index].model_copy(update={"operator_note": note[:240]}),
            planner_mode="llm-ranked-candidates",
            planner_error=planner_error,
        )

    def _select_candidate(
        self,
        observation: GridObservation,
        task: TaskSpec,
        candidate_actions: list[GridAction],
    ) -> tuple[int | None, str | None, str | None]:
        assert self._client is not None
        candidate_lines = []
        for index, candidate in enumerate(candidate_actions):
            candidate_lines.append(
                f"{index}: battery={candidate.battery_dispatch_kw}, "
                f"diesel={candidate.diesel_dispatch_kw}, "
                f"grid={candidate.grid_import_kw}, "
                f"curtail={candidate.flexible_curtailment_kw}, "
                f"repair={candidate.repair_focus}"
            )
        prompt = (
            "You are choosing the safest next microgrid control action.\n"
            "Return strict JSON with keys candidate_index and operator_note.\n"
            "candidate_index must be one of the listed integers.\n"
            "operator_note must be one short sentence under 180 characters.\n"
            "Prefer actions that protect critical load, handle active faults early, preserve reserve for later peaks, "
            "and avoid unnecessary diesel or curtailment.\n\n"
            f"Task: {task.title}\n"
            f"Objective: {task.objective}\n"
            f"Current interval: {observation.current_interval_label}\n"
            f"Demand: {observation.demand_kw}\n"
            f"Critical load: {observation.critical_load_kw}\n"
            f"Renewables: {observation.renewable_kw}\n"
            f"Grid limit: {observation.grid_import_limit_kw}\n"
            f"Diesel limit: {observation.diesel_limit_kw}\n"
            f"Battery SOC: {observation.battery_soc_kwh}\n"
            f"Risk level: {observation.risk_level}\n"
            f"Active faults: {[fault.name for fault in observation.active_faults]}\n"
            f"Candidates:\n" + "\n".join(candidate_lines) + "\n"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                temperature=0.0,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": "You return compact valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            message = response.choices[0].message.content or ""
            payload = self._parse_candidate_response(message)
            if payload is None:
                return None, None, "invalid_llm_response"
            candidate_index = payload.get("candidate_index")
            operator_note = " ".join(str(payload.get("operator_note", "")).split()).strip()
            if not isinstance(candidate_index, int) or not (0 <= candidate_index < len(candidate_actions)):
                return None, None, "llm_candidate_out_of_range"
            if not operator_note:
                operator_note = f"LLM selected candidate {candidate_index}"
            return candidate_index, operator_note[:180], None
        except Exception as exc:
            return None, None, str(exc).replace("\n", " ")[:240]

    @staticmethod
    def _parse_candidate_response(message: str) -> dict[str, object] | None:
        text = message.strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        index_match = re.search(r"candidate_index[^0-9-]*(-?\d+)", text)
        if index_match:
            note = " ".join(text.split()).strip()
            return {
                "candidate_index": int(index_match.group(1)),
                "operator_note": note[:180],
            }

        return None
