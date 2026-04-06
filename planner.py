from __future__ import annotations

import os
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
HF_TOKEN = os.getenv("HF_TOKEN")
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
        self._task_notes: dict[str, str] = {}
        self._client = None

        if OpenAI is not None and hf_token:
            self._client = OpenAI(base_url=api_base_url, api_key=hf_token)

    @property
    def llm_enabled(self) -> bool:
        return self._client is not None

    def choose_action(
        self,
        observation: GridObservation,
        task: TaskSpec,
        base_action: GridAction | None = None,
    ) -> PolicyDecision:
        chosen_action = base_action or heuristic_plan_action(observation, task)
        if not self._client:
            return PolicyDecision(
                action=chosen_action.model_copy(update={"operator_note": "Heuristic fallback policy"}),
                planner_mode="heuristic",
            )

        task_note = self._task_notes.get(task.task_id)
        if task_note is None:
            task_note, planner_error = self._generate_task_note(observation, task)
            if task_note:
                self._task_notes[task.task_id] = task_note
            else:
                return PolicyDecision(
                    action=chosen_action.model_copy(update={"operator_note": "Heuristic fallback policy"}),
                    planner_mode="heuristic",
                    planner_error=planner_error,
                )

        note = self._task_notes.get(task.task_id, "Heuristic fallback policy")
        return PolicyDecision(
            action=chosen_action.model_copy(update={"operator_note": note[:240]}),
            planner_mode="llm-assisted-heuristic",
        )

    def _generate_task_note(self, observation: GridObservation, task: TaskSpec) -> tuple[str | None, str | None]:
        assert self._client is not None
        prompt = (
            "You are advising a microgrid operator. Return one short sentence, under 180 characters, "
            "describing the high-level control strategy. Do not include bullets, JSON, or newlines.\n\n"
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
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                temperature=0.1,
                max_tokens=60,
                messages=[
                    {"role": "system", "content": "You produce concise operational guidance."},
                    {"role": "user", "content": prompt},
                ],
            )
            message = response.choices[0].message.content or ""
            note = " ".join(message.split()).strip()
            if not note:
                return None, "empty_llm_response"
            return note[:180], None
        except Exception as exc:
            return None, str(exc).replace("\n", " ")[:240]
