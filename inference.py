from __future__ import annotations

import json
from typing import Any

from baseline import (
    candidate_actions_from_env,
    candidate_actions_from_state,
    detect_base_url,
    plan_action_from_state,
    plan_action_local,
)
from client import GridGuardianClient
from planner import MODEL_NAME, HybridPlanner
from server.gridguardian_environment import GridGuardianEnvironment
from tasks import list_tasks


ENV_NAME = "gridguardian"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_reward(value: float) -> str:
    return f"{value:.2f}"


def _format_score(value: float) -> str:
    return f"{value:.4f}"


def _format_action(action: dict[str, Any]) -> str:
    compact_action = {key: value for key, value in action.items() if key not in {"metadata", "operator_note"}}
    ordered_keys = (
        "battery_dispatch_kw",
        "diesel_dispatch_kw",
        "grid_import_kw",
        "flexible_curtailment_kw",
        "repair_focus",
    )
    return "|".join(f"{key}={compact_action[key]}" for key in ordered_keys)


def _format_error(value: str | None) -> str:
    if not value:
        return "null"
    return value.replace("\n", " ").strip() or "null"


def main() -> None:
    base_url = detect_base_url()
    remote_client = GridGuardianClient(base_url=base_url) if base_url else None
    planner = HybridPlanner()

    try:
        if remote_client:
            remote_client._ensure_sync_client()

        for task in list_tasks():
            step_index = 0
            score = 0.0
            success = False
            rewards: list[str] = []
            env = None
            observation = None

            print(
                f"[START] task={task.task_id} env={ENV_NAME} model={planner.model_name or MODEL_NAME}",
                flush=True,
            )

            try:
                if remote_client:
                    observation = remote_client.reset(task_id=task.task_id)
                else:
                    env = GridGuardianEnvironment(default_task_id=task.task_id)
                    observation = env.reset(task_id=task.task_id)

                while observation is not None and not observation.done:
                    if env is not None:
                        candidate_actions = candidate_actions_from_env(env, task)
                        base_action = plan_action_local(env, task)
                    else:
                        current_state = remote_client.state()
                        candidate_actions = candidate_actions_from_state(current_state, task)
                        base_action = plan_action_from_state(current_state, task)
                    decision = planner.choose_action(
                        observation,
                        task,
                        base_action=base_action,
                        candidate_actions=candidate_actions,
                    )
                    action = decision.action
                    if remote_client:
                        observation = remote_client.step(action)
                    else:
                        observation = env.step(action)  # type: ignore[union-attr]
                    step_index += 1
                    reward = float(observation.reward or 0.0)
                    rewards.append(_format_reward(reward))
                    last_error = observation.metadata.get("last_action_error")
                    print(
                        f"[STEP] step={step_index} action={_format_action(action.model_dump())} "
                        f"reward={_format_reward(reward)} done={_format_bool(observation.done)} "
                        f"error={_format_error(last_error)}",
                        flush=True,
                    )

                if observation is not None:
                    if remote_client:
                        final_state = remote_client.state()
                        grade = final_state.terminal_grade or observation.metadata.get("grade", {})
                    else:
                        grade = observation.metadata.get("grade", {})
                    score = float(grade.get("score", 0.0))
                    success = bool(grade.get("passed", False))
            except Exception:
                success = False
            finally:
                if env is not None:
                    env.close()
                print(
                    f"[END] success={_format_bool(success)} steps={step_index} "
                    f"score={_format_score(score)} rewards={','.join(rewards)}",
                    flush=True,
                )
    finally:
        if remote_client:
            remote_client.close()


if __name__ == "__main__":
    main()
