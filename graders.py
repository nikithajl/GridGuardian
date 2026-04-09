from __future__ import annotations

from typing import Any

from models import GridState
from tasks import TaskSpec, get_task

SCORE_EPSILON = 1e-4


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _strict_score(value: float) -> float:
    return max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, value))


def _repair_completion(task: TaskSpec, state: GridState) -> float:
    if not task.initial_faults:
        return 1.0
    return _clip(state.resolved_faults / len(task.initial_faults))


def _reserve_health(task: TaskSpec, state: GridState) -> float:
    if task.reserve_floor_kwh <= 0:
        return 1.0
    return _clip(state.battery_soc_kwh / task.reserve_floor_kwh)


def grade_episode(state: GridState, task: TaskSpec | None = None) -> dict[str, Any]:
    active_task = task or get_task(state.task_id)
    total_critical_demand = sum(interval.critical_load_kw for interval in active_task.intervals)
    total_demand = sum(interval.demand_kw for interval in active_task.intervals)
    critical_service = _clip(1.0 - (state.unmet_critical_kwh / max(1.0, total_critical_demand)))
    total_service = _clip(1.0 - (state.unmet_total_kwh / max(1.0, total_demand)))
    cost_score = _clip(1.0 - (state.operating_cost / active_task.cost_budget))
    emissions_score = _clip(1.0 - (state.emissions_kg / active_task.emissions_budget))
    repair_score = _repair_completion(active_task, state)
    reserve_score = _reserve_health(active_task, state)

    weights = active_task.grading_weights
    score = (
        weights.get("critical_service", 0.0) * critical_service
        + weights.get("total_service", 0.0) * total_service
        + weights.get("cost", 0.0) * cost_score
        + weights.get("emissions", 0.0) * emissions_score
        + weights.get("repair", 0.0) * repair_score
        + weights.get("reserve", 0.0) * reserve_score
    )
    score = _strict_score(_clip(score))

    passed = critical_service >= 0.85 and score >= 0.75
    summary = (
        f"{active_task.title}: critical_service={critical_service:.3f}, "
        f"total_service={total_service:.3f}, cost_score={cost_score:.3f}, "
        f"emissions_score={emissions_score:.3f}, repair_score={repair_score:.3f}, "
        f"reserve_score={reserve_score:.3f}"
    )

    return {
        "task_id": active_task.task_id,
        "task_title": active_task.title,
        "difficulty": active_task.difficulty,
        "score": round(score, 4),
        "passed": passed,
        "component_scores": {
            "critical_service": round(critical_service, 4),
            "total_service": round(total_service, 4),
            "cost": round(cost_score, 4),
            "emissions": round(emissions_score, 4),
            "repair": round(repair_score, 4),
            "reserve": round(reserve_score, 4),
        },
        "totals": {
            "operating_cost": round(state.operating_cost, 3),
            "emissions_kg": round(state.emissions_kg, 3),
            "unmet_total_kwh": round(state.unmet_total_kwh, 3),
            "unmet_critical_kwh": round(state.unmet_critical_kwh, 3),
            "battery_soc_kwh": round(state.battery_soc_kwh, 3),
        },
        "summary": summary,
    }
