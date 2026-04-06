from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from client import GridGuardianClient
from graders import grade_episode
from models import FaultStatus, GridAction, GridObservation, GridState
from server.gridguardian_environment import GridGuardianEnvironment
from tasks import TaskSpec, get_task, list_tasks

BEAM_WIDTH = 8
MAX_LOCAL_CANDIDATES = 10


@dataclass
class EpisodeResult:
    task_id: str
    task_title: str
    total_reward: float
    final_score: float
    steps: int
    terminal_grade: dict[str, Any]


def _fault_priority(fault: FaultStatus) -> int:
    priority = {
        "grid_tie": 3,
        "battery_cooling": 2,
        "diesel_filter": 1,
    }
    return priority.get(fault.name, 0)


def _capacity_multiplier(state: GridState, system: str) -> float:
    multiplier = 1.0
    for fault in state.active_faults:
        if not fault.resolved and fault.affected_system == system:
            multiplier *= fault.severity
    return multiplier


def _risk_bucket(risk_level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(risk_level, 0)


def _build_template_action(
    task: TaskSpec,
    state: GridState,
    reserve_target: float,
    grid_fraction: float,
    fuel_order: str,
    curtail_ratio: float,
    repair_focus: str,
    charge_kw: float = 0.0,
) -> GridAction:
    interval = task.intervals[state.current_interval_index]
    flexible = max(0.0, interval.demand_kw - interval.critical_load_kw)
    grid_limit = interval.grid_import_limit_kw * _capacity_multiplier(state, "grid")
    diesel_limit = interval.diesel_limit_kw * _capacity_multiplier(state, "diesel")
    battery_limit = task.battery_max_discharge_kw * _capacity_multiplier(state, "battery")
    charge_limit = task.battery_max_charge_kw * _capacity_multiplier(state, "battery")
    soc = state.battery_soc_kwh
    emergency_floor = max(10.0, task.reserve_floor_kwh * 0.20)

    curtailment = min(flexible, flexible * curtail_ratio)
    net_load = max(0.0, interval.demand_kw - curtailment)
    gap = max(0.0, net_load - interval.renewable_kw)

    grid_import = min(grid_limit, gap * grid_fraction)
    remaining = max(0.0, gap - grid_import)

    preferred_battery = max(0.0, soc - reserve_target)
    emergency_battery = max(0.0, soc - emergency_floor)
    battery_dispatch = 0.0
    diesel_dispatch = 0.0

    if fuel_order == "battery_first":
        battery_dispatch = min(remaining, battery_limit, preferred_battery)
        remaining -= battery_dispatch
        diesel_dispatch = min(remaining, diesel_limit)
        remaining -= diesel_dispatch
    else:
        diesel_dispatch = min(remaining, diesel_limit)
        remaining -= diesel_dispatch
        battery_dispatch = min(remaining, battery_limit, preferred_battery)
        remaining -= battery_dispatch

    if remaining > 0:
        extra_battery = min(remaining, max(0.0, battery_limit - battery_dispatch), max(0.0, emergency_battery - battery_dispatch))
        battery_dispatch += extra_battery
        remaining -= extra_battery

    if remaining > 0 and diesel_dispatch < diesel_limit:
        extra_diesel = min(remaining, diesel_limit - diesel_dispatch)
        diesel_dispatch += extra_diesel
        remaining -= extra_diesel

    if remaining > 0:
        curtailment = min(flexible, curtailment + remaining)

    if charge_kw > 0.0:
        room = max(0.0, (task.battery_capacity_kwh - soc) / 0.95)
        charge_headroom = min(charge_kw, charge_limit, room)
        extra_grid = min(max(0.0, grid_limit - grid_import), charge_headroom)
        if extra_grid > 0:
            battery_dispatch = -extra_grid
            grid_import += extra_grid

    return GridAction(
        battery_dispatch_kw=round(max(-120.0, min(120.0, battery_dispatch)), 3),
        diesel_dispatch_kw=round(max(0.0, min(200.0, diesel_dispatch)), 3),
        grid_import_kw=round(max(0.0, min(250.0, grid_import)), 3),
        flexible_curtailment_kw=round(max(0.0, min(180.0, curtailment)), 3),
        repair_focus=repair_focus,  # type: ignore[arg-type]
        operator_note="Beam-search baseline",
    )


def plan_action(observation: GridObservation, task: TaskSpec) -> GridAction:
    active_faults = sorted(observation.active_faults, key=_fault_priority, reverse=True)
    repair_focus = active_faults[0].name if active_faults else "none"

    future_intervals = task.intervals[observation.time_index:] if observation.time_index < task.horizon_steps else ()
    future_peak_demand = max((interval.demand_kw for interval in future_intervals), default=observation.demand_kw)
    future_peak_risk = any(interval.risk_level in {"high", "critical"} for interval in future_intervals[1:])

    renewable = observation.renewable_kw
    demand = observation.demand_kw
    critical = observation.critical_load_kw
    flexible = observation.flexible_load_kw
    grid_limit = observation.grid_import_limit_kw
    diesel_limit = observation.diesel_limit_kw
    price = observation.price_per_kwh
    carbon = observation.carbon_intensity
    soc = observation.battery_soc_kwh
    reserve_floor = task.reserve_floor_kwh

    emergency_floor = max(15.0, reserve_floor * 0.20)
    desired_curtailment = 0.0
    projected_shortfall = demand - renewable - grid_limit - max(0.0, soc - emergency_floor) - diesel_limit
    if projected_shortfall > 0:
        desired_curtailment = min(flexible, projected_shortfall)

    net_load = demand - desired_curtailment
    gap_after_renewables = max(0.0, net_load - renewable)

    cheap_grid = price <= 0.28 and carbon <= 0.38
    severe_grid = price >= 0.62 or carbon >= 0.58
    critical_gap = max(0.0, critical - renewable)

    if cheap_grid:
        grid_import = min(grid_limit, gap_after_renewables)
    elif severe_grid:
        grid_import = min(grid_limit, max(critical_gap, gap_after_renewables * 0.80))
    else:
        grid_import = min(grid_limit, max(critical_gap, gap_after_renewables * 0.85))

    remaining_gap = max(0.0, gap_after_renewables - grid_import)
    available_battery = max(0.0, soc - reserve_floor)
    if future_peak_risk and observation.time_index < task.horizon_steps - 2:
        available_battery *= 0.75
    battery_dispatch = min(remaining_gap, available_battery, 120.0)
    remaining_gap -= battery_dispatch

    diesel_dispatch = min(diesel_limit, remaining_gap)
    remaining_gap -= diesel_dispatch

    if remaining_gap > 0:
        extra_battery = min(max(0.0, soc - emergency_floor - battery_dispatch), remaining_gap, 120.0 - battery_dispatch)
        battery_dispatch += extra_battery
        remaining_gap -= extra_battery

    if remaining_gap > 0:
        desired_curtailment = min(flexible, desired_curtailment + remaining_gap)

    should_charge = (
        not active_faults
        and cheap_grid
        and soc < min(task.battery_capacity_kwh, reserve_floor + 45.0)
        and demand < future_peak_demand
    )
    if should_charge:
        charge_target = min(30.0, max(0.0, grid_limit - grid_import))
        if charge_target > 0:
            battery_dispatch = -charge_target
            grid_import += charge_target

    return GridAction(
        battery_dispatch_kw=round(max(-120.0, min(120.0, battery_dispatch)), 3),
        diesel_dispatch_kw=round(max(0.0, min(200.0, diesel_dispatch)), 3),
        grid_import_kw=round(max(0.0, min(250.0, grid_import)), 3),
        flexible_curtailment_kw=round(max(0.0, min(180.0, desired_curtailment)), 3),
        repair_focus=repair_focus,
        operator_note="Deterministic remote heuristic",
    )


def _beam_priority(env: GridGuardianEnvironment, task: TaskSpec) -> float:
    state = env.state
    if state.done_episode:
        return float(state.final_score) + 0.001 * state.cumulative_reward

    component_scores = grade_episode(state, task)["component_scores"]
    reserve_score = min(1.0, state.battery_soc_kwh / max(1.0, task.reserve_floor_kwh))
    unresolved_faults = sum(1 for fault in state.active_faults if not fault.resolved)
    remaining_intervals = task.intervals[state.current_interval_index:]
    remaining_demand = sum(interval.demand_kw for interval in remaining_intervals)
    remaining_supply = (
        sum(interval.renewable_kw for interval in remaining_intervals)
        + sum(interval.grid_import_limit_kw for interval in remaining_intervals)
        + sum(interval.diesel_limit_kw for interval in remaining_intervals)
        + state.battery_soc_kwh
    )
    future_buffer = max(-1.0, min(1.0, (remaining_supply - remaining_demand) / max(1.0, remaining_demand)))

    return (
        0.50 * component_scores["critical_service"]
        + 0.22 * component_scores["total_service"]
        + 0.07 * component_scores["cost"]
        + 0.05 * component_scores["emissions"]
        + 0.08 * component_scores["repair"]
        + 0.05 * reserve_score
        + 0.05 * future_buffer
        - 0.02 * unresolved_faults
    )


def _candidate_actions_for_local_env(env: GridGuardianEnvironment, task: TaskSpec) -> list[GridAction]:
    state = env.state
    interval = task.intervals[state.current_interval_index]
    active_faults = sorted(
        [fault for fault in state.active_faults if not fault.resolved],
        key=_fault_priority,
        reverse=True,
    )
    repair_choices: list[str] = ["none"]
    for fault in active_faults[:2]:
        if fault.name not in repair_choices:
            repair_choices.insert(0, fault.name)

    emergency_floor = max(10.0, task.reserve_floor_kwh * 0.20)
    mid_floor = max(emergency_floor, task.reserve_floor_kwh * 0.65)
    low_floor = max(emergency_floor, task.reserve_floor_kwh * 0.40)

    templates: list[tuple[float, float, str, float, float]] = [
        (task.reserve_floor_kwh, 1.0, "diesel_first", 0.0, 0.0),
        (mid_floor, 1.0, "battery_first", 0.0, 0.0),
        (low_floor, 1.0, "battery_first", 0.0, 0.0),
        (mid_floor, 0.85, "battery_first", 0.0, 0.0),
    ]

    if _risk_bucket(interval.risk_level) >= 2:
        templates.extend(
            [
                (low_floor, 1.0, "battery_first", 0.15, 0.0),
                (emergency_floor, 1.0, "battery_first", 0.15, 0.0),
                (emergency_floor, 1.0, "diesel_first", 0.10, 0.0),
            ]
        )
    else:
        templates.extend(
            [
                (task.reserve_floor_kwh, 0.75, "battery_first", 0.0, 0.0),
                (task.reserve_floor_kwh, 1.0, "battery_first", 0.0, 20.0),
            ]
        )

    deduped: dict[tuple[float, float, float, float, str], GridAction] = {}
    for repair_focus in repair_choices:
        for reserve_target, grid_fraction, fuel_order, curtail_ratio, charge_kw in templates:
            action = _build_template_action(
                task=task,
                state=state,
                reserve_target=reserve_target,
                grid_fraction=grid_fraction,
                fuel_order=fuel_order,
                curtail_ratio=curtail_ratio,
                repair_focus=repair_focus,
                charge_kw=charge_kw,
            )
            key = (
                action.battery_dispatch_kw,
                action.diesel_dispatch_kw,
                action.grid_import_kw,
                action.flexible_curtailment_kw,
                action.repair_focus,
            )
            deduped[key] = action

    scored: list[tuple[float, float, GridAction]] = []
    for action in deduped.values():
        probe = deepcopy(env)
        observation = probe.step(action)
        priority = probe.state.final_score if observation.done else _beam_priority(probe, task)
        scored.append((priority, probe.state.cumulative_reward, action))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [action for _, _, action in scored[:MAX_LOCAL_CANDIDATES]]


def plan_action_local(env: GridGuardianEnvironment, task: TaskSpec) -> GridAction:
    initial_candidates = _candidate_actions_for_local_env(env, task)
    if len(initial_candidates) == 1:
        return initial_candidates[0]

    beam: list[tuple[GridGuardianEnvironment, GridAction | None, float]] = [(deepcopy(env), None, 0.0)]
    remaining_depth = task.horizon_steps - env.state.current_interval_index

    for _ in range(remaining_depth):
        expanded: list[tuple[float, float, GridGuardianEnvironment, GridAction | None]] = []
        for sim_env, first_action, cumulative_reward in beam:
            if sim_env.state.done_episode:
                expanded.append((sim_env.state.final_score, cumulative_reward, sim_env, first_action))
                continue

            for action in _candidate_actions_for_local_env(sim_env, task):
                next_env = deepcopy(sim_env)
                observation = next_env.step(action)
                next_first_action = first_action or action
                next_reward = cumulative_reward + float(observation.reward or 0.0)
                priority = next_env.state.final_score if observation.done else _beam_priority(next_env, task)
                expanded.append((priority, next_reward, next_env, next_first_action))

        expanded.sort(key=lambda item: (item[0], item[1]), reverse=True)
        beam = [(env_copy, action, reward) for _, reward, env_copy, action in expanded[:BEAM_WIDTH]]
        if beam and all(candidate_env.state.done_episode for candidate_env, _, _ in beam):
            break

    best_env, best_action, _ = max(
        beam,
        key=lambda item: (
            item[0].state.final_score if item[0].state.done_episode else _beam_priority(item[0], task),
            item[2],
        ),
    )
    return best_action or initial_candidates[0]


def run_task_locally(task_id: str) -> EpisodeResult:
    task = get_task(task_id)
    env = GridGuardianEnvironment(default_task_id=task_id)
    observation = env.reset(task_id=task_id)
    total_reward = 0.0

    while not observation.done:
        action = plan_action_local(env, task)
        observation = env.step(action)
        total_reward += float(observation.reward or 0.0)

    final_state: GridState = env.state
    return EpisodeResult(
        task_id=task.task_id,
        task_title=task.title,
        total_reward=round(total_reward, 4),
        final_score=round(final_state.final_score, 4),
        steps=final_state.step_count,
        terminal_grade=final_state.terminal_grade,
    )


def run_task_remote(task_id: str, base_url: str) -> EpisodeResult:
    task = get_task(task_id)
    client = GridGuardianClient(base_url=base_url)
    observation = client.reset(task_id=task_id)
    total_reward = 0.0
    steps = 0

    while not observation.done:
        action = plan_action(observation, task)
        observation = client.step(action)
        total_reward += float(observation.reward or 0.0)
        steps += 1

    grade = observation.metadata.get("grade", {})
    return EpisodeResult(
        task_id=task.task_id,
        task_title=task.title,
        total_reward=round(total_reward, 4),
        final_score=round(float(grade.get("score", 0.0)), 4),
        steps=steps,
        terminal_grade=grade,
    )


def run_all_tasks(base_url: str | None = None) -> list[EpisodeResult]:
    runner: Callable[[str], EpisodeResult]
    if base_url:
        runner = lambda task_id: run_task_remote(task_id, base_url=base_url)
    else:
        runner = run_task_locally
    return [runner(task.task_id) for task in list_tasks()]


def detect_base_url() -> str | None:
    for key in ("OPENENV_BASE_URL", "SPACE_URL", "ENV_BASE_URL"):
        value = os.getenv(key)
        if value:
            return value.rstrip("/")
    return None


def print_baseline_summary(results: list[EpisodeResult]) -> None:
    for result in results:
        print(
            f"{result.task_id}: score={result.final_score:.4f}, "
            f"reward={result.total_reward:.4f}, steps={result.steps}"
        )


if __name__ == "__main__":
    print_baseline_summary(run_all_tasks(base_url=detect_base_url()))
