from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from compat import Environment, EnvironmentMetadata
from graders import grade_episode
from models import FaultStatus, GridAction, GridObservation, GridState, KPIBundle
from tasks import DEFAULT_TASK_ID, FaultSpec, IntervalProfile, TaskSpec, get_task, list_tasks

BATTERY_CHARGE_EFFICIENCY = 0.95
BATTERY_DISCHARGE_EFFICIENCY = 0.93


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class GridGuardianEnvironment(Environment[GridAction, GridObservation, GridState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, default_task_id: str = DEFAULT_TASK_ID):
        self._default_task_id = default_task_id
        self._task = get_task(default_task_id)
        self._state = self._build_initial_state(self._task, episode_id=str(uuid4()))

    def _build_initial_state(self, task: TaskSpec, episode_id: str) -> GridState:
        return GridState(
            episode_id=episode_id,
            task_id=task.task_id,
            task_title=task.title,
            difficulty=task.difficulty,
            current_interval_index=0,
            horizon_steps=task.horizon_steps,
            battery_soc_kwh=task.initial_battery_soc_kwh,
            battery_capacity_kwh=task.battery_capacity_kwh,
            reserve_floor_kwh=task.reserve_floor_kwh,
            active_faults=[self._fault_status(fault) for fault in task.initial_faults],
            history=[],
            done_episode=False,
            final_score=0.0,
        )

    @staticmethod
    def _fault_status(fault: FaultSpec) -> FaultStatus:
        return FaultStatus(
            name=fault.name,
            description=fault.description,
            affected_system=fault.affected_system,
            severity=fault.severity,
            repair_steps_required=fault.repair_steps_required,
            repair_progress=0,
            resolved=False,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GridObservation:
        available_tasks = list_tasks()
        if task_id is None:
            if seed is not None:
                self._task = random.Random(seed).choice(available_tasks)
            else:
                self._task = get_task(self._default_task_id)
        else:
            self._task = get_task(task_id)

        self._state = self._build_initial_state(
            self._task,
            episode_id=episode_id or str(uuid4()),
        )
        self._state.last_action_summary = "Episode initialized. Dispatch the microgrid."
        return self._make_observation(
            reward=0.0,
            done=False,
            last_action_summary=self._state.last_action_summary,
            extra_metadata={"task_catalog": [task.task_id for task in available_tasks]},
        )

    def step(
        self,
        action: GridAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GridObservation:
        if self._state.done_episode:
            return self._make_observation(
                reward=0.0,
                done=True,
                last_action_summary="Episode already completed. Reset before stepping again.",
                extra_metadata={"warning": "episode_already_done", "grade": deepcopy(self._state.terminal_grade)},
            )

        interval = self._current_interval()
        action = self._sanitize_action(action, interval)
        battery_multiplier = self._capacity_multiplier("battery")
        grid_multiplier = self._capacity_multiplier("grid")
        diesel_multiplier = self._capacity_multiplier("diesel")

        actual_battery_dispatch = self._apply_battery_dispatch(action.battery_dispatch_kw, battery_multiplier)
        actual_grid_import = _clip(action.grid_import_kw, 0.0, interval.grid_import_limit_kw * grid_multiplier)
        actual_diesel_dispatch = _clip(action.diesel_dispatch_kw, 0.0, interval.diesel_limit_kw * diesel_multiplier)

        battery_discharge = max(0.0, actual_battery_dispatch)
        battery_charge = max(0.0, -actual_battery_dispatch)
        flexible_available = max(0.0, interval.demand_kw - interval.critical_load_kw)
        curtailed_flexible = _clip(action.flexible_curtailment_kw, 0.0, flexible_available)
        total_supply = interval.renewable_kw + actual_grid_import + actual_diesel_dispatch + battery_discharge
        total_load = interval.demand_kw - curtailed_flexible + battery_charge
        shortage = max(0.0, total_load - total_supply)

        flexible_requested_after_curtail = max(0.0, flexible_available - curtailed_flexible)
        unmet_flexible = min(shortage, flexible_requested_after_curtail)
        unmet_critical = max(0.0, shortage - flexible_requested_after_curtail)
        served_critical = max(0.0, interval.critical_load_kw - unmet_critical)
        served_flexible = max(0.0, flexible_requested_after_curtail - unmet_flexible)
        total_unserved = curtailed_flexible + shortage

        step_cost = (
            actual_grid_import * interval.price_per_kwh
            + actual_diesel_dispatch * self._task.diesel_variable_cost
            + curtailed_flexible * 0.05
        )
        step_emissions = (
            actual_grid_import * interval.carbon_intensity
            + actual_diesel_dispatch * self._task.diesel_emissions_factor
        )

        repair_bonus, repair_note = self._advance_repairs(action.repair_focus)

        self._state.operating_cost += step_cost
        self._state.emissions_kg += step_emissions
        self._state.unmet_total_kwh += total_unserved
        self._state.unmet_critical_kwh += unmet_critical
        self._state.step_count += 1

        reserve_health = min(1.0, self._state.battery_soc_kwh / max(1.0, self._task.reserve_floor_kwh))
        critical_service_ratio = served_critical / interval.critical_load_kw if interval.critical_load_kw else 1.0
        total_service_ratio = (served_critical + served_flexible) / interval.demand_kw if interval.demand_kw else 1.0
        repair_completion_ratio = self._repair_completion_ratio()
        step_reward = self._compute_step_reward(
            critical_service_ratio=critical_service_ratio,
            total_service_ratio=total_service_ratio,
            step_cost=step_cost,
            step_emissions=step_emissions,
            reserve_health=reserve_health,
            repair_bonus=repair_bonus,
        )
        self._state.cumulative_reward += step_reward

        action_summary = (
            f"Interval {interval.label}: battery={actual_battery_dispatch:.1f} kW, "
            f"grid={actual_grid_import:.1f} kW, diesel={actual_diesel_dispatch:.1f} kW, "
            f"curtailment={curtailed_flexible:.1f} kW, critical_unserved={unmet_critical:.1f} kW. "
            f"{repair_note}"
        )
        self._state.last_action_summary = action_summary
        self._state.history.append(
            {
                "label": interval.label,
                "battery_dispatch_kw": round(actual_battery_dispatch, 3),
                "grid_import_kw": round(actual_grid_import, 3),
                "diesel_dispatch_kw": round(actual_diesel_dispatch, 3),
                "flexible_curtailment_kw": round(curtailed_flexible, 3),
                "critical_service_ratio": round(critical_service_ratio, 4),
                "total_service_ratio": round(total_service_ratio, 4),
                "step_reward": round(step_reward, 4),
            }
        )

        self._state.current_interval_index += 1
        done = self._state.current_interval_index >= self._task.horizon_steps
        self._state.done_episode = done

        grade: dict[str, Any] | None = None
        if done:
            grade = grade_episode(self._state, self._task)
            self._state.final_score = float(grade["score"])
            self._state.terminal_grade = grade

        return self._make_observation(
            reward=step_reward,
            done=done,
            last_action_summary=action_summary,
            kpis=KPIBundle(
                critical_service_ratio=critical_service_ratio,
                total_service_ratio=total_service_ratio,
                operating_cost=self._state.operating_cost,
                emissions_kg=self._state.emissions_kg,
                unmet_critical_kwh=self._state.unmet_critical_kwh,
                unmet_total_kwh=self._state.unmet_total_kwh,
                repair_completion_ratio=repair_completion_ratio,
                reserve_health=reserve_health,
                final_score=self._state.final_score,
            ),
            extra_metadata={
                "current_step_metrics": {
                    "step_cost": round(step_cost, 4),
                    "step_emissions": round(step_emissions, 4),
                    "critical_service_ratio": round(critical_service_ratio, 4),
                    "total_service_ratio": round(total_service_ratio, 4),
                },
                "grade": grade or {},
            },
        )

    @property
    def state(self) -> GridState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="GridGuardian",
            description=(
                "Climate-resilience microgrid control room for dispatching storage, grid imports, "
                "diesel backup, curtailment, and repair priorities."
            ),
            version="1.0.0",
            author="Codex for Meta PyTorch OpenEnv Hackathon",
            documentation_url="https://github.com/meta-pytorch/OpenEnv",
        )

    def _current_interval(self) -> IntervalProfile:
        index = min(self._state.current_interval_index, self._task.horizon_steps - 1)
        return self._task.intervals[index]

    def _next_interval_label(self) -> str | None:
        if self._state.current_interval_index + 1 >= self._task.horizon_steps:
            return None
        return self._task.intervals[self._state.current_interval_index + 1].label

    def _sanitize_action(self, action: GridAction, interval: IntervalProfile) -> GridAction:
        return GridAction(
            battery_dispatch_kw=action.battery_dispatch_kw,
            diesel_dispatch_kw=action.diesel_dispatch_kw,
            grid_import_kw=action.grid_import_kw,
            flexible_curtailment_kw=min(action.flexible_curtailment_kw, max(0.0, interval.demand_kw - interval.critical_load_kw)),
            repair_focus=action.repair_focus,
            operator_note=action.operator_note,
            metadata=deepcopy(action.metadata),
        )

    def _apply_battery_dispatch(self, requested_kw: float, multiplier: float) -> float:
        max_discharge = self._task.battery_max_discharge_kw * multiplier
        max_charge = self._task.battery_max_charge_kw * multiplier

        if requested_kw >= 0:
            requested = _clip(requested_kw, 0.0, max_discharge)
            max_available = self._state.battery_soc_kwh * BATTERY_DISCHARGE_EFFICIENCY
            actual = min(requested, max_available)
            self._state.battery_soc_kwh = max(0.0, self._state.battery_soc_kwh - (actual / BATTERY_DISCHARGE_EFFICIENCY))
            return actual

        requested_charge = _clip(-requested_kw, 0.0, max_charge)
        room = max(0.0, (self._task.battery_capacity_kwh - self._state.battery_soc_kwh) / BATTERY_CHARGE_EFFICIENCY)
        actual_charge = min(requested_charge, room)
        self._state.battery_soc_kwh = min(self._task.battery_capacity_kwh, self._state.battery_soc_kwh + (actual_charge * BATTERY_CHARGE_EFFICIENCY))
        return -actual_charge

    def _capacity_multiplier(self, system: str) -> float:
        multiplier = 1.0
        for fault in self._state.active_faults:
            if not fault.resolved and fault.affected_system == system:
                multiplier *= fault.severity
        return multiplier

    def _advance_repairs(self, repair_focus: str) -> tuple[float, str]:
        if repair_focus == "none":
            return 0.0, "No repair crew deployed."

        for fault in self._state.active_faults:
            if fault.name != repair_focus or fault.resolved:
                continue
            fault.repair_progress += 1
            if fault.repair_progress >= fault.repair_steps_required:
                fault.resolved = True
                self._state.resolved_faults += 1
                return 1.0, f"Repair complete for {fault.name}."
            return 0.5, f"Repair progress made on {fault.name}."

        return 0.0, f"No actionable fault matched repair focus '{repair_focus}'."

    def _repair_completion_ratio(self) -> float:
        if not self._task.initial_faults:
            return 1.0
        return _clip(self._state.resolved_faults / len(self._task.initial_faults), 0.0, 1.0)

    def _compute_step_reward(
        self,
        critical_service_ratio: float,
        total_service_ratio: float,
        step_cost: float,
        step_emissions: float,
        reserve_health: float,
        repair_bonus: float,
    ) -> float:
        normalized_cost = 1.0 - min(1.0, step_cost / max(1.0, self._task.cost_budget / self._task.horizon_steps))
        normalized_emissions = 1.0 - min(1.0, step_emissions / max(1.0, self._task.emissions_budget / self._task.horizon_steps))
        reward = (
            0.55 * critical_service_ratio
            + 0.15 * total_service_ratio
            + 0.08 * normalized_cost
            + 0.07 * normalized_emissions
            + 0.10 * reserve_health
            + 0.05 * repair_bonus
        )
        return _clip(reward, 0.0, 1.0)

    def _planning_hint(self, interval: IntervalProfile) -> str:
        active_fault_names = [fault.name for fault in self._state.active_faults if not fault.resolved]
        repair_hint = (
            f"Active faults: {', '.join(active_fault_names)}. Repair early if future risk is high."
            if active_fault_names
            else "No active faults. Use storage to absorb volatility and protect critical load."
        )
        return f"{interval.briefing} {repair_hint} Preserve at least {self._task.reserve_floor_kwh:.0f} kWh when possible."

    def _make_observation(
        self,
        reward: float,
        done: bool,
        last_action_summary: str,
        kpis: KPIBundle | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> GridObservation:
        if done:
            interval = self._task.intervals[-1]
            time_index = self._task.horizon_steps
            next_label = None
        else:
            interval = self._current_interval()
            time_index = self._state.current_interval_index
            next_label = self._next_interval_label()

        flexible_load = max(0.0, interval.demand_kw - interval.critical_load_kw)
        observation_kpis = kpis or KPIBundle(
            operating_cost=self._state.operating_cost,
            emissions_kg=self._state.emissions_kg,
            unmet_critical_kwh=self._state.unmet_critical_kwh,
            unmet_total_kwh=self._state.unmet_total_kwh,
            repair_completion_ratio=self._repair_completion_ratio(),
            reserve_health=min(1.0, self._state.battery_soc_kwh / max(1.0, self._task.reserve_floor_kwh)),
            final_score=self._state.final_score,
        )
        metadata = extra_metadata or {}

        return GridObservation(
            done=done,
            reward=reward,
            metadata=metadata,
            task_id=self._task.task_id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            task_objective=self._task.objective,
            time_index=time_index,
            horizon_steps=self._task.horizon_steps,
            current_interval_label=interval.label,
            next_interval_label=next_label,
            risk_level=interval.risk_level,
            weather_signal=interval.weather_signal,
            demand_kw=interval.demand_kw,
            critical_load_kw=interval.critical_load_kw,
            flexible_load_kw=flexible_load,
            renewable_kw=interval.renewable_kw,
            grid_import_limit_kw=interval.grid_import_limit_kw * self._capacity_multiplier("grid"),
            diesel_limit_kw=interval.diesel_limit_kw * self._capacity_multiplier("diesel"),
            price_per_kwh=interval.price_per_kwh,
            carbon_intensity=interval.carbon_intensity,
            battery_soc_kwh=self._state.battery_soc_kwh,
            battery_capacity_kwh=self._task.battery_capacity_kwh,
            active_faults=[fault.model_copy(deep=True) for fault in self._state.active_faults if not fault.resolved],
            kpis=observation_kpis,
            last_action_summary=last_action_summary,
            planning_hint=self._planning_hint(interval),
            available_actions=[
                "Adjust battery_dispatch_kw to charge or discharge storage.",
                "Tune grid_import_kw within the current import limit.",
                "Dispatch diesel only when reserve or grid power is insufficient.",
                "Use flexible_curtailment_kw to protect critical loads.",
                "Set repair_focus to battery_cooling, grid_tie, or diesel_filter when faults are active.",
            ],
        )

