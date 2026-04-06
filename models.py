from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from compat import Action, Observation, State

RepairTarget = Literal["none", "battery_cooling", "grid_tie", "diesel_filter"]


class FaultStatus(BaseModel):
    name: RepairTarget
    description: str
    affected_system: str
    severity: float = Field(..., ge=0.0, le=1.0)
    repair_steps_required: int = Field(..., ge=1)
    repair_progress: int = Field(default=0, ge=0)
    resolved: bool = False


class KPIBundle(BaseModel):
    critical_service_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    total_service_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    operating_cost: float = Field(default=0.0, ge=0.0)
    emissions_kg: float = Field(default=0.0, ge=0.0)
    unmet_critical_kwh: float = Field(default=0.0, ge=0.0)
    unmet_total_kwh: float = Field(default=0.0, ge=0.0)
    repair_completion_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    reserve_health: float = Field(default=1.0, ge=0.0, le=1.0)
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)


class GridAction(Action):
    battery_dispatch_kw: float = Field(
        default=0.0,
        ge=-120.0,
        le=120.0,
        description="Positive values discharge the battery, negative values charge it.",
    )
    diesel_dispatch_kw: float = Field(default=0.0, ge=0.0, le=200.0)
    grid_import_kw: float = Field(default=0.0, ge=0.0, le=250.0)
    flexible_curtailment_kw: float = Field(default=0.0, ge=0.0, le=180.0)
    repair_focus: RepairTarget = "none"
    operator_note: str = Field(default="", max_length=240)


class GridObservation(Observation):
    task_id: str
    task_title: str
    difficulty: str
    task_objective: str
    time_index: int = Field(..., ge=0)
    horizon_steps: int = Field(..., ge=1)
    current_interval_label: str
    next_interval_label: str | None = None
    risk_level: str
    weather_signal: str
    demand_kw: float = Field(..., ge=0.0)
    critical_load_kw: float = Field(..., ge=0.0)
    flexible_load_kw: float = Field(..., ge=0.0)
    renewable_kw: float = Field(..., ge=0.0)
    grid_import_limit_kw: float = Field(..., ge=0.0)
    diesel_limit_kw: float = Field(..., ge=0.0)
    price_per_kwh: float = Field(..., ge=0.0)
    carbon_intensity: float = Field(..., ge=0.0)
    battery_soc_kwh: float = Field(..., ge=0.0)
    battery_capacity_kwh: float = Field(..., ge=0.0)
    active_faults: list[FaultStatus] = Field(default_factory=list)
    kpis: KPIBundle = Field(default_factory=KPIBundle)
    last_action_summary: str = ""
    planning_hint: str = ""
    available_actions: list[str] = Field(default_factory=list)


class GridState(State):
    task_id: str = ""
    task_title: str = ""
    difficulty: str = ""
    current_interval_index: int = Field(default=0, ge=0)
    horizon_steps: int = Field(default=0, ge=0)
    battery_soc_kwh: float = Field(default=0.0, ge=0.0)
    battery_capacity_kwh: float = Field(default=0.0, ge=0.0)
    reserve_floor_kwh: float = Field(default=0.0, ge=0.0)
    cumulative_reward: float = 0.0
    operating_cost: float = 0.0
    emissions_kg: float = 0.0
    unmet_critical_kwh: float = 0.0
    unmet_total_kwh: float = 0.0
    resolved_faults: int = 0
    active_faults: list[FaultStatus] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    final_score: float = 0.0
    terminal_grade: dict[str, Any] = Field(default_factory=dict)
    last_action_summary: str = ""
    done_episode: bool = False

