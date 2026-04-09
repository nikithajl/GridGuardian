from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class IntervalProfile:
    label: str
    demand_kw: float
    renewable_kw: float
    critical_load_kw: float
    grid_import_limit_kw: float
    price_per_kwh: float
    carbon_intensity: float
    diesel_limit_kw: float
    risk_level: Literal["low", "medium", "high", "critical"]
    weather_signal: str
    briefing: str


@dataclass(frozen=True)
class FaultSpec:
    name: Literal["battery_cooling", "grid_tie", "diesel_filter"]
    description: str
    affected_system: str
    severity: float
    repair_steps_required: int


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    summary: str
    objective: str
    success_criteria: str
    battery_capacity_kwh: float
    initial_battery_soc_kwh: float
    reserve_floor_kwh: float
    battery_max_charge_kw: float
    battery_max_discharge_kw: float
    diesel_variable_cost: float
    diesel_emissions_factor: float
    cost_budget: float
    emissions_budget: float
    total_unserved_budget: float
    critical_unserved_budget: float
    grading_weights: dict[str, float] = field(default_factory=dict)
    intervals: tuple[IntervalProfile, ...] = field(default_factory=tuple)
    initial_faults: tuple[FaultSpec, ...] = field(default_factory=tuple)

    @property
    def horizon_steps(self) -> int:
        return len(self.intervals)


EASY_WEIGHTS = {
    "critical_service": 0.46,
    "total_service": 0.25,
    "cost": 0.07,
    "emissions": 0.04,
    "repair": 0.00,
    "reserve": 0.18,
}

MEDIUM_WEIGHTS = {
    "critical_service": 0.47,
    "total_service": 0.20,
    "cost": 0.06,
    "emissions": 0.04,
    "repair": 0.10,
    "reserve": 0.13,
}

HARD_WEIGHTS = {
    "critical_service": 0.48,
    "total_service": 0.17,
    "cost": 0.05,
    "emissions": 0.03,
    "repair": 0.14,
    "reserve": 0.13,
}


TASKS: dict[str, TaskSpec] = {
    "heatwave_hospital_cooling": TaskSpec(
        task_id="heatwave_hospital_cooling",
        title="Heatwave Hospital and Cooling Center",
        difficulty="easy",
        summary="Protect a hospital wing and cooling center during an evening heatwave.",
        objective="Keep critical services online, shift load out of the peak, and avoid burning through the battery too early.",
        success_criteria="Critical demand remains online, flexible demand disruption stays limited, and the operator closes with reserve intact.",
        battery_capacity_kwh=240.0,
        initial_battery_soc_kwh=170.0,
        reserve_floor_kwh=45.0,
        battery_max_charge_kw=120.0,
        battery_max_discharge_kw=120.0,
        diesel_variable_cost=0.62,
        diesel_emissions_factor=0.78,
        cost_budget=1100.0,
        emissions_budget=1200.0,
        total_unserved_budget=100.0,
        critical_unserved_budget=15.0,
        grading_weights=dict(EASY_WEIGHTS),
        intervals=(
            IntervalProfile("06:00", 155.0, 60.0, 90.0, 130.0, 0.18, 0.30, 110.0, "low", "Mild morning temperatures.", "Cheap imports can preserve storage for later."),
            IntervalProfile("10:00", 180.0, 120.0, 90.0, 130.0, 0.16, 0.27, 110.0, "low", "Solar output is strong.", "This is the easiest window to top up reserve."),
            IntervalProfile("14:00", 230.0, 140.0, 100.0, 120.0, 0.22, 0.25, 110.0, "medium", "Heat stress is building.", "Start preserving headroom for the evening peak."),
            IntervalProfile("18:00", 280.0, 50.0, 120.0, 115.0, 0.41, 0.42, 120.0, "high", "Solar ramps down as cooling demand spikes.", "Use storage intelligently rather than all at once."),
            IntervalProfile("20:00", 300.0, 15.0, 130.0, 105.0, 0.52, 0.50, 120.0, "critical", "The peak interval arrives with high prices.", "Serve the cooling center without exhausting the system."),
            IntervalProfile("22:00", 245.0, 0.0, 110.0, 105.0, 0.36, 0.44, 120.0, "medium", "Demand tapers but is still elevated.", "Finish the episode without an avoidable shortfall."),
        ),
    ),
    "monsoon_shelter_power": TaskSpec(
        task_id="monsoon_shelter_power",
        title="Monsoon Shelter and Pharmacy Power",
        difficulty="easy",
        summary="Manage a shelter and pharmacy microgrid through a humid monsoon evening.",
        objective="Use midday renewable surplus to prepare for evening shelter demand while keeping community disruption minimal.",
        success_criteria="Critical shelter services remain online and the microgrid closes with healthy reserve.",
        battery_capacity_kwh=240.0,
        initial_battery_soc_kwh=175.0,
        reserve_floor_kwh=45.0,
        battery_max_charge_kw=120.0,
        battery_max_discharge_kw=120.0,
        diesel_variable_cost=0.60,
        diesel_emissions_factor=0.75,
        cost_budget=1050.0,
        emissions_budget=1150.0,
        total_unserved_budget=95.0,
        critical_unserved_budget=15.0,
        grading_weights=dict(EASY_WEIGHTS),
        intervals=(
            IntervalProfile("06:00", 145.0, 35.0, 80.0, 130.0, 0.20, 0.32, 100.0, "low", "Humidity is rising but demand is manageable.", "Preserve cheap energy for the evening."),
            IntervalProfile("10:00", 165.0, 100.0, 85.0, 130.0, 0.17, 0.28, 100.0, "low", "Solar output improves through the morning.", "Charging now is cheaper than reacting later."),
            IntervalProfile("14:00", 190.0, 125.0, 90.0, 120.0, 0.19, 0.26, 100.0, "low", "A stable midday window arrives.", "This is your best rebalancing interval."),
            IntervalProfile("18:00", 225.0, 55.0, 105.0, 120.0, 0.36, 0.39, 110.0, "medium", "Shelter occupancy and refrigeration load increase.", "Use stored energy to avoid unnecessary diesel."),
            IntervalProfile("20:00", 245.0, 20.0, 115.0, 115.0, 0.44, 0.46, 110.0, "high", "Evening humidity pushes cooling demand higher.", "Keep the pharmacy and shelter fully supported."),
            IntervalProfile("22:00", 205.0, 0.0, 95.0, 115.0, 0.31, 0.40, 110.0, "medium", "Demand softens but support services remain active.", "Close the day efficiently without wasting reserve."),
        ),
    ),
    "wildfire_smoke_clinic": TaskSpec(
        task_id="wildfire_smoke_clinic",
        title="Wildfire Smoke Clinic and Clean-Air Shelter",
        difficulty="medium",
        summary="Wildfire smoke cuts solar output while a battery cooling fault limits throughput.",
        objective="Repair the battery bottleneck early and protect a clinic and clean-air shelter during the smoke surge.",
        success_criteria="Repair happens early, critical service stays online, and the operator minimizes avoidable diesel burn.",
        battery_capacity_kwh=270.0,
        initial_battery_soc_kwh=195.0,
        reserve_floor_kwh=50.0,
        battery_max_charge_kw=120.0,
        battery_max_discharge_kw=120.0,
        diesel_variable_cost=0.66,
        diesel_emissions_factor=0.80,
        cost_budget=1250.0,
        emissions_budget=1400.0,
        total_unserved_budget=120.0,
        critical_unserved_budget=20.0,
        grading_weights=dict(MEDIUM_WEIGHTS),
        intervals=(
            IntervalProfile("07:00", 170.0, 70.0, 95.0, 120.0, 0.24, 0.34, 110.0, "medium", "Smoke begins reducing rooftop solar.", "Repairing storage early unlocks later flexibility."),
            IntervalProfile("11:00", 205.0, 65.0, 100.0, 120.0, 0.28, 0.35, 110.0, "medium", "Shelter air handling ramps up.", "Stay ahead of the late-day surge."),
            IntervalProfile("15:00", 245.0, 35.0, 115.0, 115.0, 0.44, 0.47, 115.0, "high", "Dense smoke suppresses solar output.", "A repaired battery can shave the afternoon ramp."),
            IntervalProfile("17:00", 285.0, 20.0, 135.0, 110.0, 0.56, 0.53, 120.0, "high", "Air filtration drives a strong peak.", "Protect the clinic before anything flexible."),
            IntervalProfile("19:00", 300.0, 10.0, 150.0, 100.0, 0.62, 0.58, 125.0, "critical", "The dirtiest and most expensive interval arrives.", "This is where poor reserve management gets punished."),
            IntervalProfile("22:00", 230.0, 0.0, 120.0, 110.0, 0.40, 0.45, 120.0, "medium", "Smoke persists overnight but demand recedes.", "End the episode without an unnecessary generator run."),
        ),
        initial_faults=(
            FaultSpec("battery_cooling", "Battery thermal control is degraded, limiting throughput.", "battery", 0.60, 2),
        ),
    ),
    "flood_pumps_and_shelters": TaskSpec(
        task_id="flood_pumps_and_shelters",
        title="Flood Pumps, Shelter, and Clinic Coordination",
        difficulty="medium",
        summary="Coordinate pumping stations, shelters, and a clinic while a grid-tie fault derates imports.",
        objective="Repair the feeder quickly, keep pumps online through the evening surge, and preserve storage for the highest-risk interval.",
        success_criteria="The feeder is restored early and critical pumping and shelter services remain online.",
        battery_capacity_kwh=280.0,
        initial_battery_soc_kwh=205.0,
        reserve_floor_kwh=50.0,
        battery_max_charge_kw=120.0,
        battery_max_discharge_kw=120.0,
        diesel_variable_cost=0.64,
        diesel_emissions_factor=0.79,
        cost_budget=1280.0,
        emissions_budget=1425.0,
        total_unserved_budget=125.0,
        critical_unserved_budget=20.0,
        grading_weights=dict(MEDIUM_WEIGHTS),
        intervals=(
            IntervalProfile("05:00", 180.0, 25.0, 100.0, 120.0, 0.26, 0.34, 110.0, "medium", "Overnight pumping demand starts to rise.", "The grid-tie repair is worth doing early."),
            IntervalProfile("09:00", 210.0, 65.0, 115.0, 115.0, 0.29, 0.36, 115.0, "medium", "Drainage crews expand operations.", "Use the morning to avoid a compressed evening response."),
            IntervalProfile("13:00", 235.0, 75.0, 125.0, 105.0, 0.35, 0.39, 115.0, "medium", "Midday renewables help, but pumps stay active.", "Keep reserve available for the flood crest."),
            IntervalProfile("17:00", 270.0, 40.0, 150.0, 90.0, 0.54, 0.50, 120.0, "high", "The watershed reaches a dangerous level.", "Support pumps first and trim only flexible demand."),
            IntervalProfile("20:00", 285.0, 10.0, 160.0, 85.0, 0.63, 0.58, 125.0, "critical", "Evening flooding peaks as prices surge.", "Critical water removal and shelters take priority."),
            IntervalProfile("23:00", 220.0, 0.0, 130.0, 95.0, 0.41, 0.45, 120.0, "medium", "Conditions stabilize but pumping is still required.", "Finish cleanly without avoidable emissions."),
        ),
        initial_faults=(
            FaultSpec("grid_tie", "A feeder fault reduces import capacity until cleared.", "grid", 0.65, 1),
        ),
    ),
    "post_cyclone_emergency_power": TaskSpec(
        task_id="post_cyclone_emergency_power",
        title="Post-Cyclone Emergency Power",
        difficulty="hard",
        summary="A coastal microgrid is under post-cyclone stress with feeder damage and a degraded diesel generator.",
        objective="Restore import capacity quickly, ration diesel sensibly, and carry shelters and pumps through the evening emergency peak.",
        success_criteria="Repairs are prioritized correctly, critical services remain online, and avoidable curtailment stays low.",
        battery_capacity_kwh=310.0,
        initial_battery_soc_kwh=235.0,
        reserve_floor_kwh=12.0,
        battery_max_charge_kw=120.0,
        battery_max_discharge_kw=120.0,
        diesel_variable_cost=0.70,
        diesel_emissions_factor=0.83,
        cost_budget=1380.0,
        emissions_budget=1550.0,
        total_unserved_budget=145.0,
        critical_unserved_budget=22.0,
        grading_weights=dict(HARD_WEIGHTS),
        intervals=(
            IntervalProfile("05:00", 190.0, 20.0, 110.0, 100.0, 0.33, 0.41, 120.0, "high", "The cyclone has passed, but the feeder remains unstable.", "Repairing the grid tie early pays off all day."),
            IntervalProfile("09:00", 220.0, 50.0, 125.0, 110.0, 0.38, 0.43, 130.0, "high", "Shelter kitchens and water pumps come online.", "Build enough margin for the late peak."),
            IntervalProfile("13:00", 250.0, 40.0, 140.0, 95.0, 0.52, 0.55, 140.0, "high", "Solar recovery is limited by cloud bands.", "Do not let the diesel fault linger into the evening."),
            IntervalProfile("17:00", 290.0, 15.0, 170.0, 80.0, 0.78, 0.70, 150.0, "critical", "Emergency services and shelters hit their peak.", "Use every repaired asset intelligently."),
            IntervalProfile("20:00", 315.0, 5.0, 190.0, 75.0, 0.86, 0.73, 160.0, "critical", "The hardest post-storm interval arrives after dark.", "Lives and water systems come before all flexible demand."),
            IntervalProfile("23:00", 245.0, 0.0, 150.0, 85.0, 0.55, 0.58, 150.0, "high", "Demand eases as the response stabilizes.", "Close out the crisis without wasting remaining reserve."),
        ),
        initial_faults=(
            FaultSpec("grid_tie", "Storm debris damaged the feeder, reducing import capacity.", "grid", 0.55, 2),
            FaultSpec("diesel_filter", "Generator intake filters are clogged, derating dispatch capacity.", "diesel", 0.75, 1),
        ),
    ),
    "cold_snap_warming_center": TaskSpec(
        task_id="cold_snap_warming_center",
        title="Cold Snap Warming Center Recovery",
        difficulty="hard",
        summary="A cold snap drives heating demand while the site is recovering from a partial blackstart with two degraded assets.",
        objective="Recover storage and diesel flexibility early enough to cover the heating peak while protecting a clinic and warming center.",
        success_criteria="Repairs are sequenced well, critical heat stays online, and the operator avoids a late-episode collapse.",
        battery_capacity_kwh=320.0,
        initial_battery_soc_kwh=240.0,
        reserve_floor_kwh=60.0,
        battery_max_charge_kw=120.0,
        battery_max_discharge_kw=120.0,
        diesel_variable_cost=0.69,
        diesel_emissions_factor=0.82,
        cost_budget=1425.0,
        emissions_budget=1600.0,
        total_unserved_budget=150.0,
        critical_unserved_budget=22.0,
        grading_weights=dict(HARD_WEIGHTS),
        intervals=(
            IntervalProfile("06:00", 210.0, 15.0, 120.0, 120.0, 0.31, 0.38, 120.0, "high", "Heating loads are already elevated before sunrise.", "Use the morning to repair flexibility before the cold peak."),
            IntervalProfile("09:00", 225.0, 35.0, 125.0, 120.0, 0.30, 0.36, 125.0, "medium", "A brief daytime lull appears.", "This is the easiest interval for charging and repairs."),
            IntervalProfile("13:00", 250.0, 45.0, 135.0, 110.0, 0.42, 0.43, 130.0, "high", "Cloud cover limits solar recovery.", "Preserve enough capacity for the evening heating surge."),
            IntervalProfile("17:00", 295.0, 15.0, 170.0, 95.0, 0.66, 0.60, 145.0, "critical", "Commuters return as outside temperatures fall quickly.", "Keep the clinic and warming center whole."),
            IntervalProfile("20:00", 320.0, 0.0, 195.0, 90.0, 0.74, 0.66, 155.0, "critical", "This is the hardest heating interval.", "A poor repair sequence will show up here immediately."),
            IntervalProfile("23:00", 255.0, 0.0, 150.0, 100.0, 0.49, 0.48, 145.0, "high", "Demand remains high into the night.", "Close out safely without a reserve cliff."),
        ),
        initial_faults=(
            FaultSpec("battery_cooling", "Cold-weather battery thermal controls are degraded.", "battery", 0.70, 1),
            FaultSpec("diesel_filter", "Generator fuel filtration is impaired by cold-weather fouling.", "diesel", 0.78, 1),
        ),
    ),
}

DEFAULT_TASK_ID = "heatwave_hospital_cooling"


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {', '.join(TASKS)}")
    return TASKS[task_id]


def list_tasks() -> list[TaskSpec]:
    return [TASKS[key] for key in TASKS]
