---
title: GridGuardian
emoji: "⚡"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
- openenv
- sustainability
- operations
- climate
- reinforcement-learning
license: mit
pinned: false
---

# GridGuardian

GridGuardian is an OpenEnv environment for operating a climate-resilience microgrid during disruptive real-world events. The agent dispatches storage, grid imports, diesel backup, flexible demand curtailment, and repair priorities to protect clinics, shelters, pumps, and emergency infrastructure.

## Environment Summary

- Domain: climate and infrastructure operations
- Format: deterministic, multi-step decision environment
- Output score: strictly constrained to the range `(0.0, 1.0)`
- Root inference script: `inference.py`
- OpenEnv manifest: `openenv.yaml`

## Task Catalog

This submission includes 6 tasks:

1. `heatwave_hospital_cooling` - Heatwave Hospital and Cooling Center (`easy`)
   This task is about keeping a hospital wing and a public cooling center powered during a heatwave. Demand rises sharply in the evening, so the agent has to save battery for the hardest hours instead of spending it too early.
2. `monsoon_shelter_power` - Monsoon Shelter and Pharmacy Power (`easy`)
   This one is about supporting a storm shelter and pharmacy during a monsoon day. The idea is to use the easier daytime window to prepare for the evening, when people need shelter services, refrigeration, and backup power more urgently.
3. `wildfire_smoke_clinic` - Wildfire Smoke Clinic and Clean-Air Shelter (`medium`)
   This task simulates wildfire smoke reducing solar generation while a battery cooling fault limits battery performance. The agent needs to repair the battery issue early and still protect a clinic and a clean-air shelter as conditions worsen.
4. `flood_pumps_and_shelters` - Flood Pumps, Shelter, and Clinic Coordination (`medium`)
   This is a flood-response task where drainage pumps, shelters, and a clinic all need power, but the grid connection is partially damaged. The agent must prioritize pump operations and repair the feeder quickly so the system can survive the evening flood peak.
5. `post_cyclone_emergency_power` - Post-Cyclone Emergency Power (`hard`)
   This is a harder disaster-response scenario after a cyclone. Shelters, water systems, and emergency services all depend on the microgrid, while both the feeder and diesel system are degraded, so the agent has to manage repairs and scarce energy very carefully.
6. `cold_snap_warming_center` - Cold Snap Warming Center Recovery (`hard`)
   This task is about a severe cold snap after a partial blackstart. Heating demand is very high, and the agent must keep a clinic and warming center online while recovering two damaged assets early enough to survive the evening heating peak.

Each task has:

- a deterministic scenario definition
- a typed action and observation schema
- an episode horizon
- a deterministic terminal grader

## Action Schema

`GridAction` fields:

- `battery_dispatch_kw`
- `diesel_dispatch_kw`
- `grid_import_kw`
- `flexible_curtailment_kw`
- `repair_focus`
- `operator_note`

## Observation Schema

`GridObservation` includes:

- task metadata
- current interval and horizon position
- demand, critical load, flexible load, and renewable output
- grid and diesel limits
- price and carbon intensity
- battery state of charge
- active faults and repair progress
- cumulative KPI bundle

## Repository Layout

- `models.py`: action, observation, and state models
- `tasks.py`: task registry and scenario definitions
- `graders.py`: deterministic grading logic
- `server/gridguardian_environment.py`: environment simulation
- `server/app.py`: OpenEnv/FastAPI app entrypoint
- `baseline.py`: baseline planners and local evaluation helpers
- `planner.py`: OpenAI-client candidate selector for inference, with deterministic fallback
- `inference.py`: required root-level inference script
- `verify_graders.py`: local grader verification
- `validate_submission.py`: local submission-format validation

## Baseline Scores

Deterministic baseline scores with the built-in beam-search controller:

- `heatwave_hospital_cooling`: `0.9605`
- `monsoon_shelter_power`: `0.9735`
- `wildfire_smoke_clinic`: `0.9526`
- `flood_pumps_and_shelters`: `0.9559`
- `post_cyclone_emergency_power`: `0.9203`
- `cold_snap_warming_center`: `0.9558`

When `HF_TOKEN` is provided, `inference.py` uses the OpenAI client to rank candidate control actions at each step. If no LLM credentials are available, it falls back to the same deterministic beam-search policy used for the scores above.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run local validation:

```bash
python verify_graders.py
python validate_submission.py
```

Run the inference script:

```bash
python inference.py
```

Run the server:

```bash
python -m server.app
```

## Environment Variables

The inference workflow supports:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `LOCAL_IMAGE_NAME`
- `OPENENV_BASE_URL`
- `SPACE_URL`
- `ENV_BASE_URL`

If `HF_TOKEN` is not provided, inference falls back to a deterministic local policy.

## Validation Notes

- `inference.py` emits `[START]`, `[STEP]`, and `[END]` lines in the required single-line format.
- `validate_submission.py` checks required files, grader score range, and inference stdout format.
- All task scores are constrained to `(0.0, 1.0)`.
