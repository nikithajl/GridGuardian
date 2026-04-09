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
- Output score: normalized to the range `[0.0, 1.0]`
- Root inference script: `inference.py`
- OpenEnv manifest: `openenv.yaml`

## Task Catalog

This submission includes 6 tasks:

1. `heatwave_peak_shaving` (`easy`)
2. `monsoon_shelter_rebalancing` (`easy`)
3. `smoke_event_resilience` (`medium`)
4. `flood_pump_coordination` (`medium`)
5. `cyclone_islanding` (`hard`)
6. `cold_snap_blackstart` (`hard`)

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
- `planner.py`: optional OpenAI-client note generator for inference
- `inference.py`: required root-level inference script
- `verify_graders.py`: local grader verification
- `validate_submission.py`: local submission-format validation

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
- All task scores are constrained to `[0.0, 1.0]`.
