from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "openenv.yaml",
    "Dockerfile",
    "inference.py",
    "models.py",
    "tasks.py",
    "graders.py",
    "server/app.py",
    "server/gridguardian_environment.py",
]

START_RE = re.compile(r"^\[START\] task=(\S+) env=(\S+) model=(\S+)$")
STEP_RE = re.compile(
    r"^\[STEP\] step=(\d+) action=(.+?) reward=(-?\d+\.\d{2}) done=(true|false) error=(.*)$"
)
END_RE = re.compile(
    r"^\[END\] success=(true|false) steps=(\d+) score=(-?\d+(?:\.\d+)?) rewards=([0-9.,-]*)$"
)


def _validate_inference_output(stdout: str) -> None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit("inference.py produced no stdout")

    current_steps = 0
    current_rewards: list[str] = []
    started = False

    for line in lines:
        if START_RE.match(line):
            if started:
                raise SystemExit("Encountered [START] before closing previous episode")
            started = True
            current_steps = 0
            current_rewards = []
            continue

        step_match = STEP_RE.match(line)
        if step_match:
            if not started:
                raise SystemExit("Encountered [STEP] before [START]")
            current_steps += 1
            reward = step_match.group(3)
            current_rewards.append(reward)
            continue

        end_match = END_RE.match(line)
        if end_match:
            if not started:
                raise SystemExit("Encountered [END] before [START]")
            ended_steps = int(end_match.group(2))
            score = float(end_match.group(3))
            rewards_blob = end_match.group(4)
            ended_rewards = [] if not rewards_blob else rewards_blob.split(",")
            if ended_steps != current_steps:
                raise SystemExit("Mismatch between emitted [STEP] count and [END] steps value")
            if ended_rewards != current_rewards:
                raise SystemExit("Mismatch between step rewards and [END] rewards list")
            if not 0.0 <= score <= 1.0:
                raise SystemExit("Final score must be in [0, 1]")
            started = False
            current_steps = 0
            current_rewards = []
            continue

        raise SystemExit(f"Unexpected stdout line from inference.py: {line}")

    if started:
        raise SystemExit("inference.py ended without a closing [END] line")


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not Path(path).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    result = subprocess.run([sys.executable, "verify_graders.py"], check=False)
    if result.returncode != 0:
        raise SystemExit("verify_graders.py failed")

    inference = subprocess.run(
        [sys.executable, "inference.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    if inference.returncode != 0:
        raise SystemExit("inference.py failed")
    _validate_inference_output(inference.stdout)

    print("Submission validation checks passed.")


if __name__ == "__main__":
    main()
