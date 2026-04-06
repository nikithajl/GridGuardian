from __future__ import annotations

from typing import Any

import requests

from models import GridAction, GridObservation, GridState


class GridGuardianClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, task_id: str, seed: int | None = None, episode_id: str | None = None) -> GridObservation:
        payload: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        response = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        return GridObservation(**body["observation"], reward=body["reward"], done=body["done"])

    def step(self, action: GridAction) -> GridObservation:
        response = requests.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return GridObservation(**body["observation"], reward=body["reward"], done=body["done"])

    def state(self) -> GridState:
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return GridState(**response.json())

    def schema(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/schema", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def metadata(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/metadata", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

