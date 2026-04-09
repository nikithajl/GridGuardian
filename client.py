from __future__ import annotations

import os
from typing import Any

import requests

from models import GridAction, GridObservation, GridState

try:
    from openenv.core.generic_client import GenericEnvClient
except ImportError:
    GenericEnvClient = None


class GridGuardianClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(os.getenv("GRIDGUARDIAN_HTTP_TIMEOUT", str(timeout)))
        self._sync_client = None

    def _ensure_sync_client(self):
        if self._sync_client is None:
            if GenericEnvClient is None:
                raise RuntimeError("openenv-core GenericEnvClient is not available")
            self._sync_client = GenericEnvClient(
                base_url=self.base_url,
                connect_timeout_s=self.timeout,
                message_timeout_s=self.timeout,
            ).sync()
            self._sync_client.connect()
        return self._sync_client

    def reset(self, task_id: str, seed: int | None = None, episode_id: str | None = None) -> GridObservation:
        payload: dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        result = self._ensure_sync_client().reset(**payload)
        observation = result.observation or {}
        return GridObservation(**observation, reward=result.reward, done=result.done)

    def step(self, action: GridAction) -> GridObservation:
        result = self._ensure_sync_client().step(action.model_dump())
        observation = result.observation or {}
        return GridObservation(**observation, reward=result.reward, done=result.done)

    def state(self) -> GridState:
        state = self._ensure_sync_client().state()
        return GridState(**state)

    def schema(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/schema", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def metadata(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/metadata", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    def __enter__(self) -> "GridGuardianClient":
        self._ensure_sync_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
