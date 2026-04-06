from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server import Environment, create_app
    from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State
except ImportError:
    ActT = TypeVar("ActT", bound="Action")
    ObsT = TypeVar("ObsT", bound="Observation")
    StateT = TypeVar("StateT", bound="State")

    class Action(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = False
        reward: float | None = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        episode_id: Optional[str] = None
        step_count: int = 0

    class EnvironmentMetadata(BaseModel):
        name: str
        description: str
        version: str | None = None
        author: str | None = None
        documentation_url: str | None = None

    class Environment(ABC, Generic[ActT, ObsT, StateT]):
        SUPPORTS_CONCURRENT_SESSIONS: bool = False

        @abstractmethod
        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> ObsT:
            raise NotImplementedError

        @abstractmethod
        def step(
            self,
            action: ActT,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> ObsT:
            raise NotImplementedError

        @property
        @abstractmethod
        def state(self) -> StateT:
            raise NotImplementedError

        def close(self) -> None:
            return None

        def get_metadata(self) -> EnvironmentMetadata:
            return EnvironmentMetadata(
                name=self.__class__.__name__,
                description=f"{self.__class__.__name__} environment",
                version="0.0.0-local",
            )

    def create_app(*args: Any, **kwargs: Any) -> Any:
        try:
            from fastapi import FastAPI
        except ImportError:
            class DummyApp:
                openenv_missing = True

            return DummyApp()

        app = FastAPI(
            title="OpenEnv Compatibility Stub",
            description="Install openenv-core to enable the real server.",
        )

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "degraded", "detail": "openenv-core not installed"}

        return app

