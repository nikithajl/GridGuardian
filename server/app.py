from __future__ import annotations

import os

import uvicorn

from compat import create_app
from models import GridAction, GridObservation
from .gridguardian_environment import GridGuardianEnvironment

app = create_app(
    GridGuardianEnvironment,
    GridAction,
    GridObservation,
    env_name="gridguardian",
    max_concurrent_envs=8,
)

if hasattr(app, "get"):
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "env": "gridguardian"}


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
