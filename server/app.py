from __future__ import annotations

import os

import uvicorn
from fastapi.responses import HTMLResponse

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
    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    def index() -> str:
        return """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>GridGuardian</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 40px; color: #17324d; background: #f3f8fb; }
              main { max-width: 760px; margin: 0 auto; background: white; padding: 32px; border-radius: 16px; box-shadow: 0 12px 36px rgba(0,0,0,0.08); }
              h1 { margin-top: 0; }
              a { color: #0b63ce; }
              code { background: #eef4fb; padding: 2px 6px; border-radius: 6px; }
              ul { line-height: 1.8; }
            </style>
          </head>
          <body>
            <main>
              <h1>GridGuardian</h1>
              <p>Climate-resilience microgrid environment for OpenEnv.</p>
              <p>This Space is serving the backend API successfully.</p>
              <ul>
                <li><a href="/docs">Interactive API docs</a></li>
                <li><a href="/openapi.json">OpenAPI schema</a></li>
                <li><a href="/health">Health check</a></li>
              </ul>
              <p>Use <code>inference.py</code> and the OpenEnv endpoints to interact with the environment.</p>
            </main>
          </body>
        </html>
        """

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
