"""Application entry point.

Run with either:

    python -m app.main
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Nothing here opens an outbound connection.  There is no CDN, no web font, no analytics
and no telemetry: every asset is served from app/static, and the runtime works with the
network cable out (NFR-05 / T-20).
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.database.connection import connect
from app.database.migrations import database_is_seeded, init_database
from app.dependencies import class_breakdown_text, number, px, short_time, split_last, status_label
from app.routes import api_batches, api_inspections, api_materials, api_status, media, pages

BASE_DIR = Path(__file__).resolve().parent


def bootstrap_runtime_data(settings) -> None:
    """Prepare the writable data root.

    On a serverless host the deployment is read-only apart from /tmp, so the database
    that was seeded at build time is copied into /tmp on the first request of each cold
    start. Images stay where they are and are read from the bundled folder; only new
    ones written by the demo control land in /tmp. Every write is therefore per-instance
    and disappears when the instance does - which is fine for a demo and is the reason
    this deployment mode is not a station deployment. See DEPLOYMENT.md.
    """
    import shutil

    if not settings.is_serverless:
        return
    settings.runtime_data_dir.mkdir(parents=True, exist_ok=True)
    target = settings.db_file
    source = settings.bundled_db_file
    if not target.exists() and source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    bootstrap_runtime_data(settings)
    settings.ensure_dirs()
    init_database(settings.db_file)

    # Mock mode is expected to be demonstrable the moment it starts, so an empty
    # database seeds itself once.  Real mode never fabricates rows, and a serverless
    # host never seeds on demand: seeding generates hundreds of images and takes
    # minutes, far beyond any function timeout. There the database is built during
    # deployment by scripts/build_demo_bundle.py.
    if settings.is_mock and not settings.is_serverless:
        conn = connect(settings.db_file)
        try:
            needs_seed = not database_is_seeded(conn)
        finally:
            conn.close()
        if needs_seed:
            from app.database.seed import seed

            seed(settings)

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Industrial Surface-Defect Inspection System",
        description=(
            "Local interface for the VISION 404 surface-defect inspection pipeline. "
            "Read-only review interface: it reports what was found and shows the evidence. "
            "It does not issue accept or reject verdicts and it never displays a "
            "confidence percentage, because the model's scores are not calibrated."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["px"] = px
    templates.env.filters["number"] = number
    templates.env.filters["short_time"] = short_time
    templates.env.filters["status_label"] = status_label
    templates.env.filters["breakdown"] = class_breakdown_text
    templates.env.filters["split_last"] = split_last
    templates.env.globals["app_title"] = "Surface Defect Inspection"
    app.state.templates = templates
    app.state.settings = settings

    app.include_router(pages.router)
    app.include_router(api_inspections.router)
    app.include_router(api_batches.router)
    app.include_router(api_materials.router)
    app.include_router(api_status.router)
    app.include_router(media.router)

    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        if request.url.path.startswith(("/api", "/media")):
            return JSONResponse(
                status_code=404,
                content={"error": getattr(exc, "detail", "Not found"), "code": "not_found"},
            )
        return templates.TemplateResponse(
            request,
            "error.html",
            {
                "request": request,
                "code": 404,
                "headline": "PAGE NOT FOUND",
                "detail": f"There is nothing at {request.url.path}.",
                "nav_items": pages.NAV_ITEMS,
                "active": "",
                "settings": settings,
                "station": settings.station_id,
            },
            status_code=404,
        )

    @app.exception_handler(sqlite3.Error)
    async def db_error(request: Request, exc: sqlite3.Error):  # pragma: no cover - defensive
        return JSONResponse(
            status_code=503,
            content={
                "error": "The database could not be read.",
                "code": "database_error",
                "detail": str(exc),
            },
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "provider": settings.inspection_provider}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level="info",
    )


if __name__ == "__main__":
    main()
