"""
ISRO Lunar Mission Control — FastAPI Application Entry Point
============================================================
Initialises the FastAPI app, registers all routers, configures CORS for the
React frontend (localhost:5173 by default), and sets up structured logging.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# ── Router imports (we will add these as each module is created) ───────────
# from routers import upload, dem, landing_sites, ice, hazard, path_planning

# ── Application constants ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CACHE_DIR  = BASE_DIR / "data" / "cache"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Loguru configuration ───────────────────────────────────────────────────
logger.add(
    BASE_DIR / "logs" / "isro_backend.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
)

# ── FastAPI application ────────────────────────────────────────────────────
app = FastAPI(
    title="ISRO Lunar Mission Control — Backend API",
    description=(
        "Geospatial analysis backend for subsurface ice detection, "
        "safe landing site selection, and rover navigation in lunar polar regions."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── CORS ───────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving for uploaded rasters (optional debug access) ───────
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ── Routers ────────────────────────────────────────────────────────────────
# Each router is registered with the /api prefix so the frontend always hits
# /api/<endpoint>. We import them lazily here so that partial router
# implementations don't block the server from starting.

def _register_routers() -> None:
    """Import and register all API routers."""
    try:
        from routers.upload import router as upload_router
        app.include_router(upload_router, prefix="/api", tags=["Upload"])
        logger.info("Router registered: /api/upload-raster")
    except ImportError:
        logger.warning("upload router not yet available — skipping.")

    try:
        from routers.dem import router as dem_router
        app.include_router(dem_router, prefix="/api", tags=["DEM Analysis"])
        logger.info("Router registered: /api/analyze-dem")
    except ImportError:
        logger.warning("dem router not yet available — skipping.")

    try:
        from routers.landing_sites import router as landing_router
        app.include_router(landing_router, prefix="/api", tags=["Landing Sites"])
        logger.info("Router registered: /api/detect-landing-sites")
    except ImportError:
        logger.warning("landing_sites router not yet available — skipping.")

    try:
        from routers.ice import router as ice_router
        app.include_router(ice_router, prefix="/api", tags=["Ice Detection"])
        logger.info("Router registered: /api/ice-probability")
    except ImportError:
        logger.warning("ice router not yet available — skipping.")

    try:
        from routers.hazard import router as hazard_router
        app.include_router(hazard_router, prefix="/api", tags=["Hazard Map"])
        logger.info("Router registered: /api/hazard-map")
    except ImportError:
        logger.warning("hazard router not yet available — skipping.")

    try:
        from routers.path_planning import router as path_router
        app.include_router(path_router, prefix="/api", tags=["Path Planning"])
        logger.info("Router registered: /api/plan-route")
    except ImportError:
        logger.warning("path_planning router not yet available — skipping.")


_register_routers()


# ── Lifecycle events ───────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("ISRO Lunar Mission Control backend started.")
    logger.info(f"Upload directory : {UPLOAD_DIR}")
    logger.info(f"Cache directory  : {CACHE_DIR}")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("ISRO Lunar Mission Control backend shutting down.")


# ── Health check ───────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
async def health_check() -> dict:
    """
    Simple health-check endpoint.
    Returns 200 with backend status so the frontend can verify connectivity.
    """
    return {
        "status": "ok",
        "service": "ISRO Lunar Mission Control Backend",
        "version": app.version,
    }
