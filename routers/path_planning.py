"""
routers/path_planning.py
=========================
POST /api/plan-route

Runs A* rover path planning on a DEM-derived slope map (+ optional hazard map).
Accepts start/goal as (row, col) pixel coordinates and returns the full path
as waypoints and a GeoJSON LineString overlay.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from models.schemas import PlanRouteRequest, PlanRouteResponse, PathWaypoint
from services.raster_utils import open_raster
from services.terrain_analysis import analyse_dem
from services.path_planning import plan_route

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def _resolve_file(file_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{file_id}_*"))
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No uploaded file found for file_id='{file_id}'.",
        )
    return matches[0]


def _resolve_optional(file_id: str | None) -> Path | None:
    if not file_id or not file_id.strip():
        return None
    return _resolve_file(file_id.strip())


@router.post(
    "/plan-route",
    response_model=PlanRouteResponse,
    summary="Plan a safe rover route using A* path planning",
    description=(
        "Implements A* search on the terrain cost surface.\n\n"
        "**Cost function per step:**\n"
        "  `cost = distance + slope_penalty × norm_slope + hazard_penalty × norm_hazard`\n\n"
        "Returns the full waypoint list, total distance, estimated energy, "
        "safety score, and a GeoJSON LineString for the map overlay."
    ),
)
async def plan_route_endpoint(
    body: PlanRouteRequest,
) -> PlanRouteResponse:
    """Run A* path planning between two pixel coordinates on the terrain."""

    # ── Load DEM ───────────────────────────────────────────────────────────
    dem_path = _resolve_file(body.dem_file_id)
    try:
        dem_array, _, transform, crs = open_raster(dem_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Compute slope from DEM ─────────────────────────────────────────────
    try:
        dem_result = analyse_dem(dem_array, transform, crs)
    except Exception as exc:
        logger.exception("DEM analysis failed inside plan-route")
        raise HTTPException(status_code=500, detail=f"DEM analysis error: {exc}")

    slope_array = dem_result.slope_array

    # ── Optional hazard raster ─────────────────────────────────────────────
    hazard_array = None
    hazard_path  = _resolve_optional(body.hazard_file_id)
    if hazard_path is not None:
        try:
            hazard_array, _, _, _ = open_raster(hazard_path)
            if hazard_array.shape != slope_array.shape:
                min_r = min(slope_array.shape[0], hazard_array.shape[0])
                min_c = min(slope_array.shape[1], hazard_array.shape[1])
                slope_array  = slope_array[:min_r, :min_c]
                hazard_array = hazard_array[:min_r, :min_c]
                logger.warning(f"Hazard raster cropped to ({min_r},{min_c}) to match DEM.")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Hazard raster error: {exc}")

    # ── Validate start / goal bounds ───────────────────────────────────────
    rows, cols = slope_array.shape
    for coord, label in [(body.start, "start"), (body.goal, "goal")]:
        if not (0 <= coord.row < rows and 0 <= coord.col < cols):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Coordinate '{label}' ({coord.row},{coord.col}) is outside "
                    f"raster bounds ({rows}×{cols})."
                ),
            )

    # ── Run A* ────────────────────────────────────────────────────────────
    try:
        result = plan_route(
            slope_deg     = slope_array,
            transform     = transform,
            start         = (body.start.row, body.start.col),
            goal          = (body.goal.row,  body.goal.col),
            crs           = crs,
            hazard        = hazard_array,
            slope_penalty = body.slope_penalty,
            hazard_penalty = body.hazard_penalty,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("A* path planning failed")
        raise HTTPException(status_code=500, detail=f"Path planning error: {exc}")

    # ── Build response ─────────────────────────────────────────────────────
    waypoint_models = [
        PathWaypoint(
            step      = w.step,
            row       = w.row,
            col       = w.col,
            latitude  = w.latitude,
            longitude = w.longitude,
        )
        for w in result.path_waypoints
    ]

    return PlanRouteResponse(
        found             = result.found,
        path_length_steps = result.path_length_steps,
        total_distance_m  = result.total_distance_m,
        estimated_energy  = result.estimated_energy,
        safety_score      = result.safety_score,
        path_waypoints    = waypoint_models,
        path_geojson      = result.path_geojson,
    )
