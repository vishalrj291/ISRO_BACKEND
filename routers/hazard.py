"""
routers/hazard.py
==================
POST /api/hazard-map

Detects high-slope regions, rough terrain, and shadow hazards from DEM
(+ optional PSR mask), and returns per-pixel hazard classifications.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, status
from loguru import logger

from models.schemas import HazardMapResponse, JsonGrid
from services.raster_utils import open_raster, array_to_json_grid
from services.terrain_analysis import analyse_dem, generate_hazard_map

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
    "/hazard-map",
    response_model=HazardMapResponse,
    summary="Generate a multi-factor hazard map from terrain data",
    description=(
        "Classifies each pixel as LOW / MEDIUM / HIGH hazard based on:\n"
        "- Slope (weight 0.55 or 0.65 if no shadow mask)\n"
        "- Terrain Ruggedness Index (weight 0.30 or 0.35)\n"
        "- Permanent shadow mask (weight 0.15, optional)\n\n"
        "Returns normalised [0,1] hazard heatmap, risk classification grid, "
        "and area percentages for each risk class."
    ),
)
async def hazard_map(
    dem_file_id:    str = Form(..., description="file_id of the DEM raster"),
    shadow_file_id: str = Form(default="", description="file_id of the PSR/shadow mask raster (optional)"),
) -> HazardMapResponse:
    """Compute terrain hazard map from slope + roughness + optional shadow mask."""

    # ── Load DEM ───────────────────────────────────────────────────────────
    dem_path = _resolve_file(dem_file_id)
    try:
        dem_array, _, transform, crs = open_raster(dem_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Optional shadow / PSR mask ─────────────────────────────────────────
    shadow_array = None
    shadow_path  = _resolve_optional(shadow_file_id)
    if shadow_path is not None:
        try:
            shadow_array, _, _, _ = open_raster(shadow_path)
            if shadow_array.shape != dem_array.shape:
                min_r = min(dem_array.shape[0], shadow_array.shape[0])
                min_c = min(dem_array.shape[1], shadow_array.shape[1])
                dem_array    = dem_array[:min_r, :min_c]
                shadow_array = shadow_array[:min_r, :min_c]
                logger.warning(f"Shadow mask cropped to {min_r}×{min_c} to match DEM.")
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Shadow mask error: {exc}")

    # ── DEM analysis → slope + TRI ─────────────────────────────────────────
    try:
        dem_result = analyse_dem(dem_array, transform, crs)
    except Exception as exc:
        logger.exception("DEM analysis failed inside hazard-map")
        raise HTTPException(status_code=500, detail=f"DEM analysis error: {exc}")

    # ── Hazard map generation ──────────────────────────────────────────────
    try:
        hazard_result = generate_hazard_map(
            slope       = dem_result.slope_array,
            tri         = dem_result.tri_array,
            shadow_mask = shadow_array,
        )
    except Exception as exc:
        logger.exception("Hazard map generation failed")
        raise HTTPException(status_code=500, detail=f"Hazard map error: {exc}")

    # ── Serialise heatmaps ─────────────────────────────────────────────────
    import numpy as np
    hazard_grid_dict    = array_to_json_grid(hazard_result.hazard_array,                         max_pixels=256)
    risk_class_raw      = hazard_result.risk_class_array.astype(np.float32)
    risk_class_raw[risk_class_raw == -1] = float("nan")
    risk_class_grid_dict = array_to_json_grid(risk_class_raw, max_pixels=256)

    return HazardMapResponse(
        hazard_score         = hazard_result.hazard_score,
        high_risk_area_pct   = hazard_result.high_risk_area_pct,
        medium_risk_area_pct = hazard_result.medium_risk_area_pct,
        low_risk_area_pct    = hazard_result.low_risk_area_pct,
        hazard_grid          = JsonGrid(**hazard_grid_dict),
        risk_class_grid      = JsonGrid(**risk_class_grid_dict),
    )
