"""
routers/ice.py
==============
POST /api/ice-probability

Accepts up to four optional raster file_ids (radar, temperature, psr, illumination),
computes the composite ice probability score for each pixel, and returns the
heatmap, top candidate regions, and summary statistics.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, status
from loguru import logger

from models.schemas import IceProbabilityResponse, IceCandidateRegion, JsonGrid
from services.raster_utils import open_raster
from services.ice_detection import compute_ice_probability

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def _resolve_optional(file_id: str | None):
    """Return Path for a file_id, or None if not provided."""
    if not file_id or not file_id.strip():
        return None
    matches = list(UPLOAD_DIR.glob(f"{file_id.strip()}_*"))
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No uploaded file found for file_id='{file_id}'.",
        )
    return matches[0]


def _load_optional(file_id: str | None):
    """Load a raster by file_id, return (array, transform, crs) or (None, None, None)."""
    path = _resolve_optional(file_id)
    if path is None:
        return None, None, None
    try:
        arr, _, tf, crs = open_raster(path)
        return arr, tf, crs
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Raster load error ({path.name}): {exc}")


@router.post(
    "/ice-probability",
    response_model=IceProbabilityResponse,
    summary="Compute subsurface ice probability from multi-source rasters",
    description=(
        "Computes:\n"
        "  IceScore = 0.4×radar + 0.3×psr + 0.2×temperature + 0.1×illumination\n\n"
        "Missing layers have their weights redistributed so the total always = 1.0. "
        "Returns a downsampled heatmap and the top candidate ice-rich regions."
    ),
)
async def ice_probability(
    radar_file_id:       str = Form(default="", description="file_id of Mini-RF CPR raster (optional)"),
    temperature_file_id: str = Form(default="", description="file_id of Diviner temperature raster (optional)"),
    psr_file_id:         str = Form(default="", description="file_id of PSR mask raster (optional)"),
    illumination_file_id: str = Form(default="", description="file_id of illumination raster (optional)"),
    n_candidates:        int = Form(default=10, ge=1, le=50),
) -> IceProbabilityResponse:
    """Compute pixel-wise ice probability from any combination of rasters."""

    # ── Load all provided rasters ──────────────────────────────────────────
    radar_arr,  radar_tf,  radar_crs  = _load_optional(radar_file_id)
    temp_arr,   temp_tf,   temp_crs   = _load_optional(temperature_file_id)
    psr_arr,    psr_tf,    psr_crs    = _load_optional(psr_file_id)
    illum_arr,  illum_tf,  illum_crs  = _load_optional(illumination_file_id)

    if all(a is None for a in [radar_arr, temp_arr, psr_arr, illum_arr]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one raster must be provided (radar, temperature, psr, or illumination).",
        )

    # Use transform + CRS from the first available raster
    transform = next(
        tf for tf in [radar_tf, psr_tf, temp_tf, illum_tf] if tf is not None
    )
    crs = next(
        c for c in [radar_crs, psr_crs, temp_crs, illum_crs] if c is not None
    )

    # ── Compute ice probability ────────────────────────────────────────────
    try:
        result = compute_ice_probability(
            transform    = transform,
            crs          = crs,
            radar        = radar_arr,
            psr          = psr_arr,
            temperature  = temp_arr,
            illumination = illum_arr,
            n_candidates = n_candidates,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Ice probability computation failed")
        raise HTTPException(status_code=500, detail=f"Ice computation error: {exc}")

    # ── Build response ─────────────────────────────────────────────────────
    candidate_models = [
        IceCandidateRegion(
            rank        = c.rank,
            row         = c.row,
            col         = c.col,
            latitude    = c.latitude,
            longitude   = c.longitude,
            ice_score   = c.ice_score,
            radar_score = c.radar_score,
            psr_score   = c.psr_score,
            temp_score  = c.temp_score,
            illum_score = c.illum_score,
        )
        for c in result.top_candidate_regions
    ]

    return IceProbabilityResponse(
        average_ice_probability   = result.average_ice_probability,
        max_ice_score             = result.max_ice_score,
        high_probability_area_pct = result.high_probability_area_pct,
        top_candidate_regions     = candidate_models,
        ice_heatmap               = JsonGrid(**result.ice_heatmap_grid),
        component_weights         = result.component_weights,
    )
