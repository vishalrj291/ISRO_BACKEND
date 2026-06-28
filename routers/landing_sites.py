"""
routers/landing_sites.py
=========================
POST /api/detect-landing-sites

Runs the multi-criteria landing-site detection algorithm:
  slope < 7°  ∩  low roughness  ∩  (illumination > threshold if provided)

Returns top-5 ranked candidates with lat/lon, suitability score,
and a GeoJSON FeatureCollection for the frontend map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, status
from loguru import logger

from models.schemas import DetectLandingSitesResponse, LandingSite
from services.raster_utils import open_raster, array_to_json_grid
from services.terrain_analysis import (
    analyse_dem,
    detect_landing_sites,
    LandingSiteCandidate,
)
from services.raster_utils import sites_to_geojson

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


@router.post(
    "/detect-landing-sites",
    response_model=DetectLandingSitesResponse,
    summary="Detect top-N safe landing site candidates",
    description=(
        "Applies slope (<7°), roughness, and optional illumination constraints "
        "to identify the top-5 safest landing candidates.  "
        "Spatial suppression ensures candidates are geographically distributed."
    ),
)
async def detect_landing_sites_endpoint(
    dem_file_id:          str  = Form(...,         description="file_id of the DEM raster"),
    illumination_file_id: str  = Form(default="",  description="file_id of illumination raster (optional)"),
    n_sites:              int  = Form(default=5, ge=1, le=20),
    min_sep_pixels:       int  = Form(default=50, ge=1, le=500),
) -> DetectLandingSitesResponse:
    """Find the best safe landing candidates from DEM ± illumination rasters."""

    # ── Load DEM ───────────────────────────────────────────────────────────
    dem_path = _resolve_file(dem_file_id)
    try:
        dem_array, profile, transform, crs = open_raster(dem_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Optional illumination raster ───────────────────────────────────────
    illum_array = None
    if illumination_file_id.strip():
        illum_path = _resolve_file(illumination_file_id)
        try:
            illum_array, _, _, _ = open_raster(illum_path)
            # Ensure shape compatibility — resample if needed (simple crop)
            if illum_array.shape != dem_array.shape:
                min_r = min(dem_array.shape[0], illum_array.shape[0])
                min_c = min(dem_array.shape[1], illum_array.shape[1])
                dem_array   = dem_array[:min_r, :min_c]
                illum_array = illum_array[:min_r, :min_c]
                logger.warning(
                    "DEM and illumination shapes differ — cropped to "
                    f"({min_r}, {min_c}) for analysis."
                )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Illumination raster error: {exc}")

    # ── Run DEM analysis (computes slope + TRI) ────────────────────────────
    try:
        dem_result = analyse_dem(dem_array, transform, crs)
    except Exception as exc:
        logger.exception("DEM analysis failed inside detect-landing-sites")
        raise HTTPException(status_code=500, detail=f"DEM analysis error: {exc}")

    # ── Detect landing sites ───────────────────────────────────────────────
    try:
        candidates: list[LandingSiteCandidate] = detect_landing_sites(
            dem           = dem_array,
            transform     = transform,
            crs           = crs,
            illumination  = illum_array,
            slope         = dem_result.slope_array,
            tri           = dem_result.tri_array,
            n_sites       = n_sites,
            min_sep_pixels = min_sep_pixels,
        )
    except Exception as exc:
        logger.exception("Landing site detection failed")
        raise HTTPException(status_code=500, detail=f"Landing site detection error: {exc}")

    import math
    import numpy as np

    # ── Safe pixel count ───────────────────────────────────────────────────
    slope = dem_result.slope_array
    valid_slope = slope[~np.isnan(slope)]
    safe_pixel_count  = int((valid_slope < 7.0).sum())
    safe_area_percent = round(safe_pixel_count / valid_slope.size * 100.0, 2) if valid_slope.size else 0.0

    # ── Convert to Pydantic models ─────────────────────────────────────────
    site_dicts: list[dict] = []
    site_models: list[LandingSite] = []

    for c in candidates:
        illum_val = None if math.isnan(c.illumination) else c.illumination
        site_models.append(LandingSite(
            id               = c.id,
            row              = c.row,
            col              = c.col,
            latitude         = c.latitude,
            longitude        = c.longitude,
            slope_deg        = c.slope_deg,
            roughness_m      = c.roughness_m,
            illumination     = illum_val,
            suitability_score = c.suitability_score,
        ))
        site_dicts.append({
            "id":               c.id,
            "latitude":         c.latitude,
            "longitude":        c.longitude,
            "slope_deg":        c.slope_deg,
            "roughness_m":      c.roughness_m,
            "illumination":     illum_val,
            "suitability_score": c.suitability_score,
        })

    geojson = sites_to_geojson(site_dicts)

    logger.info(
        f"Landing sites detected | found={len(candidates)} "
        f"| safe_area={safe_area_percent:.1f}%"
    )

    return DetectLandingSitesResponse(
        file_id           = dem_file_id,
        candidates        = site_models,
        total_safe_pixels = safe_pixel_count,
        safe_area_percent = safe_area_percent,
        geojson           = geojson,
    )
