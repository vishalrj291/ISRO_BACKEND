"""
routers/dem.py
==============
POST /api/analyze-dem

Loads a previously uploaded DEM GeoTIFF by file_id, runs the full terrain
analysis pipeline (slope + TRI), and returns aggregated statistics plus
downsampled heatmaps for the frontend.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, status
from loguru import logger

from models.schemas import DEMAnalysisResponse, RasterStats, JsonGrid
from services.raster_utils import open_raster, array_to_json_grid
from services.terrain_analysis import analyse_dem

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def _resolve_file(file_id: str) -> Path:
    """
    Locate the file on disk whose name starts with the given file_id.
    Raises 404 if not found.
    """
    matches = list(UPLOAD_DIR.glob(f"{file_id}_*"))
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No uploaded file found for file_id='{file_id}'. "
                   f"Please upload the raster first via POST /api/upload-raster.",
        )
    return matches[0]


@router.post(
    "/analyze-dem",
    response_model=DEMAnalysisResponse,
    summary="Run full terrain analysis on a DEM raster",
    description=(
        "Given a `file_id` returned by the upload endpoint, loads the DEM, "
        "computes slope (Horn's method) and Terrain Ruggedness Index (TRI), "
        "and returns per-pixel statistics plus downsampled heatmaps."
    ),
)
async def analyze_dem(
    file_id: str = Form(..., description="file_id returned by /api/upload-raster"),
    band:    int = Form(default=1, ge=1, description="Raster band to read (1-indexed)"),
) -> DEMAnalysisResponse:
    """Compute slope, roughness, and safe-area statistics from a DEM."""

    # ── Resolve file path ──────────────────────────────────────────────────
    dem_path = _resolve_file(file_id)

    # ── Load raster ────────────────────────────────────────────────────────
    try:
        dem_array, profile, transform, crs = open_raster(dem_path, band=band)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(f"DEM load failed for file_id={file_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # ── Run analysis pipeline ──────────────────────────────────────────────
    try:
        result = analyse_dem(dem_array, transform, crs)
    except Exception as exc:
        logger.exception(f"DEM analysis failed for file_id={file_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terrain analysis error: {exc}",
        )

    # ── Serialise heatmaps ─────────────────────────────────────────────────
    slope_grid_dict = array_to_json_grid(result.slope_array, max_pixels=256)
    tri_grid_dict   = array_to_json_grid(result.tri_array,   max_pixels=256)

    slope_grid = JsonGrid(**slope_grid_dict)
    tri_grid   = JsonGrid(**tri_grid_dict)

    # ── Build response ─────────────────────────────────────────────────────
    return DEMAnalysisResponse(
        file_id             = file_id,
        average_slope_deg   = round(result.average_slope_deg,   4),
        maximum_slope_deg   = round(result.maximum_slope_deg,   4),
        std_slope_deg       = round(result.std_slope_deg,       4),
        safe_area_percent   = result.safe_area_percent,
        average_elevation_m = round(result.average_elevation_m, 4),
        max_elevation_m     = round(result.max_elevation_m,     4),
        min_elevation_m     = round(result.min_elevation_m,     4),
        elevation_range_m   = round(result.elevation_range_m,   4),
        average_roughness_m = round(result.average_roughness_m, 4),
        elevation_stats     = RasterStats(**result.elevation_stats),
        slope_stats         = RasterStats(**result.slope_stats),
        slope_grid          = slope_grid,
        tri_grid            = tri_grid,
    )
