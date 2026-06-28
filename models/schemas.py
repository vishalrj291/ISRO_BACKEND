"""
models/schemas.py
=================
Pydantic v2 request and response schemas for every API endpoint.

All response models are strict (no arbitrary extras allowed) so the frontend
always receives a predictable, documented shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Shared sub-models
# ─────────────────────────────────────────────────────────────────────────────

class RasterStats(BaseModel):
    """Basic descriptive statistics over valid pixels in a raster band."""
    min:                 float | None
    max:                 float | None
    mean:                float | None
    median:              float | None
    std:                 float | None
    valid_pixel_count:   int
    total_pixel_count:   int


class JsonGrid(BaseModel):
    """
    A 2-D array serialised for JSON transport.
    Downsampled to ≤ 256×256 for the frontend heatmap renderer.
    None values represent nodata / NaN pixels.
    """
    rows: int
    cols: int
    data: list[list[float | None]]


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

class UploadRasterResponse(BaseModel):
    """Response from POST /api/upload-raster."""
    file_id:      str = Field(..., description="UUID that uniquely identifies this upload.")
    original_name: str
    file_path:    str = Field(..., description="Server-side path for subsequent analysis calls.")
    size_bytes:   int
    bands:        int
    rows:         int
    cols:         int
    crs:          str | None
    driver:       str
    nodata:       float | None
    message:      str


# ─────────────────────────────────────────────────────────────────────────────
# DEM Analysis
# ─────────────────────────────────────────────────────────────────────────────

class DEMAnalysisResponse(BaseModel):
    """Response from POST /api/analyze-dem."""
    file_id:             str
    average_slope_deg:   float
    maximum_slope_deg:   float
    std_slope_deg:       float
    safe_area_percent:   float
    average_elevation_m: float
    max_elevation_m:     float
    min_elevation_m:     float
    elevation_range_m:   float
    average_roughness_m: float
    elevation_stats:     RasterStats
    slope_stats:         RasterStats
    slope_grid:          JsonGrid        # downsampled slope heatmap
    tri_grid:            JsonGrid        # downsampled roughness heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Landing Sites
# ─────────────────────────────────────────────────────────────────────────────

class LandingSite(BaseModel):
    """A single ranked safe-landing candidate."""
    id:               int
    row:              int
    col:              int
    latitude:         float
    longitude:        float
    slope_deg:        float
    roughness_m:      float
    illumination:     float | None     # None if no illumination raster provided
    suitability_score: float           # 0–1, higher is better


class DetectLandingSitesResponse(BaseModel):
    """Response from POST /api/detect-landing-sites."""
    file_id:          str
    candidates:       list[LandingSite]
    total_safe_pixels: int
    safe_area_percent: float
    geojson:          dict[str, Any]   # GeoJSON FeatureCollection for the map


# ─────────────────────────────────────────────────────────────────────────────
# Ice Probability
# ─────────────────────────────────────────────────────────────────────────────

class IceCandidateRegion(BaseModel):
    """A candidate ice-rich region pixel."""
    rank:          int
    row:           int
    col:           int
    latitude:      float
    longitude:     float
    ice_score:     float         # composite [0, 1]
    radar_score:   float | None
    psr_score:     float | None
    temp_score:    float | None
    illum_score:   float | None


class IceProbabilityResponse(BaseModel):
    """Response from POST /api/ice-probability."""
    average_ice_probability: float
    max_ice_score:           float
    high_probability_area_pct: float    # % pixels with ice_score > 0.6
    top_candidate_regions:   list[IceCandidateRegion]
    ice_heatmap:             JsonGrid   # downsampled ice-score array
    component_weights:       dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# Hazard Map
# ─────────────────────────────────────────────────────────────────────────────

class HazardMapResponse(BaseModel):
    """Response from POST /api/hazard-map."""
    hazard_score:         float    # area-weighted mean [0, 1]
    high_risk_area_pct:   float
    medium_risk_area_pct: float
    low_risk_area_pct:    float
    hazard_grid:          JsonGrid
    risk_class_grid:      JsonGrid  # 0=low 1=medium 2=high (int encoded as float)


# ─────────────────────────────────────────────────────────────────────────────
# Path Planning
# ─────────────────────────────────────────────────────────────────────────────

class Coordinate(BaseModel):
    """A (row, col) pixel coordinate."""
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)


class PlanRouteRequest(BaseModel):
    """Request body for POST /api/plan-route."""
    dem_file_id:      str
    hazard_file_id:   str | None = None
    start:            Coordinate
    goal:             Coordinate
    slope_penalty:    float = Field(default=2.0,  ge=0.0, le=10.0)
    hazard_penalty:   float = Field(default=5.0,  ge=0.0, le=20.0)


class PathWaypoint(BaseModel):
    """One waypoint along the planned rover path."""
    step:      int
    row:       int
    col:       int
    latitude:  float
    longitude: float


class PlanRouteResponse(BaseModel):
    """Response from POST /api/plan-route."""
    found:             bool
    path_length_steps: int
    total_distance_m:  float
    estimated_energy:  float    # arbitrary unit: sum of (distance × cost) along path
    safety_score:      float    # 1 − mean hazard along path
    path_waypoints:    list[PathWaypoint]
    path_geojson:      dict[str, Any]   # GeoJSON LineString for map overlay


# ─────────────────────────────────────────────────────────────────────────────
# Generic error response
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
