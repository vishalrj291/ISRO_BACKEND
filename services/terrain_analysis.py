"""
services/terrain_analysis.py
=============================
All terrain-analysis computations performed on lunar raster datasets.

Covers
------
1. Slope computation  from a DEM (Horn's method via numpy gradient).
2. Roughness / TRI    (Terrain Ruggedness Index).
3. Landing-site detection from slope + optional illumination + roughness.
4. Hazard map generation from slope + roughness thresholds.
5. Supporting helpers: pixel-level candidate ranking, suitability scoring.

Design principles
-----------------
* Pure-numpy computations — no GDAL subprocess calls at runtime.
* All functions accept float32 ndarrays and return float32 ndarrays or dicts.
* Physical constants are tuned for the Moon (gravity, radius, etc.).
* Every public function is thoroughly documented for maintainability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter
from rasterio.transform import Affine
from loguru import logger

from services.raster_utils import (
    pixel_to_latlon,
    normalise_0_1,
    apply_valid_mask,
    pixel_resolution_meters,
    compute_valid_stats,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration / thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Maximum slope (degrees) for a pixel to qualify as a safe landing candidate.
SAFE_SLOPE_DEG: float = 7.0

# Minimum illumination fraction [0, 1] for landing site selection.
# 0 = no constraint (used when no illumination raster is supplied).
MIN_ILLUMINATION: float = 0.4

# Terrain Ruggedness Index threshold — sites below this are considered smooth.
MAX_ROUGHNESS: float = 5.0  # metres (TRI)

# Number of top landing-site candidates to return.
N_CANDIDATES: int = 5

# Gaussian filter radius used for local slope smoothing before site detection.
SMOOTH_RADIUS_PIX: int = 3

# Hazard thresholds
HAZARD_SLOPE_HIGH: float = 15.0   # degrees — high-risk slope
HAZARD_SLOPE_MED:  float = 7.0    # degrees — medium-risk slope

HAZARD_ROUGH_HIGH: float = 10.0   # TRI metres — high roughness
HAZARD_ROUGH_MED:  float = 5.0    # TRI metres — medium roughness


# ─────────────────────────────────────────────────────────────────────────────
# Slope computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_slope(
    dem: np.ndarray,
    transform: Affine,
    crs: Any | None = None,
) -> np.ndarray:
    """
    Compute slope in degrees from a DEM using Horn's finite-difference method.

    The gradient is computed using numpy.gradient which performs a second-order
    central difference over interior pixels.  Edge pixels use first-order
    one-sided differences.

    Parameters
    ----------
    dem       : float32 2D array of elevation values (metres).
    transform : Affine transform of the DEM raster.
    crs       : CRS object (used to determine pixel size in metres).

    Returns
    -------
    slope : float32 2D array of slope values in degrees.
            NaN pixels in the DEM propagate to NaN in the slope.
    """
    x_res, y_res = pixel_resolution_meters(transform, crs)
    logger.debug(f"compute_slope | pixel_res=({x_res:.2f} m, {y_res:.2f} m)")

    # numpy.gradient returns [∂z/∂row, ∂z/∂col]
    # ∂z/∂row corresponds to the north–south gradient (y-direction)
    # ∂z/∂col corresponds to the east–west gradient (x-direction)
    dz_dy, dz_dx = np.gradient(dem, y_res, x_res)

    # Slope in radians, then convert to degrees
    slope_rad  = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg  = np.degrees(slope_rad).astype(np.float32)

    # Propagate NaN from DEM
    slope_deg[np.isnan(dem)] = np.nan

    logger.debug(
        f"Slope computed | min={np.nanmin(slope_deg):.2f}° "
        f"max={np.nanmax(slope_deg):.2f}° mean={np.nanmean(slope_deg):.2f}°"
    )
    return slope_deg


# ─────────────────────────────────────────────────────────────────────────────
# Terrain Ruggedness Index (TRI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_tri(
    dem: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """
    Compute the Terrain Ruggedness Index (TRI) per Riley et al. (1999).

    TRI is defined as the mean absolute difference between the centre pixel
    and its 8 neighbours within a (window_size × window_size) window.

    For a 3×3 window:
        TRI(i,j) = sqrt( Σ (z_neighbour − z_centre)² )

    Parameters
    ----------
    dem         : float32 2D elevation array (metres).
    window_size : Size of the square neighbourhood (must be odd, ≥ 3).

    Returns
    -------
    tri : float32 2D array (same shape as dem) of TRI values in metres.
    """
    if window_size % 2 == 0:
        window_size += 1  # ensure odd

    # Mean of the local neighbourhood
    dem_smooth = uniform_filter(dem, size=window_size, mode="nearest")

    tri = np.abs(dem - dem_smooth).astype(np.float32)
    tri[np.isnan(dem)] = np.nan

    logger.debug(
        f"TRI computed | window={window_size}px "
        f"| mean={np.nanmean(tri):.3f} m  max={np.nanmax(tri):.3f} m"
    )
    return tri


# ─────────────────────────────────────────────────────────────────────────────
# DEM summary statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DEMAnalysisResult:
    """Structured result returned by analyse_dem()."""
    average_slope_deg: float
    maximum_slope_deg: float
    std_slope_deg: float
    safe_area_percent: float          # fraction of valid pixels with slope < 7°
    average_elevation_m: float
    max_elevation_m: float
    min_elevation_m: float
    elevation_range_m: float
    average_roughness_m: float
    slope_stats: dict[str, float]
    elevation_stats: dict[str, float]
    # Arrays serialised separately via array_to_json_grid
    slope_array: np.ndarray = field(repr=False)
    tri_array:   np.ndarray = field(repr=False)


def analyse_dem(
    dem: np.ndarray,
    transform: Affine,
    crs: Any | None = None,
) -> DEMAnalysisResult:
    """
    Full DEM analysis pipeline:
      1. Compute slope map.
      2. Compute TRI roughness map.
      3. Derive aggregate statistics.

    Parameters
    ----------
    dem       : float32 2D elevation array.
    transform : Affine transform from the raster.
    crs       : CRS of the raster (used for metric pixel size).

    Returns
    -------
    DEMAnalysisResult dataclass with all stats and the computed arrays.
    """
    slope = compute_slope(dem, transform, crs)
    tri   = compute_tri(dem)

    valid_slope = slope[~np.isnan(slope)]
    valid_dem   = dem[~np.isnan(dem)]

    # Safe area = percentage of valid pixels with slope < threshold
    safe_pixels     = (valid_slope < SAFE_SLOPE_DEG).sum()
    safe_area_pct   = float(safe_pixels / valid_slope.size * 100.0) if valid_slope.size else 0.0

    elev_stats  = compute_valid_stats(dem)
    slope_stats = compute_valid_stats(slope)

    result = DEMAnalysisResult(
        average_slope_deg   = float(np.mean(valid_slope)) if valid_slope.size else float("nan"),
        maximum_slope_deg   = float(np.max(valid_slope))  if valid_slope.size else float("nan"),
        std_slope_deg       = float(np.std(valid_slope))  if valid_slope.size else float("nan"),
        safe_area_percent   = round(safe_area_pct, 2),
        average_elevation_m = float(np.mean(valid_dem))   if valid_dem.size else float("nan"),
        max_elevation_m     = float(np.max(valid_dem))    if valid_dem.size else float("nan"),
        min_elevation_m     = float(np.min(valid_dem))    if valid_dem.size else float("nan"),
        elevation_range_m   = float(np.ptp(valid_dem))    if valid_dem.size else float("nan"),
        average_roughness_m = float(np.nanmean(tri)),
        slope_stats         = slope_stats,
        elevation_stats     = elev_stats,
        slope_array         = slope,
        tri_array           = tri,
    )

    logger.info(
        f"DEM analysis complete | avg_slope={result.average_slope_deg:.2f}° "
        f"| safe_area={result.safe_area_percent:.1f}%"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Landing-site candidate detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LandingSiteCandidate:
    """A single ranked landing-site candidate."""
    id:               int
    row:              int
    col:              int
    latitude:         float
    longitude:        float
    slope_deg:        float
    roughness_m:      float
    illumination:     float          # 0–1; NaN if no illumination layer provided
    suitability_score: float         # 0–1 composite


def _suitability_score(
    slope_deg: float,
    roughness_m: float,
    illumination: float,
    slope_weight: float    = 0.50,
    rough_weight: float    = 0.30,
    illum_weight: float    = 0.20,
    max_slope: float       = SAFE_SLOPE_DEG,
    max_roughness: float   = MAX_ROUGHNESS,
) -> float:
    """
    Compute a composite suitability score in [0, 1].

    Higher is better.
    - Slope component: linear decay from 1 (slope=0) to 0 (slope=max_slope).
    - Roughness component: linear decay from 1 (TRI=0) to 0 (TRI=max_roughness).
    - Illumination component: raw fraction (already [0, 1]).

    If illumination is NaN we redistribute its weight to slope + roughness.
    """
    slope_score = max(0.0, 1.0 - (slope_deg / max_slope))
    rough_score = max(0.0, 1.0 - (roughness_m / max_roughness))

    if math.isnan(illumination):
        # Redistribute illumination weight
        total = slope_weight + rough_weight
        s = (slope_weight / total) * slope_score + (rough_weight / total) * rough_score
        return round(float(s), 4)

    illum_score = float(np.clip(illumination, 0.0, 1.0))
    s = (
        slope_weight * slope_score
        + rough_weight * rough_score
        + illum_weight * illum_score
    )
    return round(float(s), 4)


def detect_landing_sites(
    dem:          np.ndarray,
    transform:    Affine,
    crs:          Any | None      = None,
    illumination: np.ndarray | None = None,
    slope:        np.ndarray | None = None,   # pre-computed slope to avoid duplicate work
    tri:          np.ndarray | None = None,   # pre-computed TRI
    n_sites:      int             = N_CANDIDATES,
    min_sep_pixels: int           = 50,       # minimum pixel distance between candidates
) -> list[LandingSiteCandidate]:
    """
    Identify the top-N safe landing-site candidates.

    Algorithm
    ---------
    1. Compute slope and TRI if not provided.
    2. Build a combined suitability array.
    3. Apply hard exclusion masks (slope > threshold, roughness > threshold).
    4. Select the top-N pixels by suitability score using greedy suppression
       so that candidates are spatially distributed (minimum pixel separation).

    Parameters
    ----------
    dem           : float32 2D elevation array.
    transform     : Affine transform of the DEM.
    crs           : CRS object.
    illumination  : Optional float32 2D illumination fraction [0, 1].
    slope         : Pre-computed slope array (degrees), optional.
    tri           : Pre-computed TRI array (metres), optional.
    n_sites       : Number of candidates to return.
    min_sep_pixels: Minimum spatial separation between candidates in pixels.

    Returns
    -------
    List of LandingSiteCandidate objects, ranked by suitability_score descending.
    """
    if slope is None:
        slope = compute_slope(dem, transform, crs)
    if tri is None:
        tri = compute_tri(dem)

    rows, cols = dem.shape

    # ── Normalised component arrays ─────────────────────────────────────────
    slope_norm = normalise_0_1(np.clip(slope, 0, SAFE_SLOPE_DEG * 2))
    slope_score_arr = 1.0 - slope_norm  # higher score → gentler slope

    tri_norm   = normalise_0_1(np.clip(tri, 0, MAX_ROUGHNESS * 2))
    tri_score_arr = 1.0 - tri_norm

    if illumination is not None:
        illum_arr  = normalise_0_1(illumination)
        suitability = (
            0.50 * slope_score_arr
            + 0.30 * tri_score_arr
            + 0.20 * illum_arr
        ).astype(np.float32)
    else:
        suitability = (
            0.60 * slope_score_arr
            + 0.40 * tri_score_arr
        ).astype(np.float32)

    # ── Hard masks ──────────────────────────────────────────────────────────
    valid = apply_valid_mask(slope, tri)
    safe_slope_mask = slope < SAFE_SLOPE_DEG
    safe_rough_mask = tri   < MAX_ROUGHNESS

    if illumination is not None:
        safe_illum_mask = illumination > MIN_ILLUMINATION
        combined_mask   = valid & safe_slope_mask & safe_rough_mask & safe_illum_mask
    else:
        combined_mask   = valid & safe_slope_mask & safe_rough_mask

    # Zero out unsuitable pixels
    suitability_masked = np.where(combined_mask, suitability, 0.0)

    # ── Greedy candidate selection with spatial suppression ─────────────────
    candidates: list[LandingSiteCandidate] = []
    suppressed = np.zeros((rows, cols), dtype=bool)

    # Pre-compute pixel coordinate grid for distance calculation
    row_idx, col_idx = np.indices((rows, cols))

    for rank in range(n_sites):
        masked_suit = np.where(~suppressed, suitability_masked, 0.0)
        best_flat   = np.argmax(masked_suit)
        best_score  = masked_suit.flat[best_flat]

        if best_score <= 0.0:
            logger.warning(f"Only {rank} landing candidates met all criteria.")
            break

        r, c = divmod(int(best_flat), cols)

        lat, lon = pixel_to_latlon(r, c, transform, crs)

        raw_slope  = float(slope[r, c]) if not math.isnan(float(slope[r, c])) else 0.0
        raw_tri    = float(tri[r, c])   if not math.isnan(float(tri[r, c]))   else 0.0
        raw_illum  = float(illumination[r, c]) if (
            illumination is not None and not math.isnan(float(illumination[r, c]))
        ) else float("nan")

        score = _suitability_score(raw_slope, raw_tri, raw_illum)

        candidates.append(LandingSiteCandidate(
            id               = rank + 1,
            row              = r,
            col              = c,
            latitude         = round(lat, 6),
            longitude        = round(lon, 6),
            slope_deg        = round(raw_slope, 3),
            roughness_m      = round(raw_tri,   3),
            illumination     = round(raw_illum, 3) if not math.isnan(raw_illum) else float("nan"),
            suitability_score = score,
        ))

        # Suppress a disc of radius min_sep_pixels around the chosen pixel
        dist_sq = (row_idx - r)**2 + (col_idx - c)**2
        suppressed |= dist_sq < min_sep_pixels**2

    logger.info(f"Landing-site detection found {len(candidates)} candidate(s).")
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Hazard map generation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HazardMapResult:
    """Output of generate_hazard_map()."""
    hazard_score:         float       # area-weighted mean hazard [0, 1]
    high_risk_area_pct:   float       # % valid pixels classified HIGH
    medium_risk_area_pct: float       # % valid pixels classified MEDIUM
    low_risk_area_pct:    float       # % valid pixels classified LOW
    hazard_array:         np.ndarray  # float32 [0, 1] heatmap
    risk_class_array:     np.ndarray  # int8: 0=low, 1=medium, 2=high


def generate_hazard_map(
    slope: np.ndarray,
    tri:   np.ndarray,
    shadow_mask: np.ndarray | None = None,
) -> HazardMapResult:
    """
    Generate a normalised hazard map from slope and terrain roughness.

    Hazard components
    -----------------
    1. Slope hazard (weight 0.55):
        LOW    → slope < 7°
        MEDIUM → 7° ≤ slope < 15°
        HIGH   → slope ≥ 15°

    2. Roughness hazard (weight 0.30):
        LOW    → TRI < 5 m
        MEDIUM → 5 m ≤ TRI < 10 m
        HIGH   → TRI ≥ 10 m

    3. Shadow hazard (weight 0.15, optional):
        If a PSR/shadow mask is provided, permanently shadowed pixels
        receive a hazard contribution of 0.5 (partial, not absolute hazard).

    Parameters
    ----------
    slope       : float32 2D slope array (degrees).
    tri         : float32 2D TRI array (metres).
    shadow_mask : Optional binary 2D array (1 = shadowed, 0 = illuminated).

    Returns
    -------
    HazardMapResult with per-pixel hazard array and area statistics.
    """
    rows, cols = slope.shape
    valid = apply_valid_mask(slope, tri)

    # ── Component hazard arrays ─────────────────────────────────────────────
    # Slope hazard [0, 1]: linearly scaled
    slope_hazard = np.clip(slope / HAZARD_SLOPE_HIGH, 0.0, 1.0).astype(np.float32)

    # Roughness hazard [0, 1]
    rough_hazard = np.clip(tri / HAZARD_ROUGH_HIGH, 0.0, 1.0).astype(np.float32)

    # ── Combine ─────────────────────────────────────────────────────────────
    if shadow_mask is not None:
        shadow_component = (shadow_mask > 0).astype(np.float32) * 0.5
        hazard = (
            0.55 * slope_hazard
            + 0.30 * rough_hazard
            + 0.15 * shadow_component
        )
    else:
        hazard = 0.65 * slope_hazard + 0.35 * rough_hazard

    hazard = hazard.astype(np.float32)
    hazard[~valid] = np.nan

    # ── Risk classification ─────────────────────────────────────────────────
    risk_class = np.full(hazard.shape, -1, dtype=np.int8)  # -1 = nodata
    risk_class[valid & (hazard < 0.33)]  = 0   # LOW
    risk_class[valid & (hazard >= 0.33) & (hazard < 0.66)] = 1  # MEDIUM
    risk_class[valid & (hazard >= 0.66)] = 2   # HIGH

    valid_count = valid.sum()

    def pct(cls_val: int) -> float:
        if valid_count == 0:
            return 0.0
        return float((risk_class == cls_val).sum() / valid_count * 100.0)

    hazard_score = float(np.nanmean(hazard)) if valid_count > 0 else float("nan")

    result = HazardMapResult(
        hazard_score         = round(hazard_score, 4),
        high_risk_area_pct   = round(pct(2), 2),
        medium_risk_area_pct = round(pct(1), 2),
        low_risk_area_pct    = round(pct(0), 2),
        hazard_array         = hazard,
        risk_class_array     = risk_class,
    )

    logger.info(
        f"Hazard map generated | score={result.hazard_score:.3f} "
        f"| high={result.high_risk_area_pct:.1f}% "
        f"| med={result.medium_risk_area_pct:.1f}%"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Elevation profile extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_elevation_profile(
    dem:       np.ndarray,
    transform: Affine,
    start_rc:  tuple[int, int],
    end_rc:    tuple[int, int],
    n_points:  int = 200,
) -> list[dict[str, float]]:
    """
    Sample the DEM along a straight line between two pixels and return
    an elevation profile suitable for the frontend ElevationProfile chart.

    Parameters
    ----------
    dem       : float32 2D elevation array.
    transform : Affine transform.
    start_rc  : (row, col) of the profile start.
    end_rc    : (row, col) of the profile end.
    n_points  : Number of sample points along the transect.

    Returns
    -------
    List of dicts: [{"distance_m": float, "elevation_m": float}, ...]
    """
    r0, c0 = start_rc
    r1, c1 = end_rc

    rows_arr = np.linspace(r0, r1, n_points)
    cols_arr = np.linspace(c0, c1, n_points)

    # Bilinear interpolation via map_coordinates
    from scipy.ndimage import map_coordinates
    elevations = map_coordinates(dem, [rows_arr, cols_arr], order=1, mode="nearest")

    x_res, y_res = pixel_resolution_meters(transform, None)
    pixel_dist_m = math.sqrt(x_res**2 + y_res**2)

    total_dist_m = pixel_dist_m * math.sqrt((r1 - r0)**2 + (c1 - c0)**2)
    distances_m  = np.linspace(0, total_dist_m, n_points)

    profile = []
    for dist, elev in zip(distances_m, elevations):
        profile.append({
            "distance_m":  round(float(dist),  2),
            "elevation_m": round(float(elev),  2),
        })

    return profile
