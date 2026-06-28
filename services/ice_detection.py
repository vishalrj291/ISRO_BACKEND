"""
services/ice_detection.py
==========================
Subsurface ice probability estimation from multi-source lunar datasets.

Ice Score formula (per specification)
--------------------------------------
IceScore = 0.4 × radar_score
         + 0.3 × psr_score
         + 0.2 × temperature_score
         + 0.1 × illumination_score

Physical rationale
------------------
* Radar (Mini-RF CPR)       — high circular polarisation ratio → high ice probability.
* PSR mask                  — permanently shadowed regions retain surface ice.
* Temperature (Diviner)     — lower brightness temperature → more likely ice-stable.
* Illumination              — used inversely: low illumination → shadowed → ice-prone.

All input arrays are normalised independently to [0, 1] before weighting.
Temperature is inverted because colder = higher ice probability.
Illumination is inverted for the same reason.

Top candidate regions are selected via greedy suppression (same logic as
landing-site detection) so they are spatially distributed across the scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from services.raster_utils import (
    normalise_0_1,
    apply_valid_mask,
    pixel_to_latlon,
    array_to_json_grid,
)
from rasterio.transform import Affine


# ─────────────────────────────────────────────────────────────────────────────
# Component weights
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "radar":        0.40,
    "psr":          0.30,
    "temperature":  0.20,
    "illumination": 0.10,
}

# Minimum ice score to include a region in top candidates
ICE_THRESHOLD_HIGH = 0.60     # "high probability"

# Number of candidate ice-rich regions to return
N_ICE_CANDIDATES = 10

# Minimum pixel separation between candidates
MIN_SEP_PIX = 30


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IceCandidateRegion:
    rank:         int
    row:          int
    col:          int
    latitude:     float
    longitude:    float
    ice_score:    float
    radar_score:  float | None
    psr_score:    float | None
    temp_score:   float | None
    illum_score:  float | None


@dataclass
class IceProbabilityResult:
    average_ice_probability:  float
    max_ice_score:            float
    high_probability_area_pct: float
    top_candidate_regions:    list[IceCandidateRegion]
    ice_heatmap_grid:         dict[str, Any]   # JsonGrid-compatible dict
    component_weights:        dict[str, float]
    # raw array for downstream use (not serialised directly)
    ice_score_array:          np.ndarray = field(repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_radar(radar: np.ndarray) -> np.ndarray:
    """
    Normalise Mini-RF Circular Polarisation Ratio (CPR) to [0, 1].
    Higher CPR values indicate volumetric scattering consistent with ice.
    """
    return normalise_0_1(radar)


def _score_psr(psr: np.ndarray) -> np.ndarray:
    """
    PSR mask is typically binary (1 = permanently shadowed, 0 = illuminated).
    We normalise so that PSR=1 → score=1.0, PSR=0 → score=0.0.
    If the mask has continuous values (e.g., fraction of time in shadow),
    normalise_0_1 handles it naturally.
    """
    return normalise_0_1(psr)


def _score_temperature(temperature: np.ndarray) -> np.ndarray:
    """
    Diviner brightness temperature: lower temperature → higher ice probability.
    We invert so that cold pixels score near 1.0.
    """
    normed = normalise_0_1(temperature)
    return (1.0 - normed).astype(np.float32)


def _score_illumination(illumination: np.ndarray) -> np.ndarray:
    """
    Illumination: lower illumination → more shadowed → higher ice probability.
    We invert so that dark (shadowed) pixels score near 1.0.
    """
    normed = normalise_0_1(illumination)
    return (1.0 - normed).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main ice probability computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ice_probability(
    transform:   Affine,
    crs:         Any | None         = None,
    radar:       np.ndarray | None  = None,
    psr:         np.ndarray | None  = None,
    temperature: np.ndarray | None  = None,
    illumination: np.ndarray | None = None,
    n_candidates: int               = N_ICE_CANDIDATES,
    min_sep_pixels: int             = MIN_SEP_PIX,
) -> IceProbabilityResult:
    """
    Compute the composite ice probability score for every pixel.

    At least one input raster must be provided.  Weights are dynamically
    redistributed when layers are missing so the total always sums to 1.0.

    Parameters
    ----------
    transform    : Affine transform (shared by all input rasters).
    crs          : CRS object (used only for pixel→latlon conversion).
    radar        : Mini-RF CPR raster (float32).
    psr          : PSR mask raster (float32 or binary).
    temperature  : Diviner temperature raster (float32, Kelvin or DN).
    illumination : Illumination fraction raster [0, 1] (float32).
    n_candidates : Number of top ice-rich regions to return.
    min_sep_pixels: Minimum pixel separation between candidates.

    Returns
    -------
    IceProbabilityResult
    """
    # ── Validate at least one input ────────────────────────────────────────
    inputs = {
        "radar":        radar,
        "psr":          psr,
        "temperature":  temperature,
        "illumination": illumination,
    }
    available = {k: v for k, v in inputs.items() if v is not None}
    if not available:
        raise ValueError("At least one raster (radar, psr, temperature, illumination) must be provided.")

    # ── Determine reference shape ──────────────────────────────────────────
    ref_shape = next(iter(available.values())).shape
    for name, arr in available.items():
        if arr.shape != ref_shape:
            # Crop to common extent
            min_r = min(ref_shape[0], arr.shape[0])
            min_c = min(ref_shape[1], arr.shape[1])
            ref_shape = (min_r, min_c)
            logger.warning(f"Raster '{name}' shape mismatch — cropping to {ref_shape}")

    # Crop all arrays to ref_shape
    cropped: dict[str, np.ndarray] = {}
    for name, arr in available.items():
        cropped[name] = arr[:ref_shape[0], :ref_shape[1]]

    # ── Score each available component ─────────────────────────────────────
    scorers = {
        "radar":        _score_radar,
        "psr":          _score_psr,
        "temperature":  _score_temperature,
        "illumination": _score_illumination,
    }
    scored: dict[str, np.ndarray] = {}
    for name, scorer in scorers.items():
        if name in cropped:
            scored[name] = scorer(cropped[name])

    # ── Redistribute weights for missing layers ────────────────────────────
    total_weight = sum(WEIGHTS[k] for k in scored)
    effective_weights = {k: WEIGHTS[k] / total_weight for k in scored}

    # ── Composite ice score ────────────────────────────────────────────────
    ice_score = np.zeros(ref_shape, dtype=np.float32)
    for name, score_arr in scored.items():
        w = effective_weights[name]
        ice_score += w * np.where(np.isnan(score_arr), 0.0, score_arr)

    # Preserve NaN where ALL inputs are NaN
    all_nan_mask = np.ones(ref_shape, dtype=bool)
    for arr in scored.values():
        all_nan_mask &= np.isnan(arr)
    ice_score[all_nan_mask] = np.nan

    # ── Summary statistics ─────────────────────────────────────────────────
    valid_ice = ice_score[~np.isnan(ice_score)]
    avg_prob  = float(np.mean(valid_ice))   if valid_ice.size else 0.0
    max_score = float(np.max(valid_ice))    if valid_ice.size else 0.0
    high_pct  = float((valid_ice > ICE_THRESHOLD_HIGH).mean() * 100.0) if valid_ice.size else 0.0

    logger.info(
        f"Ice probability | avg={avg_prob:.3f} | max={max_score:.3f} "
        f"| high_pct={high_pct:.1f}%"
    )

    # ── Top candidate extraction with spatial suppression ──────────────────
    rows, cols = ref_shape
    row_idx, col_idx = np.indices((rows, cols))
    suppressed   = np.zeros((rows, cols), dtype=bool)
    ice_masked   = np.where(~np.isnan(ice_score), ice_score, 0.0)

    candidates: list[IceCandidateRegion] = []

    for rank in range(n_candidates):
        masked = np.where(~suppressed, ice_masked, 0.0)
        best_flat = np.argmax(masked)
        best_val  = masked.flat[best_flat]

        if best_val <= 0.0:
            break

        r, c = divmod(int(best_flat), cols)
        lat, lon = pixel_to_latlon(r, c, transform, crs)

        def _get(name: str) -> float | None:
            if name in scored:
                v = float(scored[name][r, c])
                return None if math.isnan(v) else round(v, 4)
            return None

        candidates.append(IceCandidateRegion(
            rank         = rank + 1,
            row          = r,
            col          = c,
            latitude     = round(lat, 6),
            longitude    = round(lon, 6),
            ice_score    = round(float(ice_score[r, c]), 4),
            radar_score  = _get("radar"),
            psr_score    = _get("psr"),
            temp_score   = _get("temperature"),
            illum_score  = _get("illumination"),
        ))

        dist_sq = (row_idx - r)**2 + (col_idx - c)**2
        suppressed |= dist_sq < min_sep_pixels**2

    # ── Heatmap for frontend ───────────────────────────────────────────────
    heatmap_dict = array_to_json_grid(ice_score, max_pixels=256)

    return IceProbabilityResult(
        average_ice_probability   = round(avg_prob,  4),
        max_ice_score             = round(max_score, 4),
        high_probability_area_pct = round(high_pct,  2),
        top_candidate_regions     = candidates,
        ice_heatmap_grid          = heatmap_dict,
        component_weights         = {k: round(v, 4) for k, v in effective_weights.items()},
        ice_score_array           = ice_score,
    )
