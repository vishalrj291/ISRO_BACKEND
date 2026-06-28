"""
services/raster_utils.py
========================
Low-level raster I/O and spatial utility helpers used by all analysis services.

Responsibilities
----------------
* Open GeoTIFF files safely and return (data array, profile, transform, crs).
* Pixel ↔ geographic coordinate conversion using the affine transform.
* Row/column → lat/lon and lat/lon → row/column conversions.
* Safe nodata masking (converts nodata pixels to NaN).
* Downsampling a 2D array for lightweight JSON transport to the frontend.
* Building a minimal GeoJSON FeatureCollection from candidate site dicts.
* Saving a modified numpy array back to disk as GeoTIFF.

All functions raise descriptive ValueError / IOError rather than letting
rasterio errors bubble up raw, so the routers can return clean HTTP 422 responses.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine, rowcol, xy
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────
RasterProfile = dict[str, Any]
NDArray = np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Core raster I/O
# ─────────────────────────────────────────────────────────────────────────────

def open_raster(
    path: str | Path,
    band: int = 1,
) -> tuple[NDArray, RasterProfile, Affine, Any]:
    """
    Open a GeoTIFF and return the first (or specified) band as a float32 array.

    Returns
    -------
    data      : float32 ndarray of shape (rows, cols).  Nodata pixels → NaN.
    profile   : rasterio dataset profile (metadata dict).
    transform : Affine transform of the raster.
    crs       : CRS object (or None if undefined).

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    ValueError         if the file cannot be read as a raster or band is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    try:
        with rasterio.open(path) as src:
            if band > src.count:
                raise ValueError(
                    f"Raster has {src.count} band(s); requested band {band}."
                )

            profile   = dict(src.profile)
            transform = src.transform
            crs       = src.crs
            nodata    = src.nodata

            data = src.read(band).astype(np.float32)

            # Mask nodata pixels
            if nodata is not None:
                data[data == float(nodata)] = np.nan
            else:
                # Heuristic: treat extreme sentinel values as nodata
                # (common in LOLA DEM exports: -32768 for int16)
                if data.dtype == np.float32:
                    data[data < -9000] = np.nan

            logger.debug(
                f"Opened raster {path.name} | shape={data.shape} "
                f"| nodata={nodata} | crs={crs}"
            )

    except rasterio.errors.RasterioIOError as exc:
        raise ValueError(f"Cannot read raster '{path.name}': {exc}") from exc

    return data, profile, transform, crs


def save_raster(
    array: NDArray,
    path: str | Path,
    reference_profile: RasterProfile,
    dtype: str = "float32",
    nodata: float = -9999.0,
) -> Path:
    """
    Save a 2D numpy array to a GeoTIFF file using a reference profile.

    Parameters
    ----------
    array             : 2D float array to save.
    path              : Destination file path.
    reference_profile : Profile from the source raster (provides CRS, transform, etc.)
    dtype             : Output data type string (default 'float32').
    nodata            : Nodata sentinel to encode NaN pixels.

    Returns
    -------
    Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = reference_profile.copy()
    profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    out_array = array.astype(np.float32)
    out_array[np.isnan(out_array)] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out_array[np.newaxis, :, :])

    logger.debug(f"Saved raster → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_latlon(
    row: int,
    col: int,
    transform: Affine,
    crs: Any | None = None,
) -> tuple[float, float]:
    """
    Convert a pixel (row, col) to geographic (latitude, longitude) in degrees.

    For lunar datasets the CRS is typically MOON_2000 or a simple geographic
    CRS.  We return the raw x/y from the affine transform; callers should
    interpret x as longitude and y as latitude when the CRS is geographic.

    Returns (latitude, longitude) as floats.
    """
    lon, lat = xy(transform, row, col)
    return float(lat), float(lon)


def latlon_to_pixel(
    lat: float,
    lon: float,
    transform: Affine,
) -> tuple[int, int]:
    """
    Convert geographic (lat, lon) to the nearest pixel (row, col).

    Returns (row, col) as integers.
    """
    row, col = rowcol(transform, lon, lat)
    return int(row), int(col)


# ─────────────────────────────────────────────────────────────────────────────
# Array normalisation & masking
# ─────────────────────────────────────────────────────────────────────────────

def normalise_0_1(array: NDArray) -> NDArray:
    """
    Linearly scale a 2D float array to [0, 1], ignoring NaN.

    Returns the normalised array (same shape, float32).
    If the valid range is zero (flat array), returns an array of zeros.
    """
    vmin = np.nanmin(array)
    vmax = np.nanmax(array)
    span = vmax - vmin
    if span == 0.0:
        return np.zeros_like(array, dtype=np.float32)
    normed = (array - vmin) / span
    return normed.astype(np.float32)


def apply_valid_mask(
    *arrays: NDArray,
) -> NDArray:
    """
    Return a boolean mask that is True wherever ALL input arrays have valid
    (non-NaN) values.  Useful for multi-layer analysis.

    Parameters
    ----------
    arrays : One or more 2D float arrays (must share shape).

    Returns
    -------
    valid_mask : bool ndarray, True = valid pixel.
    """
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        mask &= ~np.isnan(arr)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Downsampling for JSON transport
# ─────────────────────────────────────────────────────────────────────────────

def downsample_array(
    array: NDArray,
    max_pixels: int = 512,
) -> NDArray:
    """
    Downsample a 2D array so its largest dimension ≤ max_pixels.
    Uses block-averaging (mean pooling) to preserve statistical integrity.

    This is used before serialising heatmaps to JSON so we don't send
    multi-megabyte arrays to the browser.

    Parameters
    ----------
    array      : 2D float array.
    max_pixels : Maximum size along the largest dimension after downsampling.

    Returns
    -------
    Downsampled 2D float array.
    """
    rows, cols = array.shape
    scale = min(1.0, max_pixels / max(rows, cols))
    if scale == 1.0:
        return array

    new_rows = max(1, int(rows * scale))
    new_cols = max(1, int(cols * scale))

    # Block-average: crop array so it divides evenly, then reshape
    r_block = rows // new_rows
    c_block = cols // new_cols

    cropped = array[: new_rows * r_block, : new_cols * c_block]
    downsampled = (
        cropped
        .reshape(new_rows, r_block, new_cols, c_block)
        .mean(axis=(1, 3))
    )

    return downsampled.astype(np.float32)


def array_to_json_grid(
    array: NDArray,
    max_pixels: int = 256,
    round_digits: int = 4,
) -> dict[str, Any]:
    """
    Prepare a 2D array for JSON serialisation:
      1. Downsample to ≤ max_pixels.
      2. Replace NaN with null (None) for JSON-safe output.
      3. Return a dict with the grid and its dimensions.

    Returns
    -------
    {
        "rows"  : int,
        "cols"  : int,
        "data"  : list[list[float | None]],
    }
    """
    small = downsample_array(array, max_pixels)
    rows, cols = small.shape

    grid: list[list[float | None]] = []
    for r in range(rows):
        row_data: list[float | None] = []
        for c in range(cols):
            val = small[r, c]
            if math.isnan(val):
                row_data.append(None)
            else:
                row_data.append(round(float(val), round_digits))
        grid.append(row_data)

    return {"rows": rows, "cols": cols, "data": grid}


# ─────────────────────────────────────────────────────────────────────────────
# Spatial statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_valid_stats(array: NDArray) -> dict[str, float]:
    """
    Compute basic descriptive statistics over valid (non-NaN) pixels.

    Returns
    -------
    {
        "min"    : float,
        "max"    : float,
        "mean"   : float,
        "median" : float,
        "std"    : float,
        "valid_pixel_count" : int,
        "total_pixel_count" : int,
    }
    """
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "valid_pixel_count": 0,
            "total_pixel_count": int(array.size),
        }
    return {
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "valid_pixel_count": int(valid.size),
        "total_pixel_count": int(array.size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GeoJSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def sites_to_geojson(sites: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Convert a list of candidate site dicts (each having 'latitude',
    'longitude', and arbitrary properties) into a GeoJSON FeatureCollection.

    This is consumed directly by React-Leaflet / OpenLayers on the frontend.
    """
    features = []
    for site in sites:
        lat = site.get("latitude")
        lon = site.get("longitude")
        if lat is None or lon is None:
            continue
        props = {k: v for k, v in site.items() if k not in ("latitude", "longitude")}
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": props,
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


# ─────────────────────────────────────────────────────────────────────────────
# Pixel resolution helpers
# ─────────────────────────────────────────────────────────────────────────────

def pixel_resolution_meters(transform: Affine, crs: Any | None) -> tuple[float, float]:
    """
    Estimate pixel resolution in metres per pixel (x, y).

    For geographic CRS (degrees) we approximate using the equatorial radius of
    the Moon (1 737 400 m).  For projected CRS we read the transform directly.

    Returns (x_res_m, y_res_m) — both positive.
    """
    MOON_RADIUS_M = 1_737_400.0

    x_res = abs(transform.a)
    y_res = abs(transform.e)

    if crs is not None and crs.is_geographic:
        # degrees → metres on lunar surface
        x_res = math.radians(x_res) * MOON_RADIUS_M
        y_res = math.radians(y_res) * MOON_RADIUS_M

    return x_res, y_res
