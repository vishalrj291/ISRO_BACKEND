"""
services/path_planning.py
==========================
A* rover path planning on a lunar terrain cost surface.

Cost function
-------------
  g(n) = accumulated cost from start to node n
  h(n) = Euclidean heuristic to goal (in pixels)

  step_cost(a → b) = distance(a, b)
                   + slope_penalty  × norm_slope(b)
                   + hazard_penalty × norm_hazard(b)

where:
  distance(a, b)   = 1.0  (cardinal)  or  √2  (diagonal)
  norm_slope(b)    = slope_deg(b) / 90.0   (always in [0,1])
  norm_hazard(b)   = hazard(b)             (already in [0,1])

The algorithm searches an 8-connected grid.  Pixels that are unreachable
(slope > 30° or hazard > 0.9) are treated as impassable walls.

Reference
---------
Hart, P. E., Nilsson, N. J., & Raphael, B. (1968).
"A Formal Basis for the Heuristic Determination of Minimum Cost Paths."
IEEE Transactions on Systems Science and Cybernetics.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from rasterio.transform import Affine

from services.raster_utils import pixel_to_latlon, pixel_resolution_meters


# ─────────────────────────────────────────────────────────────────────────────
# Passability thresholds
# ─────────────────────────────────────────────────────────────────────────────

MAX_PASSABLE_SLOPE_DEG: float = 30.0    # rover cannot traverse steeper slopes
MAX_PASSABLE_HAZARD:    float = 0.90    # hazard score above which cell is blocked


# ─────────────────────────────────────────────────────────────────────────────
# 8-connected neighbour directions: (d_row, d_col, distance_weight)
# ─────────────────────────────────────────────────────────────────────────────

_NEIGHBOURS = [
    (-1,  0, 1.0),    # N
    ( 1,  0, 1.0),    # S
    ( 0,  1, 1.0),    # E
    ( 0, -1, 1.0),    # W
    (-1,  1, math.sqrt(2)),  # NE
    (-1, -1, math.sqrt(2)),  # NW
    ( 1,  1, math.sqrt(2)),  # SE
    ( 1, -1, math.sqrt(2)),  # SW
]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathWaypoint:
    step:      int
    row:       int
    col:       int
    latitude:  float
    longitude: float


@dataclass
class PathPlanningResult:
    found:             bool
    path_length_steps: int
    total_distance_m:  float
    estimated_energy:  float
    safety_score:      float
    path_waypoints:    list[PathWaypoint]
    path_geojson:      dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# A* implementation
# ─────────────────────────────────────────────────────────────────────────────

def plan_route(
    slope_deg:     np.ndarray,
    transform:     Affine,
    start:         tuple[int, int],
    goal:          tuple[int, int],
    crs:           Any | None      = None,
    hazard:        np.ndarray | None = None,
    slope_penalty: float           = 2.0,
    hazard_penalty: float          = 5.0,
) -> PathPlanningResult:
    """
    Find the minimum-cost rover path between two pixels using A*.

    Parameters
    ----------
    slope_deg      : float32 2D slope array (degrees).
    transform      : Affine transform for pixel → lat/lon conversion.
    start          : (row, col) start pixel.
    goal           : (row, col) goal pixel.
    crs            : CRS object (for metric pixel size).
    hazard         : Optional float32 2D hazard array [0,1].
    slope_penalty  : Weight applied to slope cost term (0–10).
    hazard_penalty : Weight applied to hazard cost term (0–20).

    Returns
    -------
    PathPlanningResult — includes waypoint list, GeoJSON, energy and safety.
    """
    rows, cols = slope_deg.shape
    r0, c0     = start
    r1, c1     = goal

    # ── Bounds check ───────────────────────────────────────────────────────
    for (r, c), label in [((r0, c0), "start"), ((r1, c1), "goal")]:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(
                f"A* {label} ({r},{c}) is outside raster bounds ({rows}×{cols})."
            )

    # ── Build passability mask ─────────────────────────────────────────────
    passable = ~np.isnan(slope_deg) & (slope_deg < MAX_PASSABLE_SLOPE_DEG)
    if hazard is not None:
        passable &= (hazard < MAX_PASSABLE_HAZARD) | np.isnan(hazard)

    if not passable[r0, c0]:
        raise ValueError("Start pixel is in an impassable region (slope > 30° or hazard > 0.9).")
    if not passable[r1, c1]:
        raise ValueError("Goal pixel is in an impassable region (slope > 30° or hazard > 0.9).")

    # ── Normalised slope array (0–1) ───────────────────────────────────────
    norm_slope = np.clip(slope_deg / 90.0, 0.0, 1.0)
    norm_slope[np.isnan(norm_slope)] = 1.0   # treat NaN as high cost

    norm_hazard: np.ndarray | None = None
    if hazard is not None:
        norm_hazard = np.clip(hazard, 0.0, 1.0)
        norm_hazard[np.isnan(norm_hazard)] = 0.9

    # ── Heuristic: Euclidean pixel distance ───────────────────────────────
    def h(r: int, c: int) -> float:
        return math.sqrt((r - r1)**2 + (c - c1)**2)

    # ── Data structures ────────────────────────────────────────────────────
    # g_cost[r, c] = best accumulated cost to reach (r, c)
    g_cost = np.full((rows, cols), np.inf, dtype=np.float64)
    g_cost[r0, c0] = 0.0

    came_from: dict[tuple[int,int], tuple[int,int] | None] = {(r0, c0): None}

    # Priority queue: (f_cost, row, col)
    open_heap: list[tuple[float, int, int]] = []
    heapq.heappush(open_heap, (h(r0, c0), r0, c0))

    closed: set[tuple[int, int]] = set()

    iterations = 0
    MAX_ITERATIONS = rows * cols   # prevent infinite loops on huge rasters

    logger.debug(f"A* search | start={start} goal={goal} | grid={rows}×{cols}")

    # ── Main A* loop ───────────────────────────────────────────────────────
    found = False
    while open_heap and iterations < MAX_ITERATIONS:
        iterations += 1
        f_curr, r, c = heapq.heappop(open_heap)

        if (r, c) in closed:
            continue
        closed.add((r, c))

        if (r, c) == (r1, c1):
            found = True
            break

        for dr, dc, dist_w in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in closed:
                continue
            if not passable[nr, nc]:
                continue

            step_cost = (
                dist_w
                + slope_penalty  * float(norm_slope[nr, nc])  * dist_w
                + (hazard_penalty * float(norm_hazard[nr, nc]) * dist_w
                   if norm_hazard is not None else 0.0)
            )

            tentative_g = g_cost[r, c] + step_cost
            if tentative_g < g_cost[nr, nc]:
                g_cost[nr, nc] = tentative_g
                came_from[(nr, nc)] = (r, c)
                f_val = tentative_g + h(nr, nc)
                heapq.heappush(open_heap, (f_val, nr, nc))

    logger.debug(f"A* finished | found={found} | iterations={iterations}")

    # ── Reconstruct path ───────────────────────────────────────────────────
    if not found:
        return PathPlanningResult(
            found             = False,
            path_length_steps = 0,
            total_distance_m  = 0.0,
            estimated_energy  = 0.0,
            safety_score      = 0.0,
            path_waypoints    = [],
            path_geojson      = {"type": "FeatureCollection", "features": []},
        )

    path_pixels: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = (r1, c1)
    while cur is not None:
        path_pixels.append(cur)
        cur = came_from.get(cur)
    path_pixels.reverse()

    # ── Metrics ────────────────────────────────────────────────────────────
    x_res, y_res = pixel_resolution_meters(transform, crs)
    pixel_diag_m = math.sqrt(x_res**2 + y_res**2)

    total_dist_m  = 0.0
    total_energy  = 0.0
    hazard_values: list[float] = []

    for i in range(1, len(path_pixels)):
        pr, pc = path_pixels[i - 1]
        nr, nc = path_pixels[i]
        step_dist_m = (
            x_res if (nr == pr or nc == pc) else pixel_diag_m
        )
        total_dist_m += step_dist_m

        s = float(norm_slope[nr, nc])
        h_val = float(norm_hazard[nr, nc]) if norm_hazard is not None else 0.0
        step_energy = step_dist_m * (1.0 + slope_penalty * s + hazard_penalty * h_val)
        total_energy += step_energy

        if norm_hazard is not None:
            hazard_values.append(float(hazard[nr, nc]) if not math.isnan(float(hazard[nr, nc])) else 0.0)

    safety_score = (
        round(1.0 - float(np.mean(hazard_values)), 4)
        if hazard_values else 1.0
    )

    # ── Build waypoints ────────────────────────────────────────────────────
    waypoints: list[PathWaypoint] = []
    coordinates: list[list[float]] = []

    for step, (pr, pc) in enumerate(path_pixels):
        lat, lon = pixel_to_latlon(pr, pc, transform, crs)
        waypoints.append(PathWaypoint(
            step      = step,
            row       = pr,
            col       = pc,
            latitude  = round(lat, 6),
            longitude = round(lon, 6),
        ))
        coordinates.append([round(lon, 6), round(lat, 6)])

    # ── GeoJSON LineString ─────────────────────────────────────────────────
    path_geojson: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type":        "LineString",
                    "coordinates": coordinates,
                },
                "properties": {
                    "total_distance_m": round(total_dist_m, 2),
                    "estimated_energy": round(total_energy, 2),
                    "safety_score":     safety_score,
                    "steps":            len(path_pixels),
                },
            }
        ],
    }

    logger.info(
        f"Route planned | steps={len(path_pixels)} "
        f"| dist={total_dist_m:.1f} m | energy={total_energy:.1f} | safety={safety_score}"
    )

    return PathPlanningResult(
        found             = True,
        path_length_steps = len(path_pixels),
        total_distance_m  = round(total_dist_m, 2),
        estimated_energy  = round(total_energy, 2),
        safety_score      = safety_score,
        path_waypoints    = waypoints,
        path_geojson      = path_geojson,
    )
