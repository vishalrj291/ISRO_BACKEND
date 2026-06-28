"""
routers/upload.py
=================
POST /api/upload-raster

Accepts a GeoTIFF file via multipart form upload, saves it to
backend/data/uploads/<uuid>_<original_name>, then returns raster metadata
so the frontend can reference the file in subsequent analysis calls.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import rasterio
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from models.schemas import UploadRasterResponse

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Maximum file size: 2 GB (raster datasets can be large)
MAX_SIZE_BYTES = 2 * 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = {".tif", ".tiff", ".geotiff"}


@router.post(
    "/upload-raster",
    response_model=UploadRasterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a GeoTIFF raster dataset",
    description=(
        "Upload a LRO LOLA DEM, Diviner temperature, Mini-RF radar, illumination, "
        "or PSR mask GeoTIFF.  The server saves the file and returns its metadata "
        "(shape, CRS, band count) plus a `file_id` to reference in analysis calls."
    ),
)
async def upload_raster(
    file: UploadFile = File(..., description="GeoTIFF raster file to upload"),
) -> UploadRasterResponse:
    """Upload and inspect a GeoTIFF raster file."""

    # ── Validate file extension ────────────────────────────────────────────
    original_name = file.filename or "unknown.tif"
    suffix        = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Accepted extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ── Generate unique file ID and destination path ───────────────────────
    file_id  = uuid.uuid4().hex
    safe_name = f"{file_id}_{original_name}"
    dest_path = UPLOAD_DIR / safe_name

    # ── Stream to disk ─────────────────────────────────────────────────────
    try:
        total_bytes = 0
        with open(dest_path, "wb") as out_f:
            while chunk := await file.read(1024 * 1024):   # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_SIZE_BYTES:
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_SIZE_BYTES // (1024**3)} GB.",
                    )
                out_f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        logger.error(f"File write error for '{original_name}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {exc}",
        )

    # ── Read raster metadata ───────────────────────────────────────────────
    try:
        with rasterio.open(dest_path) as src:
            bands   = src.count
            rows    = src.height
            cols    = src.width
            crs_str = src.crs.to_string() if src.crs else None
            driver  = src.driver
            nodata  = float(src.nodata) if src.nodata is not None else None
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        logger.error(f"Rasterio could not open '{original_name}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Uploaded file is not a valid GeoTIFF: {exc}",
        )

    logger.info(
        f"Raster uploaded | id={file_id} | name={original_name} "
        f"| {rows}×{cols} | {bands} band(s) | {total_bytes:,} bytes"
    )

    return UploadRasterResponse(
        file_id       = file_id,
        original_name = original_name,
        file_path     = str(dest_path),
        size_bytes    = total_bytes,
        bands         = bands,
        rows          = rows,
        cols          = cols,
        crs           = crs_str,
        driver        = driver,
        nodata        = nodata,
        message       = "Raster uploaded successfully.",
    )
