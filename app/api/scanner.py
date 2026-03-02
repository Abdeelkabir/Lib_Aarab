"""
app/api/scanner.py
All scanner endpoints.

Routes:
  POST /api/scan          — upload images + options → returns job results + PDF
  POST /api/recrop        — re-process one image with user-adjusted corners
  GET  /api/jobs/{job_id}/pdf          — download the PDF
  GET  /api/jobs/{job_id}/images/{idx} — download a single processed image
"""
import cv2
import uuid
import logging
import numpy as np
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response
import json

from app.core.config import (
    MAX_UPLOAD_SIZE_MB, MAX_IMAGES_PER_BATCH, ALLOWED_EXTENSIONS, JPEG_QUALITY
)
from app.core.sam import get_predictor
from app.models.schemas import (
    ProcessingOptions, ScanResponse, ImageResult, CropCorners,
    RecropRequest, OutputStyle, BinarizationMethod
)
from app.services.detection import detect_document, order_points, image_corners
from app.services.processing import process_image
from app.services.pdf_builder import images_to_pdf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["scanner"])

# ── Job storage (in-memory — replace with Redis/DB for multi-worker) ──────────
# Stores: { job_id: { "originals": [bgr], "processed": [bgr], "quads": [pts] } }
_jobs: dict = {}

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/tmp/docscanner_jobs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_file(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")


def _read_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image data.")
    return img


def _encode_jpeg(img_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


def _corners_to_array(corners: CropCorners) -> np.ndarray:
    return np.array([
        corners.tl, corners.tr, corners.br, corners.bl
    ], dtype=np.float32)


def _array_to_corners(quad: np.ndarray) -> CropCorners:
    return CropCorners(
        tl=quad[0].tolist(),
        tr=quad[1].tolist(),
        br=quad[2].tolist(),
        bl=quad[3].tolist(),
    )


# ── POST /api/scan ─────────────────────────────────────────────────────────────

@router.post("/scan", response_model=ScanResponse)
async def scan_documents(
    files:        List[UploadFile] = File(...),
    options_json: str              = Form("{}"),
):
    """
    Upload 1–N document images.
    Returns processed images + assembled PDF.

    options_json (Form field, JSON string):
    {
        "output_style":   "original" | "grayscale" | "morphological",
        "shadow_removal": true | false,
        "binarization":   "none" | "adaptive" | "sauvola"
    }
    """
    # Parse options
    try:
        opts = ProcessingOptions(**json.loads(options_json))
    except Exception as e:
        raise HTTPException(422, f"Invalid options: {e}")

    # Validate batch size
    if len(files) > MAX_IMAGES_PER_BATCH:
        raise HTTPException(400, f"Max {MAX_IMAGES_PER_BATCH} images per request.")

    for f in files:
        _validate_file(f)

    predictor   = get_predictor()
    job_id      = uuid.uuid4().hex
    job_dir     = OUTPUT_DIR / job_id
    job_dir.mkdir()

    originals:  List[np.ndarray] = []
    processed:  List[np.ndarray] = []
    quads:      List[any]        = []
    results:    List[ImageResult] = []

    for idx, upload in enumerate(files):
        filename = upload.filename or f"image_{idx}.jpg"
        try:
            data = await upload.read()

            # Size check
            if len(data) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                raise ValueError(f"File exceeds {MAX_UPLOAD_SIZE_MB}MB limit.")

            img_bgr   = _read_image(data)
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Detect document
            quad, _mask = detect_document(image_rgb, predictor)

            # Fall back to full image if detection failed
            if quad is None:
                logger.warning(f"[{filename}] No document detected, using full image.")
                quad = image_corners(img_bgr)

            # Process
            output = process_image(
                img_bgr   = img_bgr,
                quad      = quad,
                style     = OutputStyle(opts.output_style),
                shadow    = opts.shadow_removal,
                binarize  = BinarizationMethod(opts.binarization),
            )

            # Save individual processed image
            img_path = job_dir / f"{idx:04d}.jpg"
            cv2.imwrite(str(img_path), output, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            # Save original image (used by CropEditor to show before processing)
            orig_path = job_dir / f"{idx:04d}_orig.jpg"
            cv2.imwrite(str(orig_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            originals.append(img_bgr)
            processed.append(output)
            quads.append(quad)

            results.append(ImageResult(
                index    = idx,
                filename = filename,
                success  = True,
                corners  = _array_to_corners(quad),
                width    = output.shape[1],
                height   = output.shape[0],
            ))

        except Exception as e:
            logger.error(f"[{filename}] Processing error: {e}", exc_info=True)
            originals.append(None)
            processed.append(None)
            quads.append(None)
            results.append(ImageResult(
                index    = idx,
                filename = filename,
                success  = False,
                error    = str(e),
            ))

    # Build PDF from successful images
    valid_images = [img for img in processed if img is not None]
    if not valid_images:
        raise HTTPException(500, "All images failed to process.")

    pdf_bytes = images_to_pdf(valid_images)
    pdf_path  = job_dir / "output.pdf"
    pdf_path.write_bytes(pdf_bytes)

    # Store job for re-crop requests
    _jobs[job_id] = {
        "originals": originals,
        "processed": processed,
        "quads":     quads,
        "opts":      opts,
    }

    processed_count = sum(1 for r in results if r.success)
    failed_count    = sum(1 for r in results if not r.success)

    return ScanResponse(
        job_id      = job_id,
        total       = len(files),
        processed   = processed_count,
        failed      = failed_count,
        images      = results,
        pdf_url     = f"/api/jobs/{job_id}/pdf",
        image_urls  = [
            f"/api/jobs/{job_id}/images/{r.index}"
            for r in results if r.success
        ],
    )


# ── POST /api/recrop ──────────────────────────────────────────────────────────

@router.post("/recrop")
async def recrop_image(req: RecropRequest):
    """
    Re-process a single image with user-adjusted corner points.
    Updates the job's processed image and regenerates the PDF.
    """
    job = _jobs.get(req.job_id)
    if not job:
        raise HTTPException(404, "Job not found. Jobs expire after server restart.")

    idx = req.image_index
    if idx < 0 or idx >= len(job["originals"]):
        raise HTTPException(400, f"Image index {idx} out of range.")

    original = job["originals"][idx]
    if original is None:
        raise HTTPException(400, f"Image {idx} failed original processing, cannot re-crop.")

    # User-supplied corners
    quad = _corners_to_array(req.corners)
    opts = req.options

    try:
        output = process_image(
            img_bgr  = original,
            quad     = quad,
            style    = OutputStyle(opts.output_style),
            shadow   = opts.shadow_removal,
            binarize = BinarizationMethod(opts.binarization),
        )
    except Exception as e:
        raise HTTPException(500, f"Re-crop failed: {e}")

    # Update job
    job["processed"][idx] = output
    job["quads"][idx]     = quad

    # Save updated image
    job_dir  = OUTPUT_DIR / req.job_id
    img_path = job_dir / f"{idx:04d}.jpg"
    cv2.imwrite(str(img_path), output, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # Regenerate PDF
    valid_images = [img for img in job["processed"] if img is not None]
    pdf_bytes    = images_to_pdf(valid_images)
    (job_dir / "output.pdf").write_bytes(pdf_bytes)

    return {
        "success":   True,
        "image_url": f"/api/jobs/{req.job_id}/images/{idx}",
        "pdf_url":   f"/api/jobs/{req.job_id}/pdf",
        "width":     output.shape[1],
        "height":    output.shape[0],
        "corners":   _array_to_corners(quad),
    }


# ── GET /api/jobs/{job_id}/pdf ────────────────────────────────────────────────

@router.get("/jobs/{job_id}/pdf")
async def download_pdf(job_id: str):
    pdf_path = OUTPUT_DIR / job_id / "output.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "PDF not found.")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename="scanned_documents.pdf",
    )


# ── GET /api/jobs/{job_id}/originals/{idx} ────────────────────────────────────

@router.get("/jobs/{job_id}/originals/{idx}")
async def get_original_image(job_id: str, idx: int):
    """Return the original uploaded image (used by CropEditor)."""
    orig_path = OUTPUT_DIR / job_id / f"{idx:04d}_orig.jpg"
    if not orig_path.exists():
        raise HTTPException(404, f"Original image {idx} not found in job {job_id}.")
    return FileResponse(
        str(orig_path),
        media_type="image/jpeg",
        filename=f"original_{idx+1}.jpg",
    )


# ── GET /api/jobs/{job_id}/images/{idx} ───────────────────────────────────────

@router.get("/jobs/{job_id}/images/{idx}")
async def get_image(job_id: str, idx: int):
    img_path = OUTPUT_DIR / job_id / f"{idx:04d}.jpg"
    if not img_path.exists():
        raise HTTPException(404, f"Image {idx} not found in job {job_id}.")
    return FileResponse(
        str(img_path),
        media_type="image/jpeg",
        filename=f"page_{idx+1}.jpg",
    )
