"""
app/models/schemas.py
All request/response data models.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class OutputStyle(str, Enum):
    original   = "original"    # color, perspective-corrected
    grayscale  = "grayscale"   # grayscale, perspective-corrected
    morphological = "morphological"  # morph shadow removal


class BinarizationMethod(str, Enum):
    none     = "none"       # no binarization
    adaptive = "adaptive"
    sauvola  = "sauvola"


class ProcessingOptions(BaseModel):
    output_style:        OutputStyle        = OutputStyle.original
    shadow_removal:      bool               = False
    binarization:        BinarizationMethod = BinarizationMethod.none

    class Config:
        use_enum_values = True


# ── Per-image results ─────────────────────────────────────────────────────────

class CropCorners(BaseModel):
    """4 corner points of the detected document quad, in image coordinates."""
    tl: List[float] = Field(..., description="Top-left  [x, y]")
    tr: List[float] = Field(..., description="Top-right [x, y]")
    br: List[float] = Field(..., description="Bottom-right [x, y]")
    bl: List[float] = Field(..., description="Bottom-left  [x, y]")


class ImageResult(BaseModel):
    index:          int
    filename:       str
    success:        bool
    error:          Optional[str]   = None
    corners:        Optional[CropCorners] = None   # detected quad (for manual re-crop)
    width:          Optional[int]   = None          # output image dimensions
    height:         Optional[int]   = None


class ScanResponse(BaseModel):
    job_id:         str
    total:          int
    processed:      int
    failed:         int
    images:         List[ImageResult]
    pdf_url:        str             # URL to download the assembled PDF
    image_urls:     List[str]       # individual processed image URLs


# ── Manual re-crop ────────────────────────────────────────────────────────────

class RecropRequest(BaseModel):
    job_id:     str
    image_index: int
    corners:    CropCorners         # user-adjusted corners
    options:    ProcessingOptions
