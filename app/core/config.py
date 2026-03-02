"""
app/core/config.py
Central configuration — all tuneable constants in one place.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = BASE_DIR / "weights" / "mobile_sam.pt"

# ── SAM scoring (from your notebook) ─────────────────────────────────────────
SAM_W_CONFIDENCE  = 0.50
SAM_W_COVERAGE    = 0.20
SAM_W_RECT        = 0.30
SAM_COV_MIN       = 0.15
SAM_COV_MAX       = 0.95

# ── SAM prompt strategies ─────────────────────────────────────────────────────
# Each strategy runs one predict() call → 3 masks.
# All results are pooled and deduplicated (IoU > 0.92).
SAM_STRATEGIES = ["center_5", "grid_9", "tight_center", "corner_guards"]

# ── Image quality ─────────────────────────────────────────────────────────────
JPEG_QUALITY   = 97          # output JPEG quality (0–100)
PDF_DPI        = 300         # DPI embedded in PDF metadata
WARP_INTERP    = "lanczos4"  # perspective warp interpolation

# ── Shadow removal ────────────────────────────────────────────────────────────
MORPH_KERNEL_DIVISOR = 8     # kernel size = min(h,w) // this value
GAUSS_SIGMA_DIVISOR  = 6     # sigma = min(h,w) // this value
CLAHE_CLIP_LIMIT     = 2.5
CLAHE_TILE_SIZE      = 8

# ── Binarization ─────────────────────────────────────────────────────────────
ADAPTIVE_C           = 10    # constant for adaptive threshold
SAUVOLA_K            = 0.2   # sensitivity for Sauvola

# ── API ───────────────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB   = 30
MAX_IMAGES_PER_BATCH = 20
ALLOWED_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif"}
