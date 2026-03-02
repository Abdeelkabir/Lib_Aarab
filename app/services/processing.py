"""
app/services/processing.py
All image transformation steps after detection:
  - Perspective warp
  - Output style (original / grayscale / morphological)
  - Shadow removal
  - Binarization (adaptive / sauvola)
"""
import cv2
import numpy as np
import logging
from typing import Optional
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_sauvola

from app.core.config import (
    MORPH_KERNEL_DIVISOR, GAUSS_SIGMA_DIVISOR,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
    ADAPTIVE_C, SAUVOLA_K,
    WARP_INTERP,
)
from app.models.schemas import OutputStyle, BinarizationMethod
from app.services.detection import order_points, image_corners

logger = logging.getLogger(__name__)

# ── Interpolation flag ────────────────────────────────────────────────────────

_INTERP_FLAGS = {
    "lanczos4": cv2.INTER_LANCZOS4,
    "cubic":    cv2.INTER_CUBIC,
    "linear":   cv2.INTER_LINEAR,
}
_INTERP = _INTERP_FLAGS.get(WARP_INTERP, cv2.INTER_LANCZOS4)


# ── Perspective warp ──────────────────────────────────────────────────────────

def warp_document(img: np.ndarray, quad: Optional[np.ndarray]) -> np.ndarray:
    """
    Apply perspective transform.
    If quad is None, returns the original image unchanged.
    """
    if quad is None:
        return img

    pts        = order_points(quad)
    tl, tr, br, bl = pts

    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))

    # Guard against degenerate quads
    if W < 10 or H < 10:
        logger.warning(f"Degenerate quad detected (W={W}, H={H}), skipping warp.")
        return img

    dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (W, H), flags=_INTERP)


# ── Shadow removal ────────────────────────────────────────────────────────────

def remove_shadows_morphological(gray: np.ndarray) -> np.ndarray:
    """
    Estimate background via morphological closing, divide to normalize.
    Best for broad, gradual illumination gradients.
    """
    ks     = max(21, min(gray.shape) // MORPH_KERNEL_DIVISOR)
    ks     = ks if ks % 2 == 1 else ks + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    bg     = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return cv2.divide(gray, bg, scale=255)


def remove_shadows_gaussian(gray: np.ndarray) -> np.ndarray:
    """
    Estimate background via heavy Gaussian blur, divide to normalize.
    Slightly smoother result than morphological.
    """
    sigma = max(10, min(gray.shape) // GAUSS_SIGMA_DIVISOR)
    bg    = np.clip(gaussian_filter(gray.astype(np.float32), sigma=sigma), 1, 255)
    return np.clip(gray.astype(np.float32) / bg * 255, 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """CLAHE — adaptive local contrast equalization."""
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE),
    )
    return clahe.apply(gray)


# ── Binarization ─────────────────────────────────────────────────────────────

def binarize_adaptive(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding.
    Computes local threshold per tile — handles residual illumination variation.
    """
    block = max(11, (gray.shape[1] // 20) | 1)   # ensure odd, ~1/20 width
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block,
        C=ADAPTIVE_C,
    )


def binarize_sauvola(gray: np.ndarray) -> np.ndarray:
    """
    Sauvola's method — uses local mean + std deviation.
    Best quality for noisy or textured paper.
    """
    window = max(11, (gray.shape[1] // 20) | 1)
    thresh = threshold_sauvola(gray, window_size=window, k=SAUVOLA_K)
    return ((gray > thresh).astype(np.uint8) * 255)


# ── Output style ──────────────────────────────────────────────────────────────

def apply_output_style(warped_bgr: np.ndarray, style: OutputStyle) -> np.ndarray:
    """
    Convert warped BGR image to the requested output style.
    Always returns a BGR image (for consistent downstream handling).
    """
    if style == OutputStyle.original:
        return warped_bgr

    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    if style == OutputStyle.grayscale:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if style == OutputStyle.morphological:
        result = remove_shadows_morphological(gray)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return warped_bgr


# ── Shadow removal dispatcher ─────────────────────────────────────────────────

def apply_shadow_removal(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply morphological shadow removal to a BGR image.
    Returns BGR image.
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    result = remove_shadows_morphological(gray)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ── Binarization dispatcher ───────────────────────────────────────────────────

def apply_binarization(img_bgr: np.ndarray, method: BinarizationMethod) -> np.ndarray:
    """
    Apply the chosen binarization method.
    Returns BGR image (white background, black text).
    """
    if method == BinarizationMethod.none:
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if method == BinarizationMethod.adaptive:
        result = binarize_adaptive(gray)
    elif method == BinarizationMethod.sauvola:
        result = binarize_sauvola(gray)
    else:
        return img_bgr

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


# ── Full single-image pipeline ────────────────────────────────────────────────

def process_image(
    img_bgr:    np.ndarray,
    quad:       Optional[np.ndarray],
    style:      OutputStyle,
    shadow:     bool,
    binarize:   BinarizationMethod,
) -> np.ndarray:
    """
    Run the full processing pipeline on a single image.

    Steps:
        1. Perspective warp
        2. Output style (original / grayscale / morphological)
        3. Shadow removal (optional)
        4. Binarization (optional)
    """
    # 1. Warp
    warped = warp_document(img_bgr, quad)

    # 2. Output style
    result = apply_output_style(warped, style)

    # 3. Shadow removal
    if shadow and style != OutputStyle.morphological:
        # morphological style already includes shadow removal
        result = apply_shadow_removal(result)

    # 4. Binarization
    if binarize != BinarizationMethod.none:
        result = apply_binarization(result, binarize)

    return result
