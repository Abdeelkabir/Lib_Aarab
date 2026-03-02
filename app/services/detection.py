"""
app/services/detection.py
SAM-based document edge detection.
Runs 4 prompt strategies, pools masks, deduplicates, ranks by weighted score.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

from app.core.config import (
    SAM_W_CONFIDENCE, SAM_W_COVERAGE, SAM_W_RECT,
    SAM_COV_MIN, SAM_COV_MAX,
)

logger = logging.getLogger(__name__)


# ── Prompt strategy definitions ───────────────────────────────────────────────

def _build_strategies(h: int, w: int) -> Dict[str, dict]:
    cx, cy = w // 2, h // 2
    return {
        "center_5": {
            "color": "#FF4444",
            "points": np.array([
                [cx,          cy          ],
                [cx,          int(cy*0.6) ],
                [cx,          int(cy*1.4) ],
                [int(cx*0.6), cy          ],
                [int(cx*1.4), cy          ],
            ]),
        },
        "grid_9": {
            "color": "#FF9900",
            "points": np.array([
                [int(w*0.25), int(h*0.25)], [cx, int(h*0.25)], [int(w*0.75), int(h*0.25)],
                [int(w*0.25), cy],          [cx, cy],           [int(w*0.75), cy],
                [int(w*0.25), int(h*0.75)],[cx, int(h*0.75)],  [int(w*0.75), int(h*0.75)],
            ]),
        },
        "tight_center": {
            "color": "#00CC66",
            "points": np.array([
                [cx,               cy              ],
                [cx,               int(cy*0.8)     ],
                [cx,               int(cy*1.2)     ],
                [int(cx*0.8),      cy              ],
                [int(cx*1.2),      cy              ],
                [int(cx*0.85),     int(cy*0.85)    ],
                [int(cx*1.15),     int(cy*1.15)    ],
            ]),
        },
        "corner_guards": {
            "color": "#4488FF",
            "points": np.array([
                [cx,           cy          ],
                [int(w*0.3),   int(h*0.3)  ],
                [int(w*0.7),   int(h*0.3)  ],
                [int(w*0.3),   int(h*0.7)  ],
                [int(w*0.7),   int(h*0.7)  ],
            ]),
        },
    }


# ── IoU deduplication ─────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a  > 0, b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def _deduplicate(candidates: List[dict], iou_threshold: float = 0.92) -> List[dict]:
    unique = []
    for cand in candidates:
        if not any(_iou(cand["mask"], u["mask"]) > iou_threshold for u in unique):
            unique.append(cand)
    return unique


# ── Geometry stats (immutable, independent of weights) ────────────────────────

def _geometry_stats(mask: np.ndarray, img_area: int) -> dict:
    coverage = float(mask.sum()) / (255 * img_area)
    ys, xs   = np.where(mask)
    if len(xs) == 0:
        return {"coverage": coverage, "rectangularity": 0.0}
    bbox_area   = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
    rect        = float(mask.sum()) / (255 * bbox_area) if bbox_area > 0 else 0.0
    return {"coverage": coverage, "rectangularity": rect}


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    sam_score: float,
    coverage: float,
    rectangularity: float,
    w_sam: float,
    w_cov: float,
    w_rect: float,
    cov_min: float,
    cov_max: float,
) -> Tuple[float, bool]:
    in_range = cov_min < coverage < cov_max
    if not in_range:
        return -1.0, False
    total = w_sam + w_cov + w_rect or 1.0
    score = (w_sam * sam_score + w_cov * coverage + w_rect * rectangularity) / total
    return score, True


# ── Corner ordering ───────────────────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    d    = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    rect[1] = pts[np.argmin(d)]   # TR
    rect[3] = pts[np.argmax(d)]   # BL
    return rect


# ── Mask → quadrilateral ──────────────────────────────────────────────────────

def mask_to_quad(mask: np.ndarray) -> Optional[np.ndarray]:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  k)
    m = cv2.morphologyEx(m,    cv2.MORPH_DILATE, k)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c    = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)

    for eps in [0.02, 0.03, 0.05, 0.08]:
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2).astype(np.float32))

    # Fallback: minimum area rectangle
    box = cv2.boxPoints(cv2.minAreaRect(c))
    return order_points(box.astype(np.float32))


def image_corners(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)


# ── Main detection entry point ────────────────────────────────────────────────

def detect_document(
    image_rgb: np.ndarray,
    predictor,
    w_sam:    float = SAM_W_CONFIDENCE,
    w_cov:    float = SAM_W_COVERAGE,
    w_rect:   float = SAM_W_RECT,
    cov_min:  float = SAM_COV_MIN,
    cov_max:  float = SAM_COV_MAX,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run SAM on image_rgb, rank all masks, return:
        (quad: float32 [4,2], best_mask: uint8 HxW)
    Returns (None, None) if no valid mask found — caller should fall back to full image.
    """
    h, w      = image_rgb.shape[:2]
    img_area  = h * w
    strategies = _build_strategies(h, w)

    predictor.set_image(image_rgb)

    raw: List[dict] = []
    for name, cfg in strategies.items():
        pts    = cfg["points"]
        labels = np.ones(len(pts), dtype=int)
        try:
            masks, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=labels,
                multimask_output=True,
            )
        except Exception as e:
            logger.warning(f"SAM predict failed for strategy {name}: {e}")
            continue

        for i, (m, s) in enumerate(zip(masks, scores)):
            m_uint = m.astype(np.uint8) * 255
            stats  = _geometry_stats(m_uint, img_area)
            raw.append({
                "mask":           m_uint,
                "sam_score":      float(s),
                "strategy":       name,
                "mask_idx":       i + 1,
                **stats,
            })

    if not raw:
        logger.warning("No masks returned by SAM at all.")
        return None, None

    # Deduplicate
    unique = _deduplicate(raw)

    # Score and sort
    for r in unique:
        r["combined"], r["in_range"] = _score(
            r["sam_score"], r["coverage"], r["rectangularity"],
            w_sam, w_cov, w_rect, cov_min, cov_max,
        )

    unique.sort(key=lambda x: x["combined"], reverse=True)

    # Pick winner
    winner = unique[0]
    if not winner["in_range"]:
        logger.warning(f"Best mask out of coverage range ({winner['coverage']*100:.1f}%). Using full image.")
        return None, None

    quad = mask_to_quad(winner["mask"])
    return quad, winner["mask"]
