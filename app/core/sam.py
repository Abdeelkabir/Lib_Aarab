"""
app/core/sam.py
Loads MobileSAM once at startup, reuses across all requests.
Thread-safe singleton pattern.
"""
import threading
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_lock      = threading.Lock()
_predictor = None


def get_predictor():
    """
    Return the global SamPredictor instance.
    Loads model on first call, returns cached instance on subsequent calls.
    """
    global _predictor
    if _predictor is not None:
        return _predictor

    with _lock:
        # Double-checked locking — another thread may have loaded while we waited
        if _predictor is not None:
            return _predictor

        from app.core.config import WEIGHTS_PATH
        _predictor = _load_model(WEIGHTS_PATH)

    return _predictor


def _load_model(weights_path: Path):
    try:
        from mobile_sam import sam_model_registry, SamPredictor
    except ImportError:
        raise RuntimeError(
            "MobileSAM is not installed.\n"
            "Run: pip install git+https://github.com/ChaoningZhang/MobileSAM.git timm"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"MobileSAM weights not found at {weights_path}\n"
            f"Download: wget -O {weights_path} "
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        )

    device = _select_device()
    logger.info(f"Loading MobileSAM on {device} from {weights_path}")

    sam = sam_model_registry["vit_t"](checkpoint=str(weights_path))
    sam.to(device)
    sam.eval()

    logger.info("MobileSAM loaded and ready.")
    return SamPredictor(sam)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
