"""
app/services/pdf_builder.py
Assemble a list of processed BGR images into a single PDF.
Uses img2pdf for lossless/high-quality embedding.
"""
import cv2
import img2pdf
import io
import logging
from typing import List
from PIL import Image

from app.core.config import JPEG_QUALITY, PDF_DPI

logger = logging.getLogger(__name__)


def images_to_pdf(images_bgr: List[any]) -> bytes:
    """
    Convert a list of BGR numpy arrays into a single PDF (bytes).

    Each image is encoded as JPEG at JPEG_QUALITY, then assembled
    by img2pdf which preserves the full resolution without re-sampling.

    Returns raw PDF bytes ready to send as a response or write to disk.
    """
    if not images_bgr:
        raise ValueError("No images provided to PDF builder.")

    image_bytes_list = []

    for i, img_bgr in enumerate(images_bgr):
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            buf = io.BytesIO()
            pil_img.save(
                buf,
                format="JPEG",
                quality=JPEG_QUALITY,
                dpi=(PDF_DPI, PDF_DPI),
                optimize=True,
            )
            buf.seek(0)
            image_bytes_list.append(buf.read())

        except Exception as e:
            logger.error(f"Failed to encode image {i} for PDF: {e}")
            raise

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
    except Exception as e:
        logger.error(f"img2pdf assembly failed: {e}")
        raise

    return pdf_bytes
