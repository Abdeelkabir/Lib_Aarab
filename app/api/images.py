"""
app/api/images.py
Image search (Bing) + download + PDF export endpoints.

Routes:
  GET  /api/images/search?q=...&limit=20   — search Bing, return metadata list
  POST /api/images/fetch                   — download a single image by URL (proxy)
  POST /api/images/pdf                     — assemble selected images into PDF
"""
import io
import re
import uuid
import logging
import requests
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from bs4 import BeautifulSoup
from PIL import Image as PILImage
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images", tags=["images"])

# ── Constants ─────────────────────────────────────────────────────────────────
SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
}
FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "image/*,*/*",
    "Referer": "https://www.bing.com/",
}
ALLOWED_EXTS   = {"jpg", "jpeg", "png", "webp"}
MAX_FETCH_BYTES = 15 * 1024 * 1024   # 15 MB per image
REQUEST_TIMEOUT = 10


# ── Schemas ───────────────────────────────────────────────────────────────────

class ImageMeta(BaseModel):
    id:        str
    url:       str               # original image URL
    thumb_url: Optional[str]     # Bing thumbnail URL
    title:     Optional[str]
    source:    Optional[str]     # source page URL
    width:     Optional[int]
    height:    Optional[int]


class FetchRequest(BaseModel):
    url: str


class PdfRequest(BaseModel):
    urls:            List[str]         # ordered list of image URLs to include
    images_per_page: int = 4           # 1–12
    columns:         int = 2           # 1–4
    page_size:       str = "A4"        # A4 | Letter
    margin_mm:       float = 10.0


# ── Search ────────────────────────────────────────────────────────────────────

@router.get("/search", response_model=List[ImageMeta])
async def search_images(
    q:     str = Query(..., min_length=1, max_length=200),
    limit: int = Query(default=20, ge=1, le=50),
):
    """
    Search Bing Images and return metadata for up to `limit` results.
    Images are NOT downloaded here — only metadata + URLs are returned.
    The frontend fetches thumbnails directly; full images go through /fetch.
    """
    if not q.strip():
        raise HTTPException(400, "Query cannot be empty.")

    encoded = quote(q.strip())
    url     = f"https://www.bing.com/images/search?q={encoded}&form=HDRSC3&first=1"

    try:
        resp = requests.get(url, headers=SEARCH_HEADERS, timeout=REQUEST_TIMEOUT, verify=False)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Bing search failed: {e}")
        raise HTTPException(502, f"Search request failed: {e}")

    results = _parse_bing_results(resp.text, limit)
    logger.info(f"Search '{q}' → {len(results)} results")
    return results


def _parse_bing_results(html: str, limit: int) -> List[ImageMeta]:
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all("a", {"class": "iusc"})
    out  = []

    for tag in tags:
        if len(out) >= limit:
            break
        try:
            m_attr = tag.get("m", "")

            # Full image URL
            murl_m = re.search(r'"murl"\s*:\s*"(.*?)"', m_attr)
            if not murl_m:
                continue
            img_url = murl_m.group(1)

            # Validate extension
            base_url = img_url.split("?")[0]
            ext = base_url.rsplit(".", 1)[-1].lower()
            if ext not in ALLOWED_EXTS:
                ext = "jpg"

            # Thumbnail URL
            turl_m = re.search(r'"turl"\s*:\s*"(.*?)"', m_attr)
            thumb  = turl_m.group(1) if turl_m else None

            # Title / source
            title_m  = re.search(r'"t"\s*:\s*"(.*?)"',    m_attr)
            source_m = re.search(r'"purl"\s*:\s*"(.*?)"', m_attr)
            title    = title_m.group(1)  if title_m  else None
            source   = source_m.group(1) if source_m else None

            # Dimensions
            w_m = re.search(r'"iw"\s*:\s*(\d+)', m_attr)
            h_m = re.search(r'"ih"\s*:\s*(\d+)', m_attr)
            w   = int(w_m.group(1)) if w_m else None
            h   = int(h_m.group(1)) if h_m else None

            out.append(ImageMeta(
                id        = uuid.uuid4().hex[:10],
                url       = img_url,
                thumb_url = thumb,
                title     = title,
                source    = source,
                width     = w,
                height    = h,
            ))
        except Exception as e:
            logger.debug(f"Skipping result: {e}")
            continue

    return out


# ── Proxy fetch (CORS-safe, validates content) ────────────────────────────────

@router.post("/fetch")
async def fetch_image(req: FetchRequest):
    """
    Download a single image by URL and return it.
    Acts as a proxy so the frontend avoids CORS issues with third-party hosts.
    Validates that the response is actually an image.
    """
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "Invalid URL.")

    try:
        resp = requests.get(
            url,
            headers=FETCH_HEADERS,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            verify=False,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            raise HTTPException(422, "URL returned HTML, not an image.")

        data = b""
        for chunk in resp.iter_content(chunk_size=65536):
            data += chunk
            if len(data) > MAX_FETCH_BYTES:
                raise HTTPException(413, "Image too large (>15MB).")

        # Validate it's actually an image
        try:
            img = PILImage.open(io.BytesIO(data))
            img.verify()
        except Exception:
            raise HTTPException(422, "Downloaded content is not a valid image.")

        # Re-open after verify (verify closes the file)
        img = PILImage.open(io.BytesIO(data))
        fmt = img.format or "JPEG"
        mime = f"image/{fmt.lower().replace('jpeg', 'jpeg')}"

        return Response(content=data, media_type=mime)

    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(502, f"Failed to fetch image: {e}")


# ── PDF export ────────────────────────────────────────────────────────────────

@router.post("/pdf")
async def export_pdf(req: PdfRequest):
    """
    Download all provided image URLs, assemble into a paginated PDF, return bytes.
    Layout: grid of (images_per_page, columns) per A4/Letter page.
    Full resolution — images are only resized to fit slots, not downsampled beyond that.
    """
    import math
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader

    # Page size
    page_size = A4 if req.page_size == "A4" else LETTER
    pw, ph    = page_size

    # Layout math (all in reportlab points, 1pt = 1/72 inch)
    mm_to_pt = 72 / 25.4
    margin   = req.margin_mm * mm_to_pt
    spacing  = 6.0   # fixed 6pt gap between slots
    cols     = max(1, min(req.columns, 6))
    ipp      = max(1, min(req.images_per_page, 20))
    rows     = math.ceil(ipp / cols)

    slot_w = (pw - 2 * margin - (cols - 1) * spacing) / cols
    slot_h = (ph - 2 * margin - (rows - 1) * spacing) / rows

    # Download all images first
    images_data: List[Optional[bytes]] = []
    for url in req.urls:
        try:
            resp = requests.get(
                url,
                headers=FETCH_HEADERS,
                timeout=REQUEST_TIMEOUT,
                verify=False,
            )
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            if "text/html" in ct:
                images_data.append(None)
                continue
            images_data.append(resp.content)
        except Exception as e:
            logger.warning(f"Could not fetch {url}: {e}")
            images_data.append(None)

    valid = [(i, d) for i, d in enumerate(images_data) if d is not None]
    if not valid:
        raise HTTPException(422, "None of the provided URLs returned valid images.")

    # Build PDF
    pdf_buf = io.BytesIO()
    c = rl_canvas.Canvas(pdf_buf, pagesize=page_size)

    slot_idx = 0
    for _, img_data in valid:
        if slot_idx > 0 and slot_idx % ipp == 0:
            c.showPage()

        pos_in_page = slot_idx % ipp
        row = pos_in_page // cols
        col = pos_in_page % cols

        x = margin + col * (slot_w + spacing)
        # reportlab y=0 is bottom; we fill top-to-bottom
        y = ph - margin - (row + 1) * slot_h - row * spacing

        try:
            pil_img = PILImage.open(io.BytesIO(img_data)).convert("RGB")
            iw, ih  = pil_img.size
            i_aspect = ih / iw
            s_aspect = slot_h / slot_w

            if i_aspect > s_aspect:
                dh = slot_h
                dw = dh / i_aspect
            else:
                dw = slot_w
                dh = dw * i_aspect

            x_off = (slot_w - dw) / 2
            y_off = (slot_h - dh) / 2

            img_io = io.BytesIO()
            pil_img.save(img_io, format="JPEG", quality=95)
            img_io.seek(0)
            reader = ImageReader(img_io)

            c.drawImage(reader, x + x_off, y + y_off, width=dw, height=dh)

        except Exception as e:
            logger.warning(f"Could not render image to PDF: {e}")

        slot_idx += 1

    c.save()
    pdf_buf.seek(0)

    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="images.pdf"'},
    )
