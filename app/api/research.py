"""
app/api/research.py
AI-powered research article generation via OpenAI.

Routes:
  POST /api/research/generate   — generate article from topic + language
  POST /api/research/pdf        — export article dict to PDF, return file
  GET  /api/research/languages  — list supported languages + metadata
"""
import io
import os
import re
import json
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/research", tags=["research"])

# ── Supported languages ───────────────────────────────────────────────────────

LANGUAGES = {
    "arabic": {
        "label":     "العربية",
        "flag":      "🇦🇪",
        "direction": "rtl",
        "font":      "Amiri",
        "align":     "R",
    },
    "french": {
        "label":     "Français",
        "flag":      "🇫🇷",
        "direction": "ltr",
        "font":      "OpenSans",
        "align":     "L",
    },
    "english": {
        "label":     "English",
        "flag":      "🇬🇧",
        "direction": "ltr",
        "font":      "OpenSans",
        "align":     "L",
    },
}

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "arabic": """أنت مساعد في مكتبة مدرسية. مهمتك كتابة بحوث مدرسية باللغة العربية الفصحى للطلاب.

لكل موضوع يُعطى لك، اكتب بحثاً يحقق المعايير التالية:
- مكتوب باللغة العربية الفصحى السليمة
- مناسب لطلاب المدارس
- متوسط الطول: ليس قصيراً جداً ولا طويلاً جداً (٤ إلى ٦ فقرات)
- واضح وغني بالمعلومات وسهل الفهم
- منظم بعنوان رئيسي وعدة فقرات

أعد الرد كائن JSON فقط بالبنية التالية:
{
  "title": "عنوان البحث",
  "par_1": "الفقرة الأولى",
  "par_2": "الفقرة الثانية",
  "par_3": "الفقرة الثالثة",
  ...
}

⚠️ لا تُضف أي شرح أو تعليق أو نص إضافي — أعد كائن JSON فقط.""",

    "french": """Vous êtes un assistant dans une bibliothèque scolaire. Votre mission est de rédiger des exposés de recherche en français pour les élèves.

Pour chaque sujet donné, rédigez un exposé qui respecte ces critères :
- Écrit en français correct et accessible
- Adapté aux élèves du secondaire
- Longueur moyenne : 4 à 6 paragraphes
- Clair, informatif et facile à comprendre
- Structuré avec un titre et plusieurs paragraphes

Retournez uniquement un objet JSON avec la structure suivante :
{
  "title": "Titre de l'exposé",
  "par_1": "Premier paragraphe",
  "par_2": "Deuxième paragraphe",
  ...
}

⚠️ Ne fournissez aucune explication ou texte supplémentaire — retournez uniquement le JSON.""",

    "english": """You are an assistant at a school library. Your task is to write research articles in English for school students.

For each topic provided, write an article that follows these rules:
- Written in clear, simple English
- Suitable for school students
- Medium length: 4 to 6 paragraphs
- Clear, informative, and easy to understand
- Organized with a title and multiple paragraphs

Return only a JSON object with this structure:
{
  "title": "Article Title",
  "par_1": "First paragraph",
  "par_2": "Second paragraph",
  ...
}

⚠️ Do not include any explanation or extra text — return only the JSON.""",
}

# ── Request / Response schemas ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    topic:    str
    language: str = "arabic"
    api_key:  str              # passed per-request so we never store keys server-side


class ArticleData(BaseModel):
    title: str
    paragraphs: list[str]     # ordered list — simpler than par_1, par_2 on the wire
    language: str


class PdfRequest(BaseModel):
    article:  ArticleData
    api_key:  Optional[str] = None   # not needed for PDF, kept for interface consistency


# ── Language metadata endpoint ────────────────────────────────────────────────

@router.get("/languages")
async def get_languages():
    return [
        {
            "code":      code,
            "label":     meta["label"],
            "flag":      meta["flag"],
            "direction": meta["direction"],
        }
        for code, meta in LANGUAGES.items()
    ]


# ── Generate (streaming SSE) ──────────────────────────────────────────────────

@router.post("/generate")
async def generate_article(req: GenerateRequest):
    """
    Stream article generation as Server-Sent Events.
    Each event is a JSON chunk: { type: "token"|"done"|"error", data: ... }

    On "done": data = ArticleData
    On "token": data = string chunk (for typing effect in UI)
    On "error": data = error message string
    """
    lang = req.language.lower()
    if lang not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language '{lang}'. Choose from: {list(LANGUAGES.keys())}")
    if not req.topic.strip():
        raise HTTPException(400, "Topic cannot be empty.")
    if not req.api_key.strip():
        raise HTTPException(400, "OpenAI API key is required.")

    return StreamingResponse(
        _stream_article(req.topic.strip(), lang, req.api_key.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


async def _stream_article(topic: str, lang: str, api_key: str):
    """Async generator that yields SSE-formatted strings."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        yield _sse("error", "openai package not installed. Run: pip install openai")
        return

    client = AsyncOpenAI(api_key=api_key)

    full_text = ""
    try:
        stream = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[lang]},
                {"role": "user",   "content": topic},
            ],
            stream=True,
            temperature=0.7,
            max_tokens=2000,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_text += delta
                yield _sse("token", delta)

    except Exception as e:
        logger.error(f"OpenAI streaming error: {e}")
        yield _sse("error", str(e))
        return

    # Parse the accumulated JSON
    try:
        article_dict = _parse_article_json(full_text)
        article = ArticleData(
            title=article_dict["title"],
            paragraphs=[
                article_dict[k]
                for k in sorted(article_dict)
                if k.startswith("par_")
            ],
            language=lang,
        )
        yield _sse("done", article.model_dump())
    except Exception as e:
        logger.error(f"JSON parse error: {e}\nRaw: {full_text}")
        yield _sse("error", f"Could not parse article structure: {e}")


def _sse(event_type: str, data) -> str:
    payload = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _parse_article_json(raw: str) -> dict:
    """
    Extract JSON from the model response, handling markdown code blocks
    and minor formatting issues.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$",          "", cleaned,     flags=re.MULTILINE)
    cleaned = cleaned.strip()

    parsed = json.loads(cleaned)

    # Validate minimum structure
    if "title" not in parsed:
        raise ValueError("Missing 'title' key in response.")
    if not any(k.startswith("par_") for k in parsed):
        raise ValueError("No 'par_N' keys found in response.")

    return parsed


# ── PDF export ────────────────────────────────────────────────────────────────

@router.post("/pdf")
async def export_pdf(req: PdfRequest):
    """
    Generate a PDF from a structured article and return it as a download.
    Handles Arabic (RTL + text shaping) and LTR languages.
    """
    article = req.article
    lang    = article.language.lower()

    if lang not in LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {lang}")

    try:
        pdf_bytes = _build_pdf(article, lang)
    except FileNotFoundError as e:
        raise HTTPException(500, f"Font file missing: {e}")
    except Exception as e:
        logger.error(f"PDF generation error: {e}", exc_info=True)
        raise HTTPException(500, f"PDF generation failed: {e}")

    # ASCII fallback (latin-1 safe for HTTP headers)
    ascii_title = article.title.encode("ascii", "ignore").decode()
    safe_title  = re.sub(r"[^\w\s-]", "", ascii_title)[:40].strip().replace(" ", "_") or "article"
    ascii_name  = f"{lang}_{safe_title}.pdf"

    # RFC 5987 UTF-8 name — supports Arabic, French accents etc.
    from urllib.parse import quote as urlquote
    utf8_name = urlquote(f"{lang}_{article.title[:60]}.pdf", safe="")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"
        },
    )


def _build_pdf(article: ArticleData, lang: str) -> bytes:
    """Build PDF bytes using fpdf2. Requires font .ttf files in ./fonts/"""
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("fpdf2 not installed. Run: pip install fpdf2")

    # Font file paths — put .ttf files in backend/fonts/
    FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fonts")
    FONT_DIR = os.path.normpath(FONT_DIR)

    FONT_FILES = {
        "arabic": {
            "regular": os.path.join(FONT_DIR, "Amiri-Regular.ttf"),
            "bold":    os.path.join(FONT_DIR, "Amiri-Bold.ttf"),
            "family":  "Amiri",
        },
        "english": {
            "regular": os.path.join(FONT_DIR, "OpenSans_Condensed-Regular.ttf"),
            "bold":    os.path.join(FONT_DIR, "OpenSans_Condensed-Bold.ttf"),
            "family":  "OpenSans",
        },
        "french": {
            "regular": os.path.join(FONT_DIR, "OpenSans_Condensed-Regular.ttf"),
            "bold":    os.path.join(FONT_DIR, "OpenSans_Condensed-Bold.ttf"),
            "family":  "OpenSans",
        },
    }

    cfg     = FONT_FILES[lang]
    is_rtl  = lang == "arabic"
    align   = LANGUAGES[lang]["align"]

    for label, path in [("regular", cfg["regular"]), ("bold", cfg["bold"])]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Font file not found: {path}\n"
                f"Place your .ttf files in: {FONT_DIR}"
            )

    pdf = FPDF()
    pdf.set_margins(left=20, top=20, right=20)
    pdf.add_page()

    if is_rtl:
        pdf.set_text_shaping(True)

    # Register fonts
    pdf.add_font(cfg["family"], "",  cfg["regular"])
    pdf.add_font(cfg["family"], "B", cfg["bold"])

    # Title
    pdf.set_font(cfg["family"], style="B", size=22)
    pdf.cell(0, 16, txt=article.title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Decorative line under title
    pdf.set_draw_color(245, 166, 35)   # amber — matches app accent color
    pdf.set_line_width(0.8)
    pdf.line(20, pdf.get_y(), pdf.w - 20, pdf.get_y())
    pdf.ln(10)

    # Paragraphs
    pdf.set_font(cfg["family"], size=13)
    for para in article.paragraphs:
        if not para.strip():
            continue
        pdf.multi_cell(0, 8, txt=para.strip(), align=align)
        pdf.ln(5)

    # Footer
    pdf.set_y(-18)
    pdf.set_font(cfg["family"], size=9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 8, txt=article.title, align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.read()
