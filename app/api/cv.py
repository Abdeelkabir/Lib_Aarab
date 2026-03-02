"""
app/api/cv.py
CV generation: OpenAI structured JSON + ReportLab PDF rendering.

Routes:
  POST /api/cv/generate  -- parse free-text into structured CV JSON
  POST /api/cv/pdf       -- render CV JSON to PDF (blue or pink template)
"""
import io
import re
import json
import logging
from urllib.parse import quote as urlquote
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cv", tags=["cv"])

# --------------------------------------------------------------------------- #
#  Pydantic schemas                                                            #
# --------------------------------------------------------------------------- #

class ContactData(BaseModel):
    phone:    str = ""
    email:    str = ""
    location: str = ""

class EducationEntry(BaseModel):
    title:  str = ""
    school: str = ""
    years:  str = ""

class ExperienceEntry(BaseModel):
    title:       str = ""
    company:     str = ""
    years:       str = ""
    description: List[str] = []

class LanguageEntry(BaseModel):
    name:  str = ""
    level: str = ""

class CVData(BaseModel):
    name:        str = ""
    contact:     ContactData = ContactData()
    bio:         str = ""
    education:   List[EducationEntry] = []
    experience:  List[ExperienceEntry] = []
    competences: List[str] = []
    languages:   List[LanguageEntry] = []

class GenerateCVRequest(BaseModel):
    message: str
    history: List[dict] = []
    api_key: str

class RenderPDFRequest(BaseModel):
    cv_data:  CVData
    template: str = "blue"

# --------------------------------------------------------------------------- #
#  System prompt                                                               #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "Tu es un assistant specialise dans la creation de CV professionnels. "
    "Extrait les informations de l'utilisateur et retourne UNIQUEMENT un JSON valide.\n\n"
    "Structure JSON:\n"
    '{\n'
    '  "name": "NOM COMPLET EN MAJUSCULES",\n'
    '  "contact": {"phone": "", "email": "", "location": "Ville, Pays"},\n'
    '  "bio": "Courte phrase de motivation. Si absente: '
    'Motive et dynamique, je cherche a mettre mes competences au service '
    "d'une entreprise ambitieuse.\",\n"
    '  "education": [{"title": "", "school": "", "years": "YYYY - YYYY"}],\n'
    '  "experience": [{"title": "", "company": "", "years": "YYYY - YYYY", '
    '"description": ["tache 1", "tache 2", "tache 3"]}],\n'
    '  "competences": ["comp1","comp2","comp3","comp4","comp5","comp6"],\n'
    '  "languages": [{"name": "Arabe", "level": "Langue maternelle"}]\n'
    '}\n\n'
    "Regles:\n"
    "- competences: EXACTEMENT 6 elements, soft skills courts (2-3 mots). "
    "Exemples: Sens de l ecoute, Bon relationnel, Travail en equipe, Rigueur.\n"
    "- experience.description: 2-3 taches tres courtes. Si poste inconnu, invente des taches realistes.\n"
    "- Ne mets jamais 'Non specifie'. Si info absente, laisse vide.\n"
    "- Si seul le nom d une entreprise est donne, titre = 'Collaborateur au sein de [entreprise]'.\n"
    "- Permis de conduire => education avec title = type du permis, school = ''.\n"
    "- L utilisateur peut donner les infos progressivement: fusionne avec ce que tu sais deja.\n"
    "- Reponds TOUJOURS avec le JSON complet. Jamais de texte en dehors du JSON."
)

# --------------------------------------------------------------------------- #
#  /generate                                                                   #
# --------------------------------------------------------------------------- #

@router.post("/generate")
async def generate_cv(req: GenerateCVRequest):
    if not req.api_key.strip():
        raise HTTPException(400, "OpenAI API key is required.")
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise HTTPException(500, "openai package not installed. Run: pip install openai")

    client = AsyncOpenAI(api_key=req.api_key.strip())

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history[-10:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": req.message.strip()})

    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("OpenAI error: %s", e)
        raise HTTPException(502, "OpenAI request failed: " + str(e))

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        cv_dict = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s\nRaw: %s", e, raw)
        raise HTTPException(422, "Could not parse CV JSON: " + str(e))

    try:
        cv = CVData(**cv_dict)
    except Exception as e:
        raise HTTPException(422, "CV validation failed: " + str(e))

    return {"cv": cv.model_dump()}


# --------------------------------------------------------------------------- #
#  /pdf                                                                        #
# --------------------------------------------------------------------------- #

@router.post("/pdf")
async def render_pdf(req: RenderPDFRequest):
    template = req.template.lower()
    if template not in ("blue", "pink"):
        raise HTTPException(400, "template must be 'blue' or 'pink'")
    try:
        pdf_bytes = _build_blue_cv(req.cv_data) if template == "blue" else _build_pink_cv(req.cv_data)
    except FileNotFoundError as e:
        raise HTTPException(500, "Font missing: " + str(e))
    except Exception as e:
        logger.error("PDF render error: %s", e, exc_info=True)
        raise HTTPException(500, "PDF generation failed: " + str(e))

    aname = re.sub(r"[^\w\s-]", "", req.cv_data.name.encode("ascii","ignore").decode())[:40].strip().replace(" ","_") or "cv"
    ascii_file = "cv_" + aname + "_" + template + ".pdf"
    utf8_file  = urlquote("cv_" + req.cv_data.name + "_" + template + ".pdf", safe="")

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=\"" + ascii_file + "\"; filename*=UTF-8''" + utf8_file},
    )


# --------------------------------------------------------------------------- #
#  Blue & White template                                                       #
# --------------------------------------------------------------------------- #

def _build_blue_cv(cv):
    import os
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem

    FONT_DIR = _font_dir()
    _reg("Inter",     os.path.join(FONT_DIR, "Inter-Regular.ttf"), pdfmetrics)
    _reg("InterBold", os.path.join(FONT_DIR, "Inter-Bold.ttf"),    pdfmetrics)

    PAGE_W, PAGE_H = A4
    MARGIN    = 1.25 * cm
    CONTENT_W = PAGE_W - 2 * MARGIN
    BLUE = HexColor("#004081")
    DARK = HexColor("#1e1e1e")
    GRAY = HexColor("#666666")

    from reportlab.lib.styles import ParagraphStyle as PS
    def S(n, f="Inter", sz=10.5, col=DARK, lead=14, **kw):
        return PS(n, fontName=f, fontSize=sz, textColor=col, leading=lead, **kw)

    story = []
    sp = lambda h=6: story.append(Spacer(1, h))
    hr = lambda w=1.2, col=BLUE: story.append(_HRule(CONTENT_W, col, w))

    def section(label):
        sp(10)
        story.append(Paragraph(label, S("sec","InterBold",11,BLUE,16)))
        hr(0.8)
        sp(4)

    story.append(Paragraph(cv.name or "NOM PRENOM", S("nm","InterBold",26,BLUE,30)))
    sp(4)
    parts = [p for p in [cv.contact.phone, cv.contact.email, cv.contact.location] if p]
    story.append(Paragraph("  |  ".join(parts), S("ct","Inter",10,GRAY,13)))
    hr(2.0)
    sp(4)

    if cv.bio:
        section("PROFIL")
        story.append(Paragraph(cv.bio, S("bio","Inter",10,DARK,15,leftIndent=4)))

    if cv.experience:
        section("EXPERIENCE")
        for ex in cv.experience:
            t = Table(
                [[Paragraph(ex.title + (", " + ex.company if ex.company else ""), S("et","InterBold",10.5,DARK,14)),
                  Paragraph(ex.years, S("ey","InterBold",10.5,DARK,14))]],
                colWidths=[CONTENT_W*0.72, CONTENT_W*0.28]
            )
            t.setStyle(TableStyle([
                ("VALIGN",(0,0),(-1,-1),"TOP"),("ALIGN",(1,0),(1,0),"RIGHT"),
                ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
            ]))
            story.append(t)
            sp(3)
            for line in ex.description:
                story.append(Paragraph(line, S("ed","Inter",10,DARK,14,leftIndent=8), bulletText="•"))
            sp(8)

    if cv.education:
        section("FORMATION")
        for ed in cv.education:
            t = Table(
                [[Paragraph(ed.title, S("edt","InterBold",10.5,DARK,14)),
                  Paragraph(ed.years, S("edy","InterBold",10.5,DARK,14))]],
                colWidths=[CONTENT_W*0.72, CONTENT_W*0.28]
            )
            t.setStyle(TableStyle([
                ("VALIGN",(0,0),(-1,-1),"TOP"),("ALIGN",(1,0),(1,0),"RIGHT"),
                ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
            ]))
            story.append(t)
            if ed.school:
                story.append(Paragraph(ed.school, S("esc","Inter",10,GRAY,13)))
            sp(8)

    if cv.competences or cv.languages:
        section("COMPETENCES & LANGUES")
        col_w = (CONTENT_W - 12) / 2
        comp_fl = ListFlowable(
            [ListItem(Paragraph(c, S("ci_"+str(i),"Inter",10,DARK,14))) for i,c in enumerate(cv.competences)],
            bulletType="bullet", leftIndent=10, bulletFontSize=7
        )
        lang_fl = ListFlowable(
            [ListItem(Paragraph("<b>"+l.name+"</b>: "+l.level, S("li_"+str(i),"Inter",10,DARK,14))) for i,l in enumerate(cv.languages)],
            bulletType="bullet", leftIndent=10, bulletFontSize=7
        )
        row = Table([[comp_fl, lang_fl]], colWidths=[col_w, col_w])
        row.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ]))
        story.append(row)

    buf = io.BytesIO()
    _render_simple(story, buf, PAGE_W, PAGE_H, MARGIN)
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------- #
#  Pink Minimalist template                                                    #
# --------------------------------------------------------------------------- #

def _build_pink_cv(cv):
    import os
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import Paragraph, Spacer, Frame
    from reportlab.pdfgen import canvas as rl_canvas

    FONT_DIR = _font_dir()
    _reg("Raleway",     os.path.join(FONT_DIR, "Raleway-Regular.ttf"), pdfmetrics)
    _reg("RalewayBold", os.path.join(FONT_DIR, "Raleway-Bold.ttf"),    pdfmetrics)

    PAGE_W, PAGE_H = A4
    PINK_BG  = HexColor("#fce4ec")
    PINK_ACC = HexColor("#c2185b")
    DARK     = HexColor("#424242")
    GRAY     = HexColor("#757575")

    SIDEBAR_W = 6.0 * cm
    MARGIN    = 0.75 * cm
    MAIN_X    = SIDEBAR_W + 0.5 * cm
    MAIN_W    = PAGE_W - MAIN_X - MARGIN

    from reportlab.lib.styles import ParagraphStyle as PS
    def S(n, f="Raleway", sz=10, col=DARK, lead=14, **kw):
        return PS(n, fontName=f, fontSize=sz, textColor=col, leading=lead, **kw)

    # -- Sidebar --
    sb = []
    sp_sb = lambda h=5: sb.append(Spacer(1, h))
    def sb_sec(t):
        sp_sb(8)
        sb.append(Paragraph(t, S("ss","RalewayBold",9,PINK_ACC,14)))
        sb.append(_HRule(SIDEBAR_W - MARGIN*2, PINK_ACC, 0.5))
        sp_sb(3)

    sb.append(Paragraph(cv.name or "NOM PRENOM", S("sn","RalewayBold",13,PINK_ACC,17)))
    sp_sb(6)

    sb_sec("CONTACT")
    for val, label in [(cv.contact.phone,"Tel"),(cv.contact.email,"Email"),(cv.contact.location,"Adresse")]:
        if val:
            sb.append(Paragraph("<b>"+label+":</b> "+val, S("sc","Raleway",8,DARK,12)))
            sp_sb(2)

    if cv.competences:
        sb_sec("COMPETENCES")
        for c in cv.competences:
            sb.append(Paragraph("- " + c, S("sco_"+c[:6],"Raleway",9,DARK,13)))
            sp_sb(1)

    if cv.languages:
        sb_sec("LANGUES")
        for l in cv.languages:
            sb.append(Paragraph("<b>"+l.name+"</b>: "+l.level, S("sl_"+l.name[:4],"Raleway",9,DARK,13)))
            sp_sb(2)

    # -- Main --
    mn = []
    sp_mn = lambda h=6: mn.append(Spacer(1, h))
    def mn_sec(t):
        sp_mn(10)
        mn.append(Paragraph(t, S("ms","RalewayBold",11,PINK_ACC,16)))
        mn.append(_HRule(MAIN_W, PINK_ACC, 0.5))
        sp_mn(4)

    if cv.bio:
        mn_sec("PROFIL")
        mn.append(Paragraph(cv.bio, S("mb","Raleway",9,DARK,14)))

    if cv.experience:
        mn_sec("EXPERIENCE")
        for ex in cv.experience:
            mn.append(Paragraph(ex.title, S("et_"+ex.title[:6],"RalewayBold",10,DARK,14)))
            sub = "  -  ".join([p for p in [ex.company, ex.years] if p])
            if sub:
                mn.append(Paragraph(sub, S("ey_"+ex.title[:4],"Raleway",9,GRAY,12)))
            sp_mn(2)
            for line in ex.description:
                mn.append(Paragraph("- " + line, S("ed_"+line[:6],"Raleway",9,DARK,13)))
            sp_mn(8)

    if cv.education:
        mn_sec("FORMATION")
        for ed in cv.education:
            mn.append(Paragraph(ed.title, S("edt_"+ed.title[:6],"RalewayBold",10,DARK,14)))
            sub = "  -  ".join([p for p in [ed.school, ed.years] if p])
            if sub:
                mn.append(Paragraph(sub, S("edy_"+ed.title[:4],"Raleway",9,GRAY,12)))
            sp_mn(8)

    # -- Draw --
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(PAGE_W, PAGE_H))
    c.setFillColor(PINK_BG)
    c.rect(0, 0, SIDEBAR_W, PAGE_H, fill=1, stroke=0)

    from reportlab.platypus import Frame as RLFrame
    sb_frame = RLFrame(MARGIN, MARGIN, SIDEBAR_W-MARGIN*2, PAGE_H-MARGIN*2,
                       showBoundary=0, leftPadding=0, rightPadding=0, topPadding=4, bottomPadding=0)
    sb_frame.addFromList(sb, c)

    mn_frame = RLFrame(MAIN_X, MARGIN, MAIN_W, PAGE_H-MARGIN*2,
                       showBoundary=0, leftPadding=0, rightPadding=0, topPadding=4, bottomPadding=0)
    mn_frame.addFromList(mn, c)

    c.save()
    buf.seek(0)
    return buf.read()


# --------------------------------------------------------------------------- #
#  Shared helpers                                                              #
# --------------------------------------------------------------------------- #

def _font_dir():
    import os
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "fonts"))

def _reg(name, path, pdfmetrics):
    import os
    from reportlab.pdfbase.ttfonts import TTFont
    if not os.path.exists(path):
        raise FileNotFoundError("Font not found: " + path + "\nPlace .ttf files in backend/fonts/")
    try:
        pdfmetrics.registerFont(TTFont(name, path))
    except Exception:
        pass  # already registered is fine


def _render_simple(story, buf, page_w, page_h, margin):
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import Frame
    c = rl_canvas.Canvas(buf, pagesize=(page_w, page_h))
    frame = Frame(margin, margin, page_w-2*margin, page_h-2*margin,
                  showBoundary=0, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    frame.addFromList(list(story), c)
    c.save()


class _HRule(object):
    from reportlab.platypus import Flowable as _F
    pass

from reportlab.platypus import Flowable as _BaseFlowable
class _HRule(_BaseFlowable):
    def __init__(self, width, color, line_width=0.8):
        _BaseFlowable.__init__(self)
        self._width = width
        self.color = color
        self.line_width = line_width
        self.height = line_width + 4

    def wrap(self, availW, availH):
        return (self._width, self.height)

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.line_width)
        self.canv.line(0, self.height/2, self._width, self.height/2)
