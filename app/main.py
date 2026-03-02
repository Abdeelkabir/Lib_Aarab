"""
app/main.py
FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.scanner  import router as scanner_router
from app.api.images   import router as images_router
from app.api.research import router as research_router
from app.api.cv       import router as cv_router
from app.core.sam import get_predictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: pre-load SAM at startup ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading MobileSAM...")
    try:
        get_predictor()
        logger.info("MobileSAM ready.")
    except Exception as e:
        logger.error(f"Failed to load MobileSAM: {e}")
        # Don't crash — first request will retry and surface the error clearly
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocScanner API",
    description="Document scanning: edge detection (SAM) → warp → enhance → PDF",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", *([os.environ["FRONTEND_URL"]] if "FRONTEND_URL" in os.environ else [])],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scanner_router)
app.include_router(images_router)
app.include_router(research_router)
app.include_router(cv_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
