"""FastAPI application for UltraFaceswap."""

from pathlib import Path

try:
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(root / ".env")
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router

app = FastAPI(
    title="UltraFaceswap API",
    description="Face swap from photo onto video",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "UltraFaceswap API",
        "docs": "/docs",
        "api": "/api",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
