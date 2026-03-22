"""FastAPI application for UltraFaceswap."""

import asyncio
import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(root / ".env")
except ImportError:
    pass

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router

logger = logging.getLogger(__name__)

_tg_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Telegram bot on app startup, shut down on exit."""
    global _tg_app
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "")

    if bot_token:
        try:
            from backend.bot.telegram_bot import create_bot_app
            _tg_app = create_bot_app()
            if _tg_app:
                await _tg_app.initialize()
                await _tg_app.start()

                if webhook_url:
                    wh = webhook_url.rstrip("/") + "/api/telegram/webhook"
                    await _tg_app.bot.set_webhook(url=wh)
                    logger.info("Telegram webhook set: %s", wh)
                else:
                    await _tg_app.updater.start_polling(drop_pending_updates=True)
                    logger.info("Telegram bot started in polling mode.")
        except Exception:
            logger.exception("Failed to start Telegram bot")
            _tg_app = None

    yield

    if _tg_app:
        try:
            if _tg_app.updater and _tg_app.updater.running:
                await _tg_app.updater.stop()
            await _tg_app.stop()
            await _tg_app.shutdown()
        except Exception:
            logger.exception("Error stopping Telegram bot")


app = FastAPI(
    title="UltraFaceswap API",
    description="Face swap from photo onto video",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the React frontend if the build folder exists
_frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")


@app.get("/")
async def root_page():
    return {
        "name": "UltraFaceswap API",
        "docs": "/docs",
        "api": "/api",
        "telegram": "configured" if os.environ.get("TELEGRAM_BOT_TOKEN") else "not configured",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request):
    """Receive Telegram updates via webhook (used on HF Spaces)."""
    if not _tg_app:
        return Response(status_code=200, content="Bot not configured")
    from telegram import Update
    data = await request.json()
    update = Update.de_json(data, _tg_app.bot)
    await _tg_app.process_update(update)
    return Response(status_code=200)
