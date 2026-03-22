"""Telegram bot for UltraFaceswap.

Users send a video URL (Instagram, Pinterest, TikTok, …) and optionally
a face photo.  The bot downloads the video, runs the face-swap pipeline,
and sends the result back.

Flow:
  1. User sends a photo → stored as their custom face (per chat).
  2. User sends a URL  → download video → swap with stored/default face → reply with video.
  3. /start or /help   → instructions.
  4. /reset             → clear stored face, go back to default.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DEFAULT_FACE = os.environ.get(
    "DEFAULT_FACE_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "assets" / "default_face.png"),
)
PRESET = os.environ.get("TELEGRAM_PRESET", "best")
MAX_VIDEO_DURATION = int(os.environ.get("TELEGRAM_MAX_DURATION", "300"))

_custom_faces: Dict[int, str] = {}


def _get_face_for_chat(chat_id: int) -> str:
    """Return the custom face path for this chat, or the default."""
    custom = _custom_faces.get(chat_id)
    if custom and os.path.isfile(custom):
        return custom
    return os.path.abspath(DEFAULT_FACE)


HELP_TEXT = (
    "Welcome to *UltraFaceswap Bot*\\! 🎭\n\n"
    "Send me a video link \\(Instagram, Pinterest, TikTok, YouTube\\) "
    "and I'll swap the face in it\\.\n\n"
    "*How to use:*\n"
    "1\\. \\(Optional\\) Send a *photo* of the face you want \\— I'll remember it\\.\n"
    "2\\. Send a *video URL* \\— I'll download, swap, and send it back\\.\n"
    "3\\. Send /reset to go back to the default face\\.\n\n"
    "That's it\\! Processing takes 1–5 minutes depending on video length\\."
)


async def _run_swap_pipeline(
    source_path: str,
    target_video_path: str,
    preset: str = "best",
) -> str:
    """Run the face swap pipeline synchronously in a thread, return output path."""
    from backend.core.facefusion_runner import (
        run_facefusion_two_pass,
        run_facefusion,
        is_facefusion_available,
    )
    from backend.core.frame_validator import validate_and_repair

    if not is_facefusion_available():
        raise RuntimeError("FaceFusion is not configured on this server.")

    from backend.api.routes import PRESETS
    cfg = PRESETS.get(preset, PRESETS["best"])

    fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="uf_tg_out_")
    os.close(fd)

    enhancer_blend = cfg["face_enhancer_blend"] if cfg["face_enhancer"] else 0.0
    is_two_pass = cfg["two_pass"] and enhancer_blend > 0

    kwargs = dict(
        face_swapper_model=cfg["face_swapper_model"],
        face_swapper_pixel_boost=cfg["pixel_boost"],
        face_enhancer_blend=enhancer_blend,
        face_detector_model=cfg["face_detector_model"],
        face_detector_score=cfg.get("face_detector_score", 0.35),
        face_selector_mode=cfg.get("face_selector_mode", "reference"),
        face_mask_blur=cfg["face_mask_blur"],
    )

    loop = asyncio.get_event_loop()

    if is_two_pass:
        await loop.run_in_executor(
            None,
            lambda: run_facefusion_two_pass(
                source_path, target_video_path, output_path, **kwargs,
            ),
        )
    else:
        await loop.run_in_executor(
            None,
            lambda: run_facefusion(
                source_path, target_video_path, output_path, **kwargs,
            ),
        )

    try:
        repair = await loop.run_in_executor(
            None,
            lambda: validate_and_repair(source_path, output_path, temporal_smooth=True),
        )
    except Exception as exc:
        logger.warning("Frame validation skipped: %s", exc)
        repair = {}

    return output_path, repair


def create_bot_app():
    """Build and return a python-telegram-bot Application (not started)."""
    from telegram import Update, BotCommand
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        filters,
        ContextTypes,
    )

    if not BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled.")
        return None

    app = Application.builder().token(BOT_TOKEN).build()

    async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(HELP_TEXT, parse_mode="MarkdownV2")

    async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        old = _custom_faces.pop(chat_id, None)
        if old and os.path.isfile(old):
            os.unlink(old)
        await update.message.reply_text("Face reset to default. Send a new photo to change it.")

    async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        photo = update.message.photo[-1]
        file = await ctx.bot.get_file(photo.file_id)

        ext = ".jpg"
        fd, path = tempfile.mkstemp(suffix=ext, prefix=f"uf_face_{chat_id}_")
        os.close(fd)
        await file.download_to_drive(path)

        old = _custom_faces.pop(chat_id, None)
        if old and os.path.isfile(old):
            os.unlink(old)
        _custom_faces[chat_id] = path

        await update.message.reply_text(
            "Got it! I'll use this face for your next swaps. Now send me a video URL."
        )

    async def handle_url(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        from backend.core.downloader import download_video, is_supported_url

        chat_id = update.effective_chat.id
        text = (update.message.text or "").strip()

        if not is_supported_url(text):
            await update.message.reply_text(
                "I don't recognise that URL. Send an Instagram, Pinterest, TikTok, or YouTube link."
            )
            return

        face_path = _get_face_for_chat(chat_id)
        is_default = face_path == os.path.abspath(DEFAULT_FACE)

        status_msg = await update.message.reply_text(
            "⏳ Downloading video…"
        )

        target_path = None
        output_path = None
        try:
            loop = asyncio.get_event_loop()
            target_path = await loop.run_in_executor(
                None,
                lambda: download_video(text, max_duration=MAX_VIDEO_DURATION),
            )

            await status_msg.edit_text("🔄 Swapping face… This may take 1–5 minutes.")

            output_path, repair = await _run_swap_pipeline(face_path, target_path, PRESET)

            file_size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
            if file_size < 1024:
                await status_msg.edit_text("❌ Processing failed — output video is empty.")
                return

            caption_parts = ["✅ Done!"]
            if repair and repair.get("total_frames"):
                r = repair
                caption_parts.append(
                    f"{r['good_frames']}/{r['total_frames']} frames OK"
                )
                if r.get("repaired_frames", 0) > 0:
                    caption_parts.append(f"{r['repaired_frames']} repaired")
            caption = " · ".join(caption_parts)
            face_label = "default face" if is_default else "your uploaded face"
            caption += f"\n\nUsing {face_label} · Preset: {PRESET}"

            if file_size > 50 * 1024 * 1024:
                await status_msg.edit_text(
                    f"⚠️ Output video is {file_size // (1024*1024)}MB — too large for Telegram "
                    "(50MB limit). Try a shorter video."
                )
                return

            await status_msg.edit_text("📤 Uploading result…")
            with open(output_path, "rb") as vf:
                await update.message.reply_video(
                    video=vf,
                    caption=caption,
                    supports_streaming=True,
                    read_timeout=120,
                    write_timeout=120,
                )
            await status_msg.delete()

        except Exception as exc:
            logger.exception("Telegram swap failed")
            await status_msg.edit_text(f"❌ Error: {exc}")
        finally:
            if target_path and os.path.isfile(target_path):
                os.unlink(target_path)
            if output_path and os.path.isfile(output_path):
                os.unlink(output_path)

    async def handle_other(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Send me a video URL (Instagram, Pinterest, TikTok) or a face photo.\n"
            "Type /help for instructions."
        )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"https?://"), handle_url))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_other))

    return app
