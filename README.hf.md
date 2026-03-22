---
title: UltraFaceswap
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
hardware: t4-small
pinned: false
---

# UltraFaceswap

Face swap from photo onto video — powered by FaceFusion + InsightFace.

## Features

- **Preset-based swapping**: Quick / Best / Max quality presets
- **Frame validation & repair**: Detects and fixes flickering frames with temporal smoothing
- **URL-based input**: Paste Instagram, Pinterest, TikTok, or YouTube links
- **Telegram bot**: Send a link, get a face-swapped video back
- **Two-pass processing**: Separate swap and enhance passes for lower memory usage

## Telegram Bot

Set the `TELEGRAM_BOT_TOKEN` secret in your Space settings to enable the bot.
The webhook is automatically configured at `https://YOUR-SPACE.hf.space/api/telegram/webhook`.

## API

- `POST /api/swap-preset` — Preset-based face swap (upload files)
- `POST /api/swap-from-url` — URL-based face swap (paste link)
- `POST /api/swap-pro` — Advanced settings
- `GET /api/status/{job_id}` — Poll job progress
- `GET /api/result/{job_id}` — Download result
