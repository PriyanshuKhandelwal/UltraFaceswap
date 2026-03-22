"""Download videos from Instagram, Pinterest, TikTok, and other platforms via yt-dlp."""

import os
import logging
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_SITES = [
    "instagram.com",
    "pinterest.com",
    "pin.it",
    "tiktok.com",
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
]


def is_supported_url(url: str) -> bool:
    """Check if the URL looks like a supported video platform."""
    url_lower = url.lower().strip()
    return any(site in url_lower for site in SUPPORTED_SITES)


def download_video(
    url: str,
    output_path: Optional[str] = None,
    max_duration: int = 300,
    max_filesize: str = "200M",
) -> str:
    """Download a video from a social media URL.

    Args:
        url: Instagram Reel, Pinterest pin, TikTok, YouTube, etc.
        output_path: Where to save. Auto-generates a temp path if None.
        max_duration: Reject videos longer than this (seconds).
        max_filesize: Max download size (yt-dlp format string).

    Returns:
        Path to the downloaded mp4 file.

    Raises:
        ValueError: URL not supported or video not found.
        RuntimeError: Download failed.
    """
    import yt_dlp

    if not output_path:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="uf_dl_")
        os.close(fd)

    opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "max_filesize": max_filesize,
        "socket_timeout": 30,
        "retries": 3,
        "merge_output_format": "mp4",
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

    if max_duration > 0:
        opts["match_filter"] = yt_dlp.utils.match_filter_func(
            f"duration <= {max_duration}"
        )

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise ValueError(f"Could not extract video info from: {url}")

            actual_path = ydl.prepare_filename(info)
            if actual_path != output_path and os.path.isfile(actual_path):
                os.replace(actual_path, output_path)

            if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1024:
                raise RuntimeError("Download produced an empty or missing file.")

            duration = info.get("duration", 0)
            title = info.get("title", "Unknown")
            logger.info(
                "Downloaded '%s' (%.1fs) from %s → %s",
                title, duration or 0, info.get("extractor", "?"), output_path,
            )
            return output_path

    except yt_dlp.utils.DownloadError as exc:
        if os.path.isfile(output_path):
            os.unlink(output_path)
        raise RuntimeError(f"Download failed: {exc}") from exc
    except Exception:
        if os.path.isfile(output_path):
            os.unlink(output_path)
        raise
