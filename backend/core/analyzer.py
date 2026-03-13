"""Analyze source image and target video for settings suggestion."""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from PIL import Image


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Get video width, height, fps, duration, frame count via ffprobe."""
    meta = {"width": 0, "height": 0, "fps": 30.0, "duration": 0, "frame_count": 0}

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return meta

    try:
        import json
        data = json.loads(result.stdout)
        if "streams" and data["streams"]:
            s = data["streams"][0]
            meta["width"] = int(s.get("width", 0))
            meta["height"] = int(s.get("height", 0))
            r = s.get("r_frame_rate", "30/1")
            if "/" in r:
                a, b = map(int, r.split("/"))
                meta["fps"] = a / b if b else 30
            else:
                meta["fps"] = float(r)
        if "format" in data:
            d = data["format"].get("duration", 0)
            meta["duration"] = float(d) if d else 0
        if meta["fps"] > 0 and meta["duration"] > 0:
            meta["frame_count"] = int(meta["fps"] * meta["duration"])
    except Exception:
        pass

    return meta


def get_image_metadata(image_path: str) -> Dict[str, Any]:
    """Get image width and height."""
    meta = {"width": 0, "height": 0}
    try:
        with Image.open(image_path) as img:
            meta["width"], meta["height"] = img.size
    except Exception:
        pass
    return meta


def suggest_settings(
    source_path: str,
    target_path: str,
) -> Dict[str, Any]:
    """
    Suggest best settings based on source image and target video.

    Returns dict with swap_model, det_size, upscale, enhance, interpolate.
    """
    img_meta = get_image_metadata(source_path)
    vid_meta = get_video_metadata(target_path)

    width = vid_meta.get("width", 0)
    height = vid_meta.get("height", 0)
    fps = vid_meta.get("fps", 30)
    frame_count = vid_meta.get("frame_count", 0)
    img_w = img_meta.get("width", 0)
    img_h = img_meta.get("height", 0)

    # Resolution tier
    max_dim = max(width, height) if width and height else 0
    is_hd = max_dim >= 1080
    is_4k = max_dim >= 2160

    # Image quality
    img_min = min(img_w, img_h) if img_w and img_h else 0
    source_is_low_res = img_min > 0 and img_min < 512

    # Suggest swap model: SimSwap for HD+
    swap_model = "simswap" if is_hd else "inswapper"

    # Suggest det_size: 640 for HD+
    det_size = 640 if is_hd else 320

    # Suggest upscale: 2x for 720p or low-res source, 4x for 480p
    if max_dim <= 480:
        upscale = 4
    elif max_dim <= 720 or source_is_low_res:
        upscale = 2
    else:
        upscale = 1

    # Enhance: recommend for most cases
    enhance = True

    # Interpolate: 2x for video under 30fps or short clips, helps smoothness
    interpolate = 1
    if fps < 30 or (frame_count > 0 and frame_count < 300):
        interpolate = 2

    return {
        "swap_model": swap_model,
        "det_size": det_size,
        "upscale": upscale,
        "enhance": enhance,
        "interpolate": interpolate,
        "meta": {
            "video_width": width,
            "video_height": height,
            "video_fps": round(fps, 1),
            "video_frames": frame_count,
            "image_width": img_w,
            "image_height": img_h,
        },
    }
