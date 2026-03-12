"""Extract frames from video and preserve audio using FFmpeg."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


def extract_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    image_format: str = "png",
) -> Tuple[str, str]:
    """
    Extract all frames from a video file.

    Args:
        video_path: Path to input video file
        output_dir: Directory for extracted frames (default: temp dir)
        image_format: Output format (png or jpg)

    Returns:
        Tuple of (frames_directory, audio_path)
        audio_path is path to extracted audio, or empty string if no audio
    """
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="ultrafaceswap_frames_")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Frame output pattern: frame_000001.png
    frame_pattern = os.path.join(output_dir, f"frame_%06d.{image_format}")

    # Extract frames
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-q:v", "2",  # High quality for png
        frame_pattern,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # FFmpeg may return non-zero when video has no frames, or other issues
        if "does not contain any stream" in result.stderr:
            raise ValueError(f"Video has no video stream: {video_path}")
        raise RuntimeError(
            f"FFmpeg frame extraction failed: {result.stderr}"
        )

    # Extract audio (optional - video may not have audio)
    audio_path = os.path.join(output_dir, "audio.aac")
    audio_cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "copy",
        audio_path,
    ]
    audio_result = subprocess.run(
        audio_cmd,
        capture_output=True,
        text=True,
    )

    if audio_result.returncode != 0 or not os.path.exists(audio_path):
        audio_path = ""

    return output_dir, audio_path


def get_frame_count(frames_dir: str) -> int:
    """Count extracted frames in directory."""
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        return 0
    # Support both png and jpg
    count = len(list(frames_dir.glob("frame_*.png")))
    if count == 0:
        count = len(list(frames_dir.glob("frame_*.jpg")))
    return count
