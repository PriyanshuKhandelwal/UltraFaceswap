"""Merge processed frames back into video with audio."""

import os
import subprocess
from pathlib import Path


def merge_frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: float = 30.0,
    audio_path: str = "",
    input_format: str = "png",
) -> str:
    """
    Merge frames into video and optionally add audio.

    Args:
        frames_dir: Directory containing frame_000001.png, frame_000002.png, ...
        output_path: Output video file path
        fps: Frames per second
        audio_path: Path to audio file (optional)
        input_format: Frame format (png or jpg)

    Returns:
        Path to created video file
    """
    frames_dir = os.path.abspath(frames_dir)
    output_path = os.path.abspath(output_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    frame_pattern = os.path.join(frames_dir, f"frame_%06d.{input_format}")

    if audio_path and os.path.exists(audio_path):
        # Two-pass: create video from frames, then mux with audio
        temp_video = output_path + ".temp.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            temp_video,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg video creation failed: {result.stderr}")

        # Mux audio
        mux_cmd = [
            "ffmpeg",
            "-y",
            "-i", temp_video,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        mux_result = subprocess.run(mux_cmd, capture_output=True, text=True)
        os.remove(temp_video)
        if mux_result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio mux failed: {mux_result.stderr}")
    else:
        # Video only
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg video creation failed: {result.stderr}")

    return output_path


def get_video_fps(video_path: str) -> float:
    """Extract FPS from video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 30.0  # Default fallback

    r = result.stdout.strip()
    if "/" in r:
        num, den = map(int, r.split("/"))
        return num / den if den else 30.0
    return float(r) if r else 30.0
