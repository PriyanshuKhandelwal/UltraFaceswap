"""FaceFusion subprocess runner - optional engine for video face swapping."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


def _get_facefusion_path() -> Optional[str]:
    """Resolve FaceFusion install path from env."""
    path = os.environ.get("ULTRAFACESWAP_FACEFUSION_PATH") or os.environ.get("FACEFUSION_PATH")
    if not path:
        return None
    return os.path.abspath(path)


def _find_facefusion_script(root: str) -> Optional[str]:
    """Find main entry script (facefusion.py or run.py) in FaceFusion root."""
    for name in ("facefusion.py", "run.py"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    return None


def _get_facefusion_python() -> str:
    """Get Python executable for FaceFusion (needs 3.12, separate from UltraFaceswap)."""
    env_python = os.environ.get("ULTRAFACESWAP_FACEFUSION_PYTHON")
    if env_python and os.path.isfile(env_python):
        return env_python
    # Common conda env paths for facefusion
    candidates = [
        os.path.expanduser("~/miniconda3/envs/facefusion/bin/python"),
        os.path.expanduser("~/anaconda3/envs/facefusion/bin/python"),
        "/opt/miniconda3/envs/facefusion/bin/python",
        "/opt/homebrew/Caskroom/miniconda/base/envs/facefusion/bin/python",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    try:
        base = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True, text=True, timeout=5
        )
        if base.returncode == 0 and base.stdout.strip():
            p = os.path.join(base.stdout.strip(), "envs", "facefusion", "bin", "python")
            if os.path.isfile(p):
                return p
    except Exception:
        pass
    return sys.executable  # fallback (may fail for FaceFusion)


# FaceFusion model → valid pixel boost choices (from face_swapper/choices.py)
_MODEL_PIXEL_BOOST = {
    "inswapper_128": ["128x128", "256x256", "384x384", "512x512", "768x768", "1024x1024"],
    "inswapper_128_fp16": ["128x128", "256x256", "384x384", "512x512", "768x768", "1024x1024"],
    "simswap_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "simswap_unofficial_512": ["512x512", "768x768", "1024x1024"],
    "hyperswap_1a_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "hyperswap_1b_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "hyperswap_1c_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "blendswap_256": ["256x256", "384x384", "512x512", "768x768", "1024x1024"],
    "ghost_1_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "ghost_2_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "ghost_3_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "hififace_unofficial_256": ["256x256", "512x512", "768x768", "1024x1024"],
    "uniface_256": ["256x256", "512x512", "768x768", "1024x1024"],
}


def _valid_pixel_boost(model: str, user_choice: str) -> str:
    """Return a valid pixel boost for the model; fallback to 256x256 for unknown models."""
    choices = _MODEL_PIXEL_BOOST.get(model, ["256x256", "512x512", "768x768", "1024x1024"])
    px = str(user_choice or "128").strip().lower()
    if "x" in px:
        if px in choices:
            return px
    else:
        target = f"{px}x{px}" if px.isdigit() else "256x256"
        if target in choices:
            return target
    return choices[0]


def is_facefusion_available() -> bool:
    """Check if FaceFusion is installed and usable."""
    root = _get_facefusion_path()
    if not root or not os.path.isdir(root):
        return False
    script = _find_facefusion_script(root)
    return script is not None


def run_facefusion(
    source_path: str,
    target_path: str,
    output_path: str,
    *,
    face_swapper_model: str = "inswapper_128_fp16",
    face_swapper_pixel_boost: str = "128",
    face_enhancer_blend: float = 0.5,
    lip_sync: bool = False,
    source_audio_path: Optional[str] = None,
    timeout: Optional[int] = 3600,
) -> None:
    """
    Run FaceFusion headless for face swap.

    Args:
        source_path: Path to source face image
        target_path: Path to target video
        output_path: Path for output video
        face_swapper_model: FaceFusion model (inswapper_128_fp16, simswap_256, etc.)
        face_swapper_pixel_boost: Pixel boost (128, 256, 512)
        face_enhancer_blend: Face enhancer blend 0-1
        lip_sync: Enable lip sync
        source_audio_path: Audio file for lip sync; if None and lip_sync=True, audio is
            extracted from target video via ffmpeg
        timeout: Subprocess timeout in seconds (default 1 hour)

    Raises:
        RuntimeError: If FaceFusion not available or process fails
    """
    root = _get_facefusion_path()
    if not root:
        raise RuntimeError(
            "FaceFusion not configured. Set ULTRAFACESWAP_FACEFUSION_PATH to the FaceFusion repo root."
        )
    script = _find_facefusion_script(root)
    if not script:
        raise RuntimeError(
            f"FaceFusion script not found in {root}. Expected facefusion.py or run.py."
        )

    source_path = os.path.abspath(source_path)
    target_path = os.path.abspath(target_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    processors = ["face_swapper"]
    if face_enhancer_blend > 0:
        processors.append("face_enhancer")
    if lip_sync:
        processors.append("lip_syncer")

    # Lip sync needs source audio: pass image + audio via -s
    source_paths: list[str] = [source_path]
    extracted_audio: Optional[str] = None
    if lip_sync:
        audio_path = source_audio_path
        if not audio_path or not os.path.isfile(audio_path):
            # Extract audio from target video (.wav; FaceFusion supports flac,m4a,mp3,ogg,opus,wav)
            try:
                fd, extracted_audio = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                r = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", target_path,
                        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                        extracted_audio,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if r.returncode == 0 and os.path.isfile(extracted_audio) and os.path.getsize(extracted_audio) > 0:
                    audio_path = extracted_audio
                else:
                    err = (r.stderr or r.stdout or "")[:400]
                    raise RuntimeError(
                        f"Lip sync needs audio. Target video may have no audio track, or ffmpeg failed. "
                        f"Ensure ffmpeg is installed and the video has audio. ffmpeg: {err}"
                    )
            except FileNotFoundError:
                raise RuntimeError(
                    "Lip sync requires ffmpeg to extract audio from the target video. Install ffmpeg and try again."
                )
        if audio_path:
            source_paths.append(os.path.abspath(audio_path))

    jobs_dir = os.path.join(root, ".jobs")
    os.makedirs(jobs_dir, exist_ok=True)

    python_exe = _get_facefusion_python()
    cmd = [
        python_exe,
        script,
        "headless-run",
        "--jobs-path", jobs_dir,
        "--processors", *processors,
        "-s", *source_paths,
        "-t", target_path,
        "-o", output_path,
    ]
    model = (face_swapper_model or "inswapper_128_fp16").strip()
    pixel_boost_val = _valid_pixel_boost(model, face_swapper_pixel_boost)
    cmd.extend(["--face-swapper-model", model, "--face-swapper-pixel-boost", pixel_boost_val])
    if face_enhancer_blend > 0:
        blend_int = max(0, min(100, int(face_enhancer_blend * 100)))
        cmd.extend(["--face-enhancer-blend", str(blend_int)])

    env = os.environ.copy()
    env["ULTRAFACESWAP_FACEFUSION_PATH"] = root
    try:
        result = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        if extracted_audio and os.path.isfile(extracted_audio):
            try:
                os.unlink(extracted_audio)
            except OSError:
                pass

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "Unknown error").strip()
        cmd_str = " ".join(repr(a) for a in cmd)
        raise RuntimeError(
            f"FaceFusion failed (exit {result.returncode}): {err[:600]}\n\nCommand: {cmd_str}"
        )
