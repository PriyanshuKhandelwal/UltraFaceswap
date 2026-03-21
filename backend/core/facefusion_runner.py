"""FaceFusion subprocess runner - optional engine for video face swapping."""

import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional


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
    return sys.executable


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


def _is_progress_bar_only(text: str) -> bool:
    """True if text looks like only progress bar (no traceback)."""
    if not text or len(text) < 100:
        return False
    if any(m in text for m in ("Traceback", "Error:", "Exception:", "File \"", "RuntimeError")):
        return False
    return text.count("frame/s") >= 3 or text.count("analysing:") >= 3


def _extract_error_snippet(err: str, tail_size: int = 5000) -> str:
    """Extract the part of FaceFusion output that contains the actual error (traceback)."""
    if not err:
        return "Unknown error"
    for marker in ("Traceback (most recent call last):", "Traceback ", "Error:", "Exception:", "RuntimeError:", "CUDA out of memory", "File \""):
        idx = err.find(marker)
        if idx >= 0:
            snippet = err[idx:].strip()
            return snippet[-tail_size:] if len(snippet) > tail_size else snippet
    progress_pattern = re.compile(r"(\d+)/(\d+).*(?:frame/s|%\||analysing:|processing:)")
    lines = err.splitlines()
    last_progress_i = -1
    for i in range(len(lines) - 1, -1, -1):
        if progress_pattern.search(lines[i]):
            last_progress_i = i
            break
    after_progress = "\n".join(lines[last_progress_i + 1 :]).strip() if last_progress_i >= 0 else ""
    if len(after_progress) >= 80:
        snippet = after_progress[-tail_size:] if len(after_progress) > tail_size else after_progress
        if not _is_progress_bar_only(snippet):
            return snippet
    fallback = err[-tail_size:] if len(err) > tail_size else err
    if _is_progress_bar_only(fallback):
        return (
            "FaceFusion failed after the analysing phase (no traceback captured). "
            "Common causes: GPU out of memory during processing, or disk full. "
            "Try: turn off face enhancer, use a shorter video or lower pixel boost (256), or run the command in a terminal to see the full error."
        )
    return fallback


def _is_oom_error(returncode: int, error_text: str) -> bool:
    """Detect if an error is an out-of-memory (OOM) condition."""
    if returncode == -9:
        return True
    oom_markers = ("CUDA out of memory", "out of memory", "oom", "Cannot allocate memory", "Killed")
    lower = error_text.lower()
    return any(m.lower() in lower for m in oom_markers)


def check_output_has_swapped_faces(
    target_path: str,
    output_path: str,
    tolerance_ratio: float = 0.02,
) -> bool:
    """
    Heuristic: compare output file size to input. If they're nearly identical,
    FaceFusion likely didn't swap any faces (wrote original frames back).
    Returns True if swap appears to have occurred.
    """
    try:
        target_size = os.path.getsize(target_path)
        output_size = os.path.getsize(output_path)
        if target_size == 0 or output_size == 0:
            return False
        ratio = abs(output_size - target_size) / target_size
        return ratio > tolerance_ratio
    except OSError:
        return True


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
    source_paths: Optional[List[str]] = None,
    timeout: Optional[int] = 3600,
    face_detector_model: Optional[str] = None,
    face_detector_size: Optional[str] = None,
    face_detector_score: Optional[float] = None,
    face_selector_mode: Optional[str] = None,
    face_mask_blur: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    processors: Optional[List[str]] = None,
) -> None:
    """
    Run FaceFusion headless for face swap.

    When ``processors`` is given it overrides the auto-built list (useful for
    two-pass enhance where the second call only runs ``face_enhancer``).
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

    target_path = os.path.abspath(target_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if source_paths:
        image_paths = [os.path.abspath(p) for p in source_paths if p and os.path.isfile(p)]
        if not image_paths:
            raise RuntimeError("No valid source image paths in source_paths.")
    else:
        source_path = os.path.abspath(source_path)
        image_paths = [source_path]

    if processors is None:
        processors = ["face_swapper"]
        if face_enhancer_blend > 0:
            processors.append("face_enhancer")
        if lip_sync:
            processors.append("lip_syncer")

    source_paths_final: list[str] = list(image_paths)
    extracted_audio: Optional[str] = None
    if lip_sync:
        audio_path = source_audio_path
        if not audio_path or not os.path.isfile(audio_path):
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
            source_paths_final.append(os.path.abspath(audio_path))

    jobs_dir = os.path.join(root, ".jobs")
    os.makedirs(jobs_dir, exist_ok=True)

    python_exe = _get_facefusion_python()
    cmd = [
        python_exe,
        script,
        "headless-run",
        "--jobs-path", jobs_dir,
        "--processors", *processors,
        "-s", *source_paths_final,
        "-t", target_path,
        "-o", output_path,
    ]
    model = (face_swapper_model or "inswapper_128_fp16").strip()
    if "face_swapper" in processors:
        pixel_boost_val = _valid_pixel_boost(model, face_swapper_pixel_boost)
        cmd.extend(["--face-swapper-model", model, "--face-swapper-pixel-boost", pixel_boost_val])
    if "face_enhancer" in processors and face_enhancer_blend > 0:
        blend_int = max(0, min(100, int(face_enhancer_blend * 100)))
        cmd.extend(["--face-enhancer-blend", str(blend_int)])
    if face_detector_model and face_detector_model.strip():
        cmd.extend(["--face-detector-model", face_detector_model.strip()])
    if face_detector_size and face_detector_size.strip():
        cmd.extend(["--face-detector-size", face_detector_size.strip()])
    if face_mask_blur is not None and 0 <= face_mask_blur <= 1:
        cmd.extend(["--face-mask-blur", str(round(face_mask_blur, 2))])
    if face_detector_score is not None and 0 < face_detector_score <= 1:
        cmd.extend(["--face-detector-score", str(round(face_detector_score, 2))])
    if face_selector_mode and face_selector_mode.strip():
        cmd.extend(["--face-selector-mode", face_selector_mode.strip()])

    env = os.environ.copy()
    env["ULTRAFACESWAP_FACEFUSION_PATH"] = root
    env["PYTHONUNBUFFERED"] = "1"

    _progress_re = re.compile(r"(\d+)/(\d+)")

    def _parse_progress(line: str) -> Optional[tuple]:
        matches = _progress_re.findall(line)
        if not matches:
            return None
        cur, tot = matches[-1]
        return (int(cur), int(tot))

    def _read_stream_lines(stream, on_line, collected: list) -> None:
        """Read a stream char-by-char, splitting on \\r and \\n (tqdm uses \\r)."""
        buf = []
        while True:
            ch = stream.read(1)
            if not ch:
                if buf:
                    line = "".join(buf)
                    collected.append(line)
                    on_line(line)
                break
            if ch in ("\r", "\n"):
                if buf:
                    line = "".join(buf)
                    collected.append(line)
                    on_line(line)
                    buf = []
            else:
                buf.append(ch)

    if progress_callback is not None:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            stdout_lines: List[str] = []
            stderr_lines: List[str] = []
            last_reported: tuple = (0, 0)

            def _on_stderr_line(line: str) -> None:
                nonlocal last_reported
                parsed = _parse_progress(line)
                if parsed and parsed != last_reported:
                    last_reported = parsed
                    progress_callback(parsed[0], parsed[1])

            def _on_stdout_line(line: str) -> None:
                nonlocal last_reported
                parsed = _parse_progress(line)
                if parsed and parsed != last_reported:
                    last_reported = parsed
                    progress_callback(parsed[0], parsed[1])

            stderr_thread = threading.Thread(
                target=_read_stream_lines,
                args=(proc.stderr, _on_stderr_line, stderr_lines),
                daemon=True,
            )
            stdout_thread = threading.Thread(
                target=_read_stream_lines,
                args=(proc.stdout, _on_stdout_line, stdout_lines),
                daemon=True,
            )
            stderr_thread.start()
            stdout_thread.start()

            try:
                stdout_thread.join(timeout=timeout)
                stderr_thread.join(timeout=10)
            finally:
                proc.wait(timeout=timeout)
            result = subprocess.CompletedProcess(
                cmd, proc.returncode,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
            )
        finally:
            if extracted_audio and os.path.isfile(extracted_audio):
                try:
                    os.unlink(extracted_audio)
                except OSError:
                    pass
    else:
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
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        if stderr and stdout:
            err = stderr + "\n\n[stdout tail]\n" + (stdout[-1500:] if len(stdout) > 1500 else stdout)
        else:
            err = stderr or stdout or "Unknown error"
        err_snippet = _extract_error_snippet(err)
        friendly = "FaceFusion failed after the analysing phase" in err_snippet
        hint = ""
        if result.returncode == -9:
            hint = (
                " The process was killed (exit -9 = SIGKILL). This usually means the system ran out of memory (OOM). "
                "Try: lower pixel boost (e.g. 256), disable face enhancer, use a shorter video, or add more RAM/swap."
            )
        elif result.returncode == 1 and not friendly:
            hint = " The last lines below usually contain the real cause (e.g. CUDA OOM, missing file)."
        if friendly:
            detail = f"FaceFusion failed (exit {result.returncode}):\n\n{err_snippet}"
        else:
            cmd_str = " ".join(repr(a) for a in cmd)
            detail = f"FaceFusion failed (exit {result.returncode}):{hint}\n{err_snippet}\n\nCommand: {cmd_str}"
        raise RuntimeError(detail)


def run_facefusion_two_pass(
    source_path: str,
    target_path: str,
    output_path: str,
    *,
    face_swapper_model: str = "hyperswap_1a_256",
    face_swapper_pixel_boost: str = "256",
    face_enhancer_blend: float = 0.5,
    lip_sync: bool = False,
    source_audio_path: Optional[str] = None,
    source_paths: Optional[List[str]] = None,
    timeout: Optional[int] = 3600,
    face_detector_model: Optional[str] = None,
    face_detector_size: Optional[str] = None,
    face_detector_score: Optional[float] = None,
    face_selector_mode: Optional[str] = None,
    face_mask_blur: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    pass_started_callback: Optional[Callable[[int], None]] = None,
    auto_fallback: bool = True,
) -> dict:
    """
    Two-pass FaceFusion: pass 1 swaps faces, pass 2 enhances.
    Halves peak memory vs running both processors at once.

    ``pass_started_callback(pass_number)`` is called with 0 before pass 1
    and 1 before pass 2, so callers can adjust progress ranges.

    If ``auto_fallback`` is True and the enhance pass OOMs, the swap-only
    result is kept and a warning is returned instead of failing the job.

    Returns a dict with keys:
        - ``enhanced``: bool — True if enhance pass succeeded
        - ``warning``: Optional[str] — non-None if fallback was used
    """
    fd, intermediate_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    swap_target = output_path if face_enhancer_blend <= 0 else intermediate_path

    try:
        if pass_started_callback:
            pass_started_callback(0)

        run_facefusion(
            source_path,
            target_path,
            swap_target,
            face_swapper_model=face_swapper_model,
            face_swapper_pixel_boost=face_swapper_pixel_boost,
            face_enhancer_blend=0.0,
            lip_sync=lip_sync,
            source_audio_path=source_audio_path,
            source_paths=source_paths,
            timeout=timeout,
            face_detector_model=face_detector_model,
            face_detector_size=face_detector_size,
            face_detector_score=face_detector_score,
            face_selector_mode=face_selector_mode,
            face_mask_blur=face_mask_blur,
            progress_callback=progress_callback,
            processors=["face_swapper"] + (["lip_syncer"] if lip_sync else []),
        )

        if face_enhancer_blend <= 0:
            return {"enhanced": False, "warning": None}

        if pass_started_callback:
            pass_started_callback(1)

        try:
            run_facefusion(
                source_path,
                intermediate_path,
                output_path,
                face_swapper_model=face_swapper_model,
                face_swapper_pixel_boost=face_swapper_pixel_boost,
                face_enhancer_blend=face_enhancer_blend,
                lip_sync=False,
                source_paths=source_paths,
                timeout=timeout,
                face_detector_model=face_detector_model,
                face_detector_size=face_detector_size,
                face_detector_score=face_detector_score,
                face_selector_mode=face_selector_mode,
                face_mask_blur=face_mask_blur,
                progress_callback=progress_callback,
                processors=["face_enhancer"],
            )
            return {"enhanced": True, "warning": None}
        except RuntimeError as e:
            err_str = str(e)
            if auto_fallback and _is_oom_error(-9 if "exit -9" in err_str else 1, err_str):
                import shutil
                shutil.copy2(intermediate_path, output_path)
                return {
                    "enhanced": False,
                    "warning": (
                        "Face enhancement was skipped due to memory limits. "
                        "The result uses face swap only. To get enhanced output, "
                        "try a shorter video or lower pixel boost."
                    ),
                }
            raise
    finally:
        if os.path.isfile(intermediate_path):
            try:
                os.unlink(intermediate_path)
            except OSError:
                pass
