"""FastAPI routes for UltraFaceswap."""

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.queue.jobs import Job, JobStatus, job_store
from backend.core.extractor import extract_frames, get_frame_count
from backend.core.face_swap import FaceSwapper, load_source_face
from backend.core.enhancer import enhance_face
from backend.core.upscaler import upscale_image
from backend.core.interpolator import interpolate_frames
from backend.core.merger import merge_frames_to_video, get_video_fps
from backend.core.analyzer import suggest_settings
from backend.core.hair import apply_hair_color_matching
from backend.core.cloth import apply_cloth_color_change, apply_cloth_color_change_to_video, parse_color_hex
from backend.core.facefusion_runner import run_facefusion, is_facefusion_available
import cv2

router = APIRouter(prefix="/api", tags=["face-swap"])

# Output directory for completed jobs
OUTPUT_DIR = os.environ.get("ULTRAFACESWAP_OUTPUT", "/tmp/ultrafaceswap_output")


@router.get("/capabilities")
async def get_capabilities() -> dict:
    """Return available engines. FaceFusion requires separate install + FACEFUSION_PATH."""
    return {
        "classic": True,
        "facefusion": is_facefusion_available(),
    }


def _settings_suffix(
    swap_model: str,
    det_size: int,
    upscale: int,
    interpolate: int,
    enhance: bool,
    hair_match: bool,
    engine: str = "classic",
    facefusion_model: str = "",
    facefusion_pixel_boost: str = "",
) -> str:
    """Short settings string for filenames: model_d640_u1_i1_enh0_hair1 or facefusion_inswapper128_p128"""
    if engine == "facefusion":
        return f"facefusion_{facefusion_model}_p{facefusion_pixel_boost}"
    return f"{swap_model}_d{det_size}_u{upscale}_i{interpolate}_enh{1 if enhance else 0}_hair{1 if hair_match else 0}"


def _run_facefusion_task(
    job_id: str,
    source_path: str,
    target_path: str,
    facefusion_model: str = "inswapper_128_fp16",
    facefusion_pixel_boost: str = "128",
    facefusion_face_enhancer: bool = False,
    facefusion_lip_sync: bool = False,
    cloth_color: Optional[str] = None,
    cloth_strength: float = 0.6,
) -> None:
    """Run FaceFusion subprocess for face swap. Cloth change is applied to target video first if cloth_color set."""
    if not is_facefusion_available():
        job_store.update(job_id, status=JobStatus.FAILED, error="FaceFusion not configured. Set ULTRAFACESWAP_FACEFUSION_PATH.")
        for p in [source_path, target_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return

    settings = {
        "engine": "facefusion",
        "facefusion_model": facefusion_model,
        "facefusion_pixel_boost": facefusion_pixel_boost,
        "facefusion_face_enhancer": facefusion_face_enhancer,
        "facefusion_lip_sync": facefusion_lip_sync,
    }
    job_store.update(
        job_id,
        status=JobStatus.PROCESSING,
        stage="swapping",
        progress=10,
        settings=settings,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    cloth_video_path: Optional[str] = None
    video_to_swap = target_path

    try:
        if cloth_color and parse_color_hex(cloth_color) is not None:
            job_store.update(job_id, stage="cloth")
            fd, cloth_video_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            apply_cloth_color_change_to_video(
                target_path,
                cloth_video_path,
                cloth_color,
                strength=cloth_strength,
            )
            video_to_swap = cloth_video_path
        job_store.update(job_id, progress=50, stage="swapping")

        stop_progress = threading.Event()

        def _facefusion_progress_loop() -> None:
            """Gradually bump progress 50 -> 90 while FaceFusion runs (no real progress from subprocess)."""
            while not stop_progress.wait(timeout=8):
                job = job_store.get(job_id)
                if not job or job.status != JobStatus.PROCESSING or job.progress >= 90:
                    break
                job_store.update(job_id, progress=min(90, job.progress + 6))

        progress_thread = threading.Thread(target=_facefusion_progress_loop, daemon=True)
        progress_thread.start()
        try:
            run_facefusion(
                source_path,
                video_to_swap,
                output_path,
                face_swapper_model=facefusion_model,
                face_swapper_pixel_boost=facefusion_pixel_boost,
                face_enhancer_blend=0.5 if facefusion_face_enhancer else 0.0,
                lip_sync=facefusion_lip_sync,
            )
        finally:
            stop_progress.set()
            progress_thread.join(timeout=1)

        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            stage="done",
            result_path=output_path,
            settings=settings,
        )
    except Exception as e:
        job_store.update(job_id, status=JobStatus.FAILED, error=str(e))
    finally:
        for p in [source_path, target_path, cloth_video_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def run_swap_task(
    job_id: str,
    source_path: str,
    target_path: str,
    engine: str = "classic",
    use_enhancer: bool = False,
    swap_model: str = "inswapper",
    det_size: int = 640,
    upscale: int = 1,
    interpolate: int = 1,
    hair_match: bool = True,
    cloth_color: Optional[str] = None,
    cloth_strength: float = 0.6,
    facefusion_model: str = "inswapper_128_fp16",
    facefusion_pixel_boost: str = "128",
    facefusion_face_enhancer: bool = False,
    facefusion_lip_sync: bool = False,
) -> None:
    """Run face swap in background thread."""
    try:
        if engine == "facefusion":
            _run_facefusion_task(
                job_id, source_path, target_path,
                facefusion_model=facefusion_model,
                facefusion_pixel_boost=facefusion_pixel_boost,
                facefusion_face_enhancer=facefusion_face_enhancer,
                facefusion_lip_sync=facefusion_lip_sync,
                cloth_color=cloth_color,
                cloth_strength=cloth_strength,
            )
            return

        settings = {
            "engine": "classic",
            "swap_model": swap_model,
            "det_size": det_size,
            "upscale": upscale,
            "interpolate": interpolate,
            "enhance": use_enhancer,
            "hair_match": hair_match,
        }
        job_store.update(
            job_id,
            status=JobStatus.PROCESSING,
            stage="extracting",
            settings=settings,
        )

        frames_dir, audio_path = extract_frames(target_path)
        frame_count = get_frame_count(frames_dir)
        fps = get_video_fps(target_path)

        job_store.update(
            job_id,
            total_frames=frame_count,
            progress=5,
            stage="swapping",
        )

        det_size_tuple = (det_size, det_size)
        source_bgr = load_source_face(source_path)
        swapper = FaceSwapper(swap_model=swap_model, det_size=det_size_tuple)
        source_bbox = swapper.get_primary_face_bbox(source_bgr)

        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
        for i, frame_path in enumerate(frame_files):
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                continue
            try:
                # Cloth color change on original frame first (before face swap)
                if cloth_color and parse_color_hex(cloth_color) is not None:
                    target_bbox = swapper.get_primary_face_bbox(frame_bgr)
                    if target_bbox is not None:
                        frame_bgr = apply_cloth_color_change(
                            frame_bgr, target_bbox, cloth_color, strength=cloth_strength
                        )
                result = swapper.process_frame(source_bgr, frame_bgr)
                if hair_match and source_bbox is not None:
                    target_bbox = swapper.get_primary_face_bbox(result)
                    if target_bbox is not None:
                        result = apply_hair_color_matching(
                            source_bgr, result, source_bbox, target_bbox, strength=0.55
                        )
                if use_enhancer:
                    result = enhance_face(result, use_gfpgan=True)
                if upscale in (2, 4):
                    result = upscale_image(result, scale=upscale)
                cv2.imwrite(str(frame_path), result)
            except Exception as e:
                import traceback
                print(f"[UltraFaceswap] Frame {frame_path.name} swap failed: {e}")
                traceback.print_exc()
                # Write original frame so pipeline continues; avoid silent failure
                cv2.imwrite(str(frame_path), frame_bgr)
            progress = 5 + int(85 * (i + 1) / len(frame_files))
            job_store.update(job_id, progress=progress, processed_frames=i + 1)

        output_fps = fps
        if interpolate in (2, 4):
            job_store.update(job_id, stage="interpolating", progress=88)
            interpolate_frames(frames_dir, factor=interpolate, input_format="png")
            output_fps = fps * interpolate

        job_store.update(job_id, stage="merging", progress=92)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

        merge_frames_to_video(
            frames_dir,
            output_path,
            fps=output_fps,
            audio_path=audio_path,
            input_format="png",
        )

        shutil.rmtree(frames_dir, ignore_errors=True)
        os.remove(source_path)
        os.remove(target_path)

        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            stage="done",
            result_path=output_path,
            settings=settings,
        )
    except Exception as e:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=str(e),
        )
        # Cleanup
        for p in [source_path, target_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@router.post("/suggest")
async def suggest_settings_endpoint(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
) -> dict:
    """Analyze uploaded files and suggest best settings."""
    if not source.content_type or not source.content_type.startswith("image/"):
        raise HTTPException(400, "Source must be an image")
    if not target.content_type or "video" not in target.content_type:
        raise HTTPException(400, "Target must be a video")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source.filename or "img").suffix) as s:
        shutil.copyfileobj(source.file, s)
        source_path = s.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
        shutil.copyfileobj(target.file, t)
        target_path = t.name

    try:
        result = suggest_settings(source_path, target_path)
    finally:
        for p in (source_path, target_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    return result


def _run_facefusion_multi_task(
    job_id: str,
    source_paths: List[str],
    target_path: str,
) -> None:
    """Run FaceFusion with multiple source images (multi-angle). Fixed: hyperswap_1a_256, enhance on."""
    if not is_facefusion_available():
        job_store.update(job_id, status=JobStatus.FAILED, error="FaceFusion not configured.")
        for p in source_paths + [target_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return

    settings = {
        "engine": "facefusion",
        "facefusion_model": "hyperswap_1a_256",
        "facefusion_pixel_boost": "256",
        "facefusion_face_enhancer": True,
        "facefusion_lip_sync": False,
        "multi_angle": True,
    }
    job_store.update(
        job_id,
        status=JobStatus.PROCESSING,
        stage="swapping",
        progress=10,
        settings=settings,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")

    stop_progress = threading.Event()

    def _progress_loop() -> None:
        while not stop_progress.wait(timeout=8):
            job = job_store.get(job_id)
            if not job or job.status != JobStatus.PROCESSING or job.progress >= 90:
                break
            job_store.update(job_id, progress=min(90, job.progress + 6))

    progress_thread = threading.Thread(target=_progress_loop, daemon=True)
    progress_thread.start()

    try:
        job_store.update(job_id, progress=50, stage="swapping")
        run_facefusion(
            source_paths[0],
            target_path,
            output_path,
            face_swapper_model="hyperswap_1a_256",
            face_swapper_pixel_boost="256",
            face_enhancer_blend=0.5,
            lip_sync=False,
            source_paths=source_paths,
        )
    except Exception as e:
        job_store.update(job_id, status=JobStatus.FAILED, error=str(e))
    finally:
        stop_progress.set()
        progress_thread.join(timeout=1)

    for p in source_paths + [target_path]:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    job = job_store.get(job_id)
    if job and job.status != JobStatus.FAILED:
        job_store.update(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            stage="done",
            result_path=output_path,
            settings=settings,
        )


@router.post("/swap-multi")
async def create_swap_multi_job(
    sources: List[UploadFile] = File(...),
    target: UploadFile = File(...),
) -> dict:
    """
    Multi-angle face swap: upload 1–5 source photos (one or different angles) and target video.
    Uses FaceFusion only with hyperswap_1a_256 and enhancement on.
    """
    if not is_facefusion_available():
        raise HTTPException(400, "FaceFusion not available. Install FaceFusion and set ULTRAFACESWAP_FACEFUSION_PATH.")
    if len(sources) < 1 or len(sources) > 5:
        raise HTTPException(400, "Upload between 1 and 5 source images.")
    for f in sources:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(400, "All sources must be images (jpg, png).")
    if not target.content_type or "video" not in target.content_type:
        raise HTTPException(400, "Target must be a video (mp4, webm).")

    job = job_store.create()
    source_paths: List[str] = []
    try:
        for i, src in enumerate(sources):
            ext = Path(src.filename or "img").suffix or ".png"
            fd, path = tempfile.mkstemp(suffix=ext)
            os.close(fd)
            with open(path, "wb") as out:
                shutil.copyfileobj(src.file, out)
            source_paths.append(path)
        fd, target_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        with open(target_path, "wb") as out:
            shutil.copyfileobj(target.file, out)
    except Exception as e:
        for p in source_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise HTTPException(500, str(e))

    thread = threading.Thread(
        target=_run_facefusion_multi_task,
        args=(job.id, source_paths, target_path),
    )
    thread.start()

    return {
        "job_id": job.id,
        "status": job.status.value,
        "message": "Multi-angle job started. Poll /api/status/{job_id} for progress.",
    }


@router.post("/swap")
async def create_swap_job(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    engine: str = Form("classic"),
    enhance: bool = Form(False),
    swap_model: str = Form("inswapper"),
    det_size: int = Form(640),
    upscale: int = Form(1),
    interpolate: int = Form(1),
    hair_match: bool = Form(True),
    facefusion_model: str = Form("inswapper_128_fp16"),
    facefusion_pixel_boost: str = Form("128"),
    facefusion_face_enhancer: bool = Form(False),
    facefusion_lip_sync: bool = Form(False),
    cloth_color: Optional[str] = Form(None),
    cloth_strength: float = Form(0.6),
) -> dict:
    """
    Upload source photo and target video, start face swap job.
    Returns job ID for status polling.
    """
    if not source.content_type or not source.content_type.startswith("image/"):
        raise HTTPException(400, "Source must be an image (jpg, png)")
    if not target.content_type or "video" not in target.content_type:
        raise HTTPException(400, "Target must be a video (mp4, webm)")
    if engine not in ("classic", "facefusion"):
        engine = "classic"
    if engine == "facefusion" and not is_facefusion_available():
        raise HTTPException(400, "FaceFusion not available. Install FaceFusion and set ULTRAFACESWAP_FACEFUSION_PATH.")
    if swap_model not in ("inswapper", "simswap"):
        swap_model = "inswapper"
    if det_size not in (320, 640):
        det_size = 640
    if upscale not in (1, 2, 4):
        upscale = 1
    if interpolate not in (1, 2, 4):
        interpolate = 1

    job = job_store.create()

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source.filename or "img").suffix) as s:
        shutil.copyfileobj(source.file, s)
        source_path = s.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
        shutil.copyfileobj(target.file, t)
        target_path = t.name

    thread = threading.Thread(
        target=run_swap_task,
        args=(job.id, source_path, target_path),
        kwargs={
            "engine": engine,
            "use_enhancer": enhance,
            "swap_model": swap_model,
            "det_size": det_size,
            "upscale": upscale,
            "interpolate": interpolate,
            "hair_match": hair_match,
            "facefusion_model": facefusion_model,
            "facefusion_pixel_boost": facefusion_pixel_boost,
            "facefusion_face_enhancer": facefusion_face_enhancer,
            "facefusion_lip_sync": facefusion_lip_sync,
            "cloth_color": cloth_color,
            "cloth_strength": cloth_strength,
        },
    )
    thread.start()

    return {
        "job_id": job.id,
        "status": job.status.value,
        "message": "Job started. Poll /api/status/{job_id} for progress.",
    }


@router.get("/status/{job_id}")
async def get_status(job_id: str) -> dict:
    """Get job status and progress."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    """Download result video if job completed."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job not ready: {job.status.value}")
    if not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(404, "Result file not found")

    s = job.settings or {}
    engine = s.get("engine", "classic")
    if s.get("multi_angle"):
        suffix = "facefusion_hyperswap_1a_256_p256_multi"
    elif engine == "facefusion":
        suffix = _settings_suffix(
            "", 0, 1, 1, False, True,
            engine="facefusion",
            facefusion_model=s.get("facefusion_model", "inswapper_128_fp16"),
            facefusion_pixel_boost=s.get("facefusion_pixel_boost", "128"),
        )
    else:
        suffix = _settings_suffix(
            s.get("swap_model", "inswapper"),
            s.get("det_size", 640),
            s.get("upscale", 1),
            s.get("interpolate", 1),
            s.get("enhance", False),
            s.get("hair_match", True),
        )
    filename = f"ultrafaceswap_{suffix}.mp4"
    return FileResponse(
        job.result_path,
        media_type="video/mp4",
        filename=filename,
    )
