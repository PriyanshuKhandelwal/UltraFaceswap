"""FastAPI routes for UltraFaceswap."""

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

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
import cv2

router = APIRouter(prefix="/api", tags=["face-swap"])

# Output directory for completed jobs
OUTPUT_DIR = os.environ.get("ULTRAFACESWAP_OUTPUT", "/tmp/ultrafaceswap_output")


def _settings_suffix(swap_model: str, det_size: int, upscale: int, interpolate: int, enhance: bool, hair_match: bool) -> str:
    """Short settings string for filenames: model_d640_u1_i1_enh0_hair1"""
    return f"{swap_model}_d{det_size}_u{upscale}_i{interpolate}_enh{1 if enhance else 0}_hair{1 if hair_match else 0}"


def run_swap_task(
    job_id: str,
    source_path: str,
    target_path: str,
    use_enhancer: bool = False,
    swap_model: str = "inswapper",
    det_size: int = 640,
    upscale: int = 1,
    interpolate: int = 1,
    hair_match: bool = True,
) -> None:
    """Run face swap in background thread."""
    try:
        job_store.update(job_id, status=JobStatus.PROCESSING, stage="extracting")

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

        settings = {
            "swap_model": swap_model,
            "det_size": det_size,
            "upscale": upscale,
            "interpolate": interpolate,
            "enhance": use_enhancer,
            "hair_match": hair_match,
        }
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


@router.post("/swap")
async def create_swap_job(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    enhance: bool = Form(False),
    swap_model: str = Form("inswapper"),
    det_size: int = Form(640),
    upscale: int = Form(1),
    interpolate: int = Form(1),
    hair_match: bool = Form(True),
) -> dict:
    """
    Upload source photo and target video, start face swap job.
    Returns job ID for status polling.
    """
    if not source.content_type or not source.content_type.startswith("image/"):
        raise HTTPException(400, "Source must be an image (jpg, png)")
    if not target.content_type or "video" not in target.content_type:
        raise HTTPException(400, "Target must be a video (mp4, webm)")
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
            "use_enhancer": enhance,
            "swap_model": swap_model,
            "det_size": det_size,
            "upscale": upscale,
            "interpolate": interpolate,
            "hair_match": hair_match,
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
