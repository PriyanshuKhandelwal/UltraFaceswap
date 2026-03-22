"""FastAPI routes for UltraFaceswap."""

import os
import shutil
import tempfile
import threading
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
from backend.core.analyzer import suggest_settings, get_video_metadata
from backend.core.hair import apply_hair_color_matching
from backend.core.cloth import apply_cloth_color_change, apply_cloth_color_change_to_video, parse_color_hex
from backend.core.facefusion_runner import (
    run_facefusion,
    run_facefusion_two_pass,
    is_facefusion_available,
    check_output_has_swapped_faces,
)
from backend.core.frame_validator import validate_and_repair
from backend.core.downloader import download_video, is_supported_url
import cv2

router = APIRouter(prefix="/api", tags=["face-swap"])

DEFAULT_FACE_PATH = os.environ.get(
    "DEFAULT_FACE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "assets", "default_face.png"),
)

def _get_default_face() -> Optional[str]:
    """Return the absolute path to the default face image, or None."""
    p = os.path.abspath(DEFAULT_FACE_PATH)
    return p if os.path.isfile(p) else None

OUTPUT_DIR = os.environ.get("ULTRAFACESWAP_OUTPUT", "/tmp/ultrafaceswap_output")

# ---------------------------------------------------------------------------
# Preset definitions — the single source of truth for Quick / Best / Max
# ---------------------------------------------------------------------------
PRESETS = {
    "quick": {
        "face_swapper_model": "hyperswap_1a_256",
        "pixel_boost": "256",
        "face_enhancer": False,
        "face_enhancer_blend": 0.0,
        "face_detector_model": "retinaface",
        "face_detector_score": 0.35,
        "face_selector_mode": "reference",
        "face_mask_blur": 0.3,
        "two_pass": False,
        "label": "Quick",
    },
    "best": {
        "face_swapper_model": "hyperswap_1a_256",
        "pixel_boost": "256",
        "face_enhancer": True,
        "face_enhancer_blend": 0.5,
        "face_detector_model": "retinaface",
        "face_detector_score": 0.35,
        "face_selector_mode": "reference",
        "face_mask_blur": 0.3,
        "two_pass": True,
        "label": "Best",
    },
    "max": {
        "face_swapper_model": "hyperswap_1a_256",
        "pixel_boost": "512",
        "face_enhancer": True,
        "face_enhancer_blend": 0.5,
        "face_detector_model": "retinaface",
        "face_detector_score": 0.3,
        "face_selector_mode": "reference",
        "face_mask_blur": 0.35,
        "two_pass": True,
        "label": "Max",
    },
}


@router.get("/capabilities")
async def get_capabilities() -> dict:
    """Return available engines and preset list."""
    return {
        "classic": True,
        "facefusion": is_facefusion_available(),
        "presets": {k: v["label"] for k, v in PRESETS.items()},
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
    if engine == "facefusion":
        return f"facefusion_{facefusion_model}_p{facefusion_pixel_boost}"
    return f"{swap_model}_d{det_size}_u{upscale}_i{interpolate}_enh{1 if enhance else 0}_hair{1 if hair_match else 0}"


# ---------------------------------------------------------------------------
# Core FaceFusion task — used by preset, pro, and multi-angle routes
# ---------------------------------------------------------------------------

def _run_facefusion_task(
    job_id: str,
    source_path: str,
    target_path: str,
    facefusion_model: str = "hyperswap_1a_256",
    facefusion_pixel_boost: str = "256",
    facefusion_face_enhancer: bool = True,
    facefusion_face_enhancer_blend: Optional[float] = None,
    facefusion_lip_sync: bool = False,
    cloth_color: Optional[str] = None,
    cloth_strength: float = 0.6,
    face_detector_model: Optional[str] = "retinaface",
    face_detector_size: Optional[str] = None,
    face_detector_score: Optional[float] = 0.35,
    face_selector_mode: Optional[str] = "reference",
    face_mask_blur: Optional[float] = 0.3,
    pro_mode: bool = False,
    preset: Optional[str] = None,
    two_pass: bool = True,
    source_paths: Optional[List[str]] = None,
    temporal_smooth: bool = True,
    _keep_source: bool = False,
) -> None:
    """Run FaceFusion subprocess for face swap with two-pass support and OOM fallback."""
    if not is_facefusion_available():
        job_store.update(job_id, status=JobStatus.FAILED, error="FaceFusion not configured. Set ULTRAFACESWAP_FACEFUSION_PATH.")
        if not _keep_source:
            _cleanup_paths([source_path, target_path])
        else:
            _cleanup_paths([target_path])
        return

    settings = {
        "engine": "facefusion",
        "facefusion_model": facefusion_model,
        "facefusion_pixel_boost": facefusion_pixel_boost,
        "facefusion_face_enhancer": facefusion_face_enhancer,
        "facefusion_lip_sync": facefusion_lip_sync,
    }
    if preset:
        settings["preset"] = preset
    if pro_mode:
        settings["pro_mode"] = True
    if source_paths and len(source_paths) > 1:
        settings["multi_angle"] = True
    if face_detector_model:
        settings["face_detector_model"] = face_detector_model
    if face_detector_size:
        settings["face_detector_size"] = face_detector_size
    if face_mask_blur is not None:
        settings["face_mask_blur"] = face_mask_blur
    if two_pass and facefusion_face_enhancer:
        settings["two_pass"] = True

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
                target_path, cloth_video_path, cloth_color, strength=cloth_strength,
            )
            video_to_swap = cloth_video_path

        video_meta = get_video_metadata(video_to_swap)
        total_frames = video_meta.get("frame_count") or 0
        if total_frames > 0:
            job_store.update(job_id, total_frames=total_frames, stage="swapping")

        enhancer_blend = 0.0
        if facefusion_face_enhancer:
            if facefusion_face_enhancer_blend is not None and 0 <= facefusion_face_enhancer_blend <= 1:
                enhancer_blend = facefusion_face_enhancer_blend
            else:
                enhancer_blend = 0.5

        is_two_pass = two_pass and enhancer_blend > 0
        current_pass = [0]

        def _on_progress(processed: int, total: int) -> None:
            if is_two_pass:
                base = 10 if current_pass[0] == 0 else 50
                span = 35 if current_pass[0] == 0 else 35
            else:
                base = 10
                span = 75
            pct = base + int(span * processed / total) if total > 0 else base
            job_store.update(
                job_id,
                processed_frames=processed,
                total_frames=total if total > 0 else total_frames,
                progress=min(pct, 88),
            )

        def _on_pass_started(pass_num: int) -> None:
            current_pass[0] = pass_num
            if pass_num == 0:
                job_store.update(job_id, progress=10, stage="swapping", processed_frames=0)
            else:
                job_store.update(job_id, progress=55, stage="enhancing", processed_frames=0)

        job_store.update(job_id, progress=10, stage="swapping")

        try:
            if is_two_pass:
                result_info = run_facefusion_two_pass(
                    source_path,
                    video_to_swap,
                    output_path,
                    face_swapper_model=facefusion_model,
                    face_swapper_pixel_boost=facefusion_pixel_boost,
                    face_enhancer_blend=enhancer_blend,
                    lip_sync=facefusion_lip_sync,
                    source_paths=source_paths,
                    face_detector_model=face_detector_model,
                    face_detector_size=face_detector_size,
                    face_detector_score=face_detector_score,
                    face_selector_mode=face_selector_mode,
                    face_mask_blur=face_mask_blur,
                    progress_callback=_on_progress,
                    pass_started_callback=_on_pass_started,
                    auto_fallback=True,
                )
                warning = result_info.get("warning")
                if warning:
                    settings["warning"] = warning
                    settings["facefusion_face_enhancer"] = False
            else:
                run_facefusion(
                    source_path,
                    video_to_swap,
                    output_path,
                    face_swapper_model=facefusion_model,
                    face_swapper_pixel_boost=facefusion_pixel_boost,
                    face_enhancer_blend=enhancer_blend,
                    lip_sync=facefusion_lip_sync,
                    source_paths=source_paths,
                    face_detector_model=face_detector_model,
                    face_detector_size=face_detector_size,
                    face_detector_score=face_detector_score,
                    face_selector_mode=face_selector_mode,
                    face_mask_blur=face_mask_blur,
                    progress_callback=_on_progress,
                )

            if os.path.isfile(output_path) and not check_output_has_swapped_faces(video_to_swap, output_path):
                settings["no_face_warning"] = True
                settings["warning"] = (
                    "No faces appear to have been swapped. The output looks identical to the input. "
                    "This usually means FaceFusion couldn't detect a face in the source photo or target video. "
                    "Try a clearer, front-facing source photo."
                )

            # --- Frame consistency validation & repair ---
            if os.path.isfile(output_path) and not settings.get("no_face_warning"):
                job_store.update(job_id, stage="validating", progress=89)

                def _on_validate_progress(stage: str, current: int, vtotal: int) -> None:
                    if stage == "validating":
                        pct = 89 + int(6 * current / vtotal) if vtotal > 0 else 89
                    else:
                        pct = 95 + int(4 * current / vtotal) if vtotal > 0 else 95
                    job_store.update(job_id, stage=stage, progress=min(pct, 99))

                try:
                    repair_result = validate_and_repair(
                        source_path,
                        output_path,
                        source_paths=source_paths,
                        temporal_smooth=temporal_smooth,
                        progress_callback=_on_validate_progress,
                    )
                    settings["validation"] = repair_result
                    if repair_result.get("repaired_frames", 0) > 0 or repair_result.get("failed_frames", 0) > 0:
                        settings["repair_details"] = repair_result.get("repair_details", "")
                except Exception as val_err:
                    settings["validation_error"] = str(val_err)

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
        paths_to_clean = [target_path, cloth_video_path]
        if not _keep_source:
            paths_to_clean.insert(0, source_path)
        if source_paths:
            paths_to_clean.extend(p for p in source_paths if p != source_path or not _keep_source)
        _cleanup_paths(paths_to_clean)


def _cleanup_paths(paths: list) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Preset-based endpoint (new simplified API)
# ---------------------------------------------------------------------------

@router.post("/swap-preset")
async def create_swap_preset_job(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    preset: str = Form("best"),
    lip_sync: bool = Form(False),
    temporal_smooth: bool = Form(True),
) -> dict:
    """
    Simple preset-based face swap. Presets: quick, best, max.
    Uses hyperswap_1a_256 + retinaface with two-pass enhance for best/max.
    """
    if not is_facefusion_available():
        raise HTTPException(400, "FaceFusion not available. Install FaceFusion and set ULTRAFACESWAP_FACEFUSION_PATH.")
    if not source.content_type or not source.content_type.startswith("image/"):
        raise HTTPException(400, "Source must be an image (jpg, png)")
    if not target.content_type or "video" not in target.content_type:
        raise HTTPException(400, "Target must be a video (mp4, webm)")
    if preset not in PRESETS:
        preset = "best"

    cfg = PRESETS[preset]
    job = job_store.create()

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source.filename or "img").suffix) as s:
        shutil.copyfileobj(source.file, s)
        source_path = s.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
        shutil.copyfileobj(target.file, t)
        target_path = t.name

    thread = threading.Thread(
        target=_run_facefusion_task,
        args=(job.id, source_path, target_path),
        kwargs={
            "facefusion_model": cfg["face_swapper_model"],
            "facefusion_pixel_boost": cfg["pixel_boost"],
            "facefusion_face_enhancer": cfg["face_enhancer"],
            "facefusion_face_enhancer_blend": cfg["face_enhancer_blend"],
            "facefusion_lip_sync": lip_sync,
            "face_detector_model": cfg["face_detector_model"],
            "face_detector_score": cfg.get("face_detector_score", 0.35),
            "face_selector_mode": cfg.get("face_selector_mode", "reference"),
            "face_mask_blur": cfg["face_mask_blur"],
            "two_pass": cfg["two_pass"],
            "preset": preset,
            "temporal_smooth": temporal_smooth,
        },
    )
    thread.start()

    return {
        "job_id": job.id,
        "status": job.status.value,
        "preset": preset,
        "message": f"{cfg['label']} job started. Poll /api/status/{job.id} for progress.",
    }


# ---------------------------------------------------------------------------
# URL-based endpoint (download from Instagram/Pinterest/TikTok + swap)
# ---------------------------------------------------------------------------

@router.post("/swap-from-url")
async def create_swap_from_url_job(
    url: str = Form(...),
    preset: str = Form("best"),
    source: Optional[UploadFile] = File(None),
    temporal_smooth: bool = Form(True),
) -> dict:
    """Download video from URL and face swap with default or custom face.

    Supports Instagram Reels, Pinterest pins, TikTok, YouTube, etc.
    If no source face is uploaded, the default face image is used.
    """
    if not is_facefusion_available():
        raise HTTPException(400, "FaceFusion not available.")
    if not is_supported_url(url):
        raise HTTPException(400, "URL not supported. Try Instagram, Pinterest, TikTok, or YouTube links.")

    if source and source.content_type and source.content_type.startswith("image/"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source.filename or "img").suffix) as s:
            shutil.copyfileobj(source.file, s)
            source_path = s.name
    else:
        default = _get_default_face()
        if not default:
            raise HTTPException(400, "No source face uploaded and no default face configured.")
        source_path = default

    try:
        target_path = download_video(url)
    except Exception as exc:
        if source_path != _get_default_face() and os.path.isfile(source_path):
            os.unlink(source_path)
        raise HTTPException(400, f"Could not download video: {exc}")

    if preset not in PRESETS:
        preset = "best"
    cfg = PRESETS[preset]
    job = job_store.create()

    is_default_face = source_path == _get_default_face()

    thread = threading.Thread(
        target=_run_facefusion_task,
        args=(job.id, source_path, target_path),
        kwargs={
            "facefusion_model": cfg["face_swapper_model"],
            "facefusion_pixel_boost": cfg["pixel_boost"],
            "facefusion_face_enhancer": cfg["face_enhancer"],
            "facefusion_face_enhancer_blend": cfg["face_enhancer_blend"],
            "face_detector_model": cfg["face_detector_model"],
            "face_detector_score": cfg.get("face_detector_score", 0.35),
            "face_selector_mode": cfg.get("face_selector_mode", "reference"),
            "face_mask_blur": cfg["face_mask_blur"],
            "two_pass": cfg["two_pass"],
            "preset": preset,
            "temporal_smooth": temporal_smooth,
            "_keep_source": is_default_face,
        },
    )
    thread.start()

    return {
        "job_id": job.id,
        "status": job.status.value,
        "preset": preset,
        "source": "default" if is_default_face else "uploaded",
        "message": f"Downloading & processing. Poll /api/status/{job.id} for progress.",
    }


# ---------------------------------------------------------------------------
# Multi-angle endpoint (now uses two-pass + retinaface by default)
# ---------------------------------------------------------------------------

@router.post("/swap-multi")
async def create_swap_multi_job(
    sources: List[UploadFile] = File(...),
    target: UploadFile = File(...),
    face_enhancer: bool = Form(True),
    temporal_smooth: bool = Form(True),
) -> dict:
    """
    Multi-angle face swap: upload 1-5 source photos and target video.
    Uses hyperswap_1a_256 + retinaface. Two-pass enhance by default.
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
        _cleanup_paths(source_paths)
        raise HTTPException(500, str(e))

    thread = threading.Thread(
        target=_run_facefusion_task,
        args=(job.id, source_paths[0], target_path),
        kwargs={
            "facefusion_model": "hyperswap_1a_256",
            "facefusion_pixel_boost": "256",
            "facefusion_face_enhancer": face_enhancer,
            "facefusion_face_enhancer_blend": 0.5 if face_enhancer else 0.0,
            "face_detector_model": "retinaface",
            "face_detector_score": 0.35,
            "face_selector_mode": "reference",
            "face_mask_blur": 0.3,
            "two_pass": face_enhancer,
            "source_paths": source_paths,
            "temporal_smooth": temporal_smooth,
        },
    )
    thread.start()

    return {
        "job_id": job.id,
        "status": job.status.value,
        "message": "Multi-angle job started. Poll /api/status/{job.id} for progress.",
    }


# ---------------------------------------------------------------------------
# Pro endpoint (advanced users — kept for backward compat / power users)
# ---------------------------------------------------------------------------

@router.post("/swap-pro")
async def create_swap_pro_job(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
    facefusion_model: str = Form("hyperswap_1a_256"),
    facefusion_pixel_boost: str = Form("256"),
    facefusion_face_enhancer: bool = Form(True),
    facefusion_face_enhancer_blend: float = Form(0.5),
    facefusion_lip_sync: bool = Form(False),
    face_detector_model: Optional[str] = Form("retinaface"),
    face_detector_size: Optional[str] = Form(None),
    face_detector_score: Optional[float] = Form(0.35),
    face_selector_mode: Optional[str] = Form("reference"),
    face_mask_blur: Optional[float] = Form(0.3),
    two_pass: bool = Form(True),
    temporal_smooth: bool = Form(True),
    cloth_color: Optional[str] = Form(None),
    cloth_strength: float = Form(0.6),
) -> dict:
    """Pro face swap with full FaceFusion options + two-pass support."""
    if not is_facefusion_available():
        raise HTTPException(400, "FaceFusion not available. Install FaceFusion and set ULTRAFACESWAP_FACEFUSION_PATH.")
    if not source.content_type or not source.content_type.startswith("image/"):
        raise HTTPException(400, "Source must be an image (jpg, png)")
    if not target.content_type or "video" not in target.content_type:
        raise HTTPException(400, "Target must be a video (mp4, webm)")

    job = job_store.create()
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(source.filename or "img").suffix) as s:
        shutil.copyfileobj(source.file, s)
        source_path = s.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
        shutil.copyfileobj(target.file, t)
        target_path = t.name

    blend = facefusion_face_enhancer_blend if facefusion_face_enhancer else 0.0

    thread = threading.Thread(
        target=_run_facefusion_task,
        args=(job.id, source_path, target_path),
        kwargs={
            "facefusion_model": facefusion_model,
            "facefusion_pixel_boost": facefusion_pixel_boost,
            "facefusion_face_enhancer": facefusion_face_enhancer,
            "facefusion_face_enhancer_blend": blend,
            "facefusion_lip_sync": facefusion_lip_sync,
            "cloth_color": cloth_color,
            "cloth_strength": cloth_strength,
            "face_detector_model": face_detector_model or "retinaface",
            "face_detector_size": face_detector_size or None,
            "face_detector_score": face_detector_score if face_detector_score is not None else 0.35,
            "face_selector_mode": face_selector_mode or "reference",
            "face_mask_blur": face_mask_blur,
            "pro_mode": True,
            "two_pass": two_pass and facefusion_face_enhancer,
            "temporal_smooth": temporal_smooth,
        },
    )
    thread.start()

    return {
        "job_id": job.id,
        "status": job.status.value,
        "message": "Pro job started. Poll /api/status/{job.id} for progress.",
    }


# ---------------------------------------------------------------------------
# Legacy standard swap endpoint (classic + facefusion engine)
# ---------------------------------------------------------------------------

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
    """Run face swap in background thread (legacy standard tab)."""
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
                two_pass=False,
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
        _cleanup_paths([source_path, target_path])


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
        _cleanup_paths([source_path, target_path])

    return result


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
    """Legacy standard swap endpoint."""
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
        "message": "Job started. Poll /api/status/{job.id} for progress.",
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
    preset = s.get("preset")
    if preset:
        enh = "enh1" if s.get("facefusion_face_enhancer") else "enh0"
        suffix = f"{preset}_{s.get('facefusion_model', 'hyperswap_1a_256')}_p{s.get('facefusion_pixel_boost', '256')}_{enh}"
    elif s.get("pro_mode"):
        suffix = f"pro_{s.get('facefusion_model', 'hyperswap_1a_256')}_p{s.get('facefusion_pixel_boost', '256')}"
    elif s.get("multi_angle"):
        suffix = "multi_hyperswap_1a_256_p256"
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
