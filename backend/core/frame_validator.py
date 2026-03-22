"""Post-processing validator and repair for face-swap frame consistency.

After FaceFusion produces a video, this module checks every frame to verify
the face was actually swapped (by comparing face embeddings to the source).
Frames where the swap failed silently are repaired by copying the face region
from the nearest successfully-swapped neighbor frame with feathered blending.
"""

import os
import cv2
import shutil
import subprocess
import tempfile
import logging
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Face analysis helpers
# ---------------------------------------------------------------------------

def _load_face_analyzer(det_size=(640, 640)):
    """Load a lightweight InsightFace analyzer for validation."""
    import onnxruntime
    from insightface.app import FaceAnalysis

    providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        providers = ["CPUExecutionProvider"]

    models_dir = os.environ.get(
        "ULTRAFACESWAP_MODELS",
        str(Path(__file__).resolve().parent.parent.parent / "models"),
    )

    app = FaceAnalysis(name="buffalo_l", root=models_dir, providers=providers)
    app.prepare(ctx_id=0, det_size=det_size)
    return app


def _get_face_info(analyzer, img_bgr: np.ndarray) -> Optional[Dict]:
    """Detect the highest-confidence face; return embedding + geometry."""
    faces = analyzer.get(img_bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.det_score)
    return {
        "embedding": face.normed_embedding,
        "bbox": face.bbox.astype(int).tolist(),
        "kps": face.kps.tolist() if face.kps is not None else None,
        "score": float(face.det_score),
    }


# ---------------------------------------------------------------------------
# Frame repair helpers
# ---------------------------------------------------------------------------

def _create_feathered_mask(h: int, w: int, feather_ratio: float = 0.25) -> np.ndarray:
    """Elliptical mask with Gaussian-blurred edges for seamless blending."""
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (int(w * 0.43), int(h * 0.43))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur_k = max(3, int(min(h, w) * feather_ratio) | 1)
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    return mask[:, :, np.newaxis]


def _extract_face_crop(frame: np.ndarray, face: Dict, expand: float = 0.45):
    """Extract an expanded face region and its placement coordinates."""
    h, w = frame.shape[:2]
    bb = face["bbox"]
    cx, cy = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
    fw, fh = bb[2] - bb[0], bb[3] - bb[1]
    ew, eh = int(fw * (1 + expand)), int(fh * (1 + expand))

    sx1 = max(0, cx - ew // 2)
    sy1 = max(0, cy - eh // 2)
    sx2 = min(w, cx + ew // 2)
    sy2 = min(h, cy + eh // 2)
    crop = frame[sy1:sy2, sx1:sx2].copy()
    return crop, (sx1, sy1, cx, cy)


def _paste_face_crop(
    target_frame: np.ndarray,
    crop: np.ndarray,
    src_origin: tuple,
    dst_offset: tuple = (0, 0),
) -> np.ndarray:
    """Paste *crop* onto *target_frame* at *src_origin* shifted by *dst_offset*, with feathered blending."""
    h, w = target_frame.shape[:2]
    ch, cw = crop.shape[:2]
    if ch < 20 or cw < 20:
        return target_frame

    tx1 = src_origin[0] + dst_offset[0]
    ty1 = src_origin[1] + dst_offset[1]

    c_x1 = max(0, -tx1)
    c_y1 = max(0, -ty1)
    d_x1 = max(0, tx1)
    d_y1 = max(0, ty1)
    pw = min(cw - c_x1, w - d_x1)
    ph = min(ch - c_y1, h - d_y1)
    if pw < 20 or ph < 20:
        return target_frame

    src = crop[c_y1:c_y1 + ph, c_x1:c_x1 + pw].astype(np.float32)
    mask = _create_feathered_mask(ph, pw)

    result = target_frame.copy()
    dst = result[d_y1:d_y1 + ph, d_x1:d_x1 + pw].astype(np.float32)
    blended = src * mask + dst * (1.0 - mask)
    result[d_y1:d_y1 + ph, d_x1:d_x1 + pw] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def _repair_frame(
    bad_frame: np.ndarray,
    good_frame: np.ndarray,
    good_face: Dict,
    bad_face: Optional[Dict] = None,
    expand: float = 0.45,
) -> np.ndarray:
    """Copy the swapped face from *good_frame* onto *bad_frame* with feathered blending.

    If *bad_face* is provided, the paste position is adjusted to track
    the face's movement between the two frames.
    """
    crop, (sx1, sy1, gcx, gcy) = _extract_face_crop(good_frame, good_face, expand)

    dx, dy = 0, 0
    if bad_face is not None:
        bb = bad_face["bbox"]
        bcx, bcy = (bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2
        dx, dy = bcx - gcx, bcy - gcy

    return _paste_face_crop(bad_frame, crop, (sx1, sy1), (dx, dy))


def _repair_frame_interpolated(
    bad_frame: np.ndarray,
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    prev_face: Dict,
    next_face: Dict,
    bad_face: Optional[Dict],
    t: float,
    expand: float = 0.45,
) -> np.ndarray:
    """Repair using weighted blend of prev and next good frames.

    *t* is 0.0 at *prev_frame* and 1.0 at *next_frame*.
    The face crop position is linearly interpolated, and the pixel
    content is alpha-blended between the two sources.
    """
    crop_a, (ax1, ay1, acx, acy) = _extract_face_crop(prev_frame, prev_face, expand)
    crop_b, (bx1, by1, bcx, bcy) = _extract_face_crop(next_frame, next_face, expand)

    ch = min(crop_a.shape[0], crop_b.shape[0])
    cw = min(crop_a.shape[1], crop_b.shape[1])
    if ch < 20 or cw < 20:
        return _repair_frame(bad_frame, prev_frame, prev_face, bad_face, expand)

    crop_a = cv2.resize(crop_a, (cw, ch))
    crop_b = cv2.resize(crop_b, (cw, ch))
    blended_crop = ((1.0 - t) * crop_a.astype(np.float32) +
                    t * crop_b.astype(np.float32))
    blended_crop = np.clip(blended_crop, 0, 255).astype(np.uint8)

    interp_x1 = int(ax1 * (1 - t) + bx1 * t)
    interp_y1 = int(ay1 * (1 - t) + by1 * t)
    interp_cx = int(acx * (1 - t) + bcx * t)
    interp_cy = int(acy * (1 - t) + bcy * t)

    dx, dy = 0, 0
    if bad_face is not None:
        bb = bad_face["bbox"]
        face_cx = (bb[0] + bb[2]) // 2
        face_cy = (bb[1] + bb[3]) // 2
        dx = face_cx - interp_cx
        dy = face_cy - interp_cy

    return _paste_face_crop(bad_frame, blended_crop, (interp_x1, interp_y1), (dx, dy))


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def _extract_frames_to_dir(video_path: str, frames_dir: str) -> int:
    pattern = os.path.join(frames_dir, "frame_%06d.png")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-q:v", "1", pattern],
        capture_output=True, text=True, check=True,
    )
    return len(list(Path(frames_dir).glob("frame_*.png")))


def _extract_audio(video_path: str, audio_path: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", audio_path],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and os.path.isfile(audio_path) and os.path.getsize(audio_path) > 0


def _get_fps(video_path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return 30.0
    txt = r.stdout.strip()
    if "/" in txt:
        n, d = map(int, txt.split("/"))
        return n / d if d else 30.0
    return float(txt) if txt else 30.0


def _merge_frames(frames_dir: str, output_path: str, fps: float, audio_path: Optional[str] = None):
    pattern = os.path.join(frames_dir, "frame_%06d.png")
    if audio_path and os.path.isfile(audio_path):
        temp = output_path + ".repair_tmp.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", temp],
            capture_output=True, text=True, check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
            capture_output=True, text=True, check=True,
        )
        if os.path.isfile(temp):
            os.unlink(temp)
    else:
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path],
            capture_output=True, text=True, check=True,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _group_consecutive_runs(indices: List[int]) -> List[List[int]]:
    """Group a sorted list of indices into runs of consecutive integers."""
    if not indices:
        return []
    runs: List[List[int]] = [[indices[0]]]
    for i in indices[1:]:
        if i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def validate_and_repair(
    source_img_path: str,
    output_video_path: str,
    *,
    source_paths: Optional[List[str]] = None,
    similarity_threshold: float = 0.3,
    max_neighbor_distance: int = 15,
    temporal_smooth: bool = True,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict:
    """Validate every frame and repair ones where the swap failed.

    Uses InsightFace face recognition to compare each output frame's face
    against the source.  Frames that don't match are repaired by blending
    the face region from the nearest successfully-swapped neighbor.

    When *temporal_smooth* is True, consecutive bad frames between two good
    anchors are repaired with linearly interpolated face blending instead of
    copying from a single neighbor.  This avoids the "frozen face" artifact.

    Modifies *output_video_path* in-place if repairs are needed.

    Args:
        source_img_path: Path to source face image.
        output_video_path: FaceFusion output video (overwritten if repaired).
        source_paths: Additional source images (multi-angle).
        similarity_threshold: Cosine similarity below this marks a bad frame.
        max_neighbor_distance: Skip repair if nearest good frame is farther.
        temporal_smooth: Interpolate between two good neighbors for smoother repair.
        progress_callback: ``callback(stage, current, total)``

    Returns:
        Dict with total_frames, good_frames, repaired_frames, failed_frames,
        repair_details (human-readable summary).
    """
    result: Dict = {
        "total_frames": 0,
        "good_frames": 0,
        "repaired_frames": 0,
        "failed_frames": 0,
        "repair_details": "",
    }

    # ---- Load analyzer ----
    try:
        analyzer = _load_face_analyzer(det_size=(640, 640))
    except Exception as exc:
        logger.warning("Frame validator: cannot load analyzer: %s", exc)
        result["repair_details"] = f"Skipped validation (analyzer unavailable: {exc})"
        return result

    # ---- Source embeddings ----
    all_source = list(source_paths or [])
    if source_img_path not in all_source:
        all_source.insert(0, source_img_path)

    source_embeddings: List[np.ndarray] = []
    for sp in all_source:
        img = cv2.imread(sp)
        if img is None:
            continue
        info = _get_face_info(analyzer, img)
        if info is not None:
            source_embeddings.append(info["embedding"])

    if not source_embeddings:
        del analyzer
        result["repair_details"] = "Skipped validation: no face found in source image."
        return result

    # ---- Validation pass (read via VideoCapture — no disk I/O) ----
    cap = cv2.VideoCapture(output_video_path)
    if not cap.isOpened():
        del analyzer
        result["repair_details"] = "Skipped validation: could not open output video."
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 5:
        cap.release()
        del analyzer
        result["total_frames"] = total
        result["good_frames"] = total
        result["repair_details"] = "Video too short for validation."
        return result

    frame_data: List[Dict] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        info = _get_face_info(analyzer, frame)
        if info is None:
            frame_data.append({"good": False, "face_info": None, "similarity": -1.0})
        else:
            max_sim = max(float(np.dot(info["embedding"], se)) for se in source_embeddings)
            frame_data.append({
                "good": max_sim >= similarity_threshold,
                "face_info": {k: v for k, v in info.items() if k != "embedding"},
                "similarity": max_sim,
            })

        idx += 1
        if progress_callback:
            progress_callback("validating", idx, total)

    cap.release()
    del analyzer

    actual_total = len(frame_data)
    result["total_frames"] = actual_total

    good_indices = [i for i, d in enumerate(frame_data) if d["good"]]
    bad_indices = [i for i, d in enumerate(frame_data) if not d["good"]]
    result["good_frames"] = len(good_indices)

    if not bad_indices:
        result["repair_details"] = f"All {actual_total} frames validated — no flickering detected."
        return result

    if not good_indices:
        result["failed_frames"] = len(bad_indices)
        result["repair_details"] = (
            f"None of the {actual_total} frames passed validation. "
            "Face swap may have failed entirely. Try a clearer source photo."
        )
        return result

    # ---- Repair pass ----
    frames_dir = tempfile.mkdtemp(prefix="uf_repair_")
    audio_path = os.path.join(frames_dir, "audio.aac")

    good_set = set(good_indices)

    try:
        _extract_frames_to_dir(output_video_path, frames_dir)
        has_audio = _extract_audio(output_video_path, audio_path)
        video_fps = _get_fps(output_video_path)

        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
        n_frames = len(frame_files)

        repaired = 0
        failed = 0
        progress_done = 0
        total_bad = len(bad_indices)

        if temporal_smooth:
            runs = _group_consecutive_runs(sorted(bad_indices))
            for run in runs:
                prev_good = max((g for g in good_indices if g < run[0]), default=None)
                next_good = min((g for g in good_indices if g > run[-1]), default=None)

                prev_dist = (run[0] - prev_good) if prev_good is not None else float("inf")
                next_dist = (next_good - run[-1]) if next_good is not None else float("inf")
                span = (run[-1] - run[0]) + 1

                if min(prev_dist, next_dist) > max_neighbor_distance:
                    failed += span
                    progress_done += span
                    if progress_callback:
                        progress_callback("repairing", progress_done, total_bad)
                    continue

                has_both = (prev_good is not None and next_good is not None
                            and prev_dist <= max_neighbor_distance
                            and next_dist <= max_neighbor_distance)

                if has_both:
                    prev_img = cv2.imread(str(frame_files[prev_good]))
                    next_img = cv2.imread(str(frame_files[next_good]))
                    total_gap = next_good - prev_good
                else:
                    anchor = prev_good if prev_good is not None and prev_dist <= max_neighbor_distance else next_good
                    anchor_img = cv2.imread(str(frame_files[anchor]))

                for bad_idx in run:
                    if bad_idx >= n_frames:
                        failed += 1
                        progress_done += 1
                        continue

                    bad_img = cv2.imread(str(frame_files[bad_idx]))
                    if bad_img is None:
                        failed += 1
                        progress_done += 1
                        continue

                    bad_face = frame_data[bad_idx].get("face_info")

                    if has_both and prev_img is not None and next_img is not None:
                        t = (bad_idx - prev_good) / total_gap if total_gap > 0 else 0.5
                        fixed = _repair_frame_interpolated(
                            bad_img, prev_img, next_img,
                            frame_data[prev_good]["face_info"],
                            frame_data[next_good]["face_info"],
                            bad_face, t,
                        )
                    else:
                        a_face = frame_data[anchor]["face_info"]
                        if anchor_img is not None and a_face is not None:
                            fixed = _repair_frame(bad_img, anchor_img, a_face, bad_face)
                        else:
                            failed += 1
                            progress_done += 1
                            continue

                    cv2.imwrite(str(frame_files[bad_idx]), fixed)
                    repaired += 1
                    progress_done += 1

                    if progress_callback:
                        progress_callback("repairing", progress_done, total_bad)
        else:
            for j, bad_idx in enumerate(bad_indices):
                nearest_good = min(good_indices, key=lambda g: abs(g - bad_idx))
                distance = abs(nearest_good - bad_idx)

                if distance > max_neighbor_distance:
                    failed += 1
                    if progress_callback:
                        progress_callback("repairing", j + 1, total_bad)
                    continue

                if bad_idx >= n_frames or nearest_good >= n_frames:
                    failed += 1
                    continue

                bad_img = cv2.imread(str(frame_files[bad_idx]))
                good_img = cv2.imread(str(frame_files[nearest_good]))
                if bad_img is None or good_img is None:
                    failed += 1
                    continue

                fixed = _repair_frame(
                    bad_img, good_img,
                    frame_data[nearest_good]["face_info"],
                    frame_data[bad_idx].get("face_info"),
                )
                cv2.imwrite(str(frame_files[bad_idx]), fixed)
                repaired += 1

                if progress_callback:
                    progress_callback("repairing", j + 1, total_bad)

        result["repaired_frames"] = repaired
        result["failed_frames"] = failed

        if repaired > 0:
            _merge_frames(
                frames_dir,
                output_video_path,
                video_fps,
                audio_path if has_audio else None,
            )

        pct_good = len(good_indices) / actual_total * 100
        parts = [f"{len(good_indices)}/{actual_total} frames ({pct_good:.0f}%) swapped correctly."]
        if repaired > 0:
            mode = "with temporal smoothing" if temporal_smooth else "from neighbors"
            parts.append(f"{repaired} frames repaired {mode}.")
        if failed > 0:
            parts.append(
                f"{failed} frames could not be repaired (nearest good frame too far). "
                "Consider a clearer source photo or different detector settings."
            )
        result["repair_details"] = " ".join(parts)

    except Exception as exc:
        logger.exception("Frame repair failed")
        result["repair_details"] = f"Frame repair encountered an error: {exc}. Output video unchanged."
    finally:
        if os.path.isdir(frames_dir):
            shutil.rmtree(frames_dir, ignore_errors=True)

    return result
