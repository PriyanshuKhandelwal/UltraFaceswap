"""Cloth color change - recolor torso/clothing region below the face."""

import os
import re
import shutil
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Union


def parse_color_hex(hex_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse '#RRGGBB' or 'RRGGBB' to BGR (OpenCV order). Returns (B, G, R) or None."""
    if not hex_str or not isinstance(hex_str, str):
        return None
    hex_str = hex_str.strip().lstrip("#")
    if not re.match(r"^[0-9A-Fa-f]{6}$", hex_str):
        return None
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)


def _get_cloth_region_mask(
    img_shape: Tuple[int, int],
    face_bbox: Tuple[int, int, int, int],
    expand_below: float = 2.2,
    expand_sides: float = 0.55,
    neck_margin: float = 0.2,
) -> np.ndarray:
    """Create a soft mask for the torso/cloth region only (below chin/neck).

    Mask starts below the face bbox so it never covers face, jaw, or hair.
    Uses soft elliptical-style falloff to avoid visible rectangular overlay.

    Args:
        img_shape: (height, width) of image
        face_bbox: (x1, y1, x2, y2) face bounding box
        expand_below: How many face heights below chin to include (torso)
        expand_sides: Horizontal expansion as fraction of face width (each side)
        neck_margin: Extra gap in face heights below face bottom before mask starts

    Returns:
        Float mask [0, 1] of shape (H, W)
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = face_bbox
    fw = x2 - x1
    fh = y2 - y1

    # Start mask below the face (below chin) + small neck margin so we never touch face/neck
    cloth_y1 = y2 + int(fh * neck_margin)
    cloth_y2 = min(h, cloth_y1 + int(fh * expand_below))
    pad_x = int(fw * expand_sides)
    cloth_x1 = max(0, x1 - pad_x)
    cloth_x2 = min(w, x2 + pad_x)

    mask = np.zeros((h, w), dtype=np.float32)
    if cloth_y2 <= cloth_y1 or cloth_x2 <= cloth_x1:
        return mask

    sub_h = cloth_y2 - cloth_y1
    sub_w = cloth_x2 - cloth_x1
    if sub_h < 2 or sub_w < 2:
        return mask

    # Soft elliptical-style mask (stronger in center, soft edges) to avoid boxy overlay
    yc, xc = sub_h / 2.0, sub_w / 2.0
    yy, xx = np.mgrid[0:sub_h, 0:sub_w].astype(np.float32)
    # Elliptical falloff: 1 in center, 0 at edges
    ey = (yy - yc) / (yc + 1e-6)
    ex = (xx - xc) / (xc + 1e-6)
    soft = 1.0 - np.minimum(1.0, np.sqrt(ey * ey + ex * ex))
    soft = np.clip(soft, 0, 1).astype(np.float32)
    # Strong blur for very soft edges (no visible rectangle)
    k = max(5, min(sub_w, sub_h) // 2 | 1)
    soft = cv2.GaussianBlur(soft, (k, k), k * 0.3)
    mask[cloth_y1:cloth_y2, cloth_x1:cloth_x2] = soft
    return np.clip(mask, 0, 1).astype(np.float32)


def _recolor_region_to_target_lab(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    target_bgr: Tuple[int, int, int],
    strength: float = 0.6,
) -> np.ndarray:
    """Recolor masked region toward target BGR color in LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_bgr_arr = np.array([[target_bgr]], dtype=np.uint8)
    target_lab = cv2.cvtColor(target_bgr_arr, cv2.COLOR_BGR2LAB)
    target_mean_lab = target_lab[0, 0].astype(np.float32)

    valid = mask > 0.01
    if not np.any(valid):
        return img_bgr

    pixels = lab[valid]
    src_mean = np.mean(pixels, axis=0)
    shift = (target_mean_lab - src_mean) * strength
    mask_3d = np.stack([mask] * 3, axis=-1)
    lab_out = lab + mask_3d * shift
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


def apply_cloth_color_change(
    frame_bgr: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    target_color: Union[Tuple[int, int, int], str],
    strength: float = 0.6,
) -> np.ndarray:
    """
    Change clothing/torso color toward a target color.

    Uses face bbox to define a torso region below the face and shifts
    its color toward target_color in LAB space.

    Args:
        frame_bgr: Frame image (BGR)
        face_bbox: Face bbox (x1, y1, x2, y2) in frame
        target_color: Target color as (B, G, R) or hex string '#RRGGBB'
        strength: Blend strength 0..1 (0.5–0.8 typical)

    Returns:
        Frame with cloth region recolored toward target_color
    """
    if isinstance(target_color, str):
        bgr = parse_color_hex(target_color)
        if bgr is None:
            return frame_bgr
    else:
        bgr = tuple(int(x) for x in target_color[:3])

    mask = _get_cloth_region_mask(frame_bgr.shape, face_bbox)
    if not np.any(mask > 0.01):
        return frame_bgr
    return _recolor_region_to_target_lab(frame_bgr, mask, bgr, strength=strength)


def apply_cloth_color_change_to_video(
    video_path: str,
    output_path: str,
    target_color: Union[Tuple[int, int, int], str],
    strength: float = 0.6,
    temp_dir: Optional[str] = None,
) -> str:
    """
    Apply cloth color change to every frame of a video (e.g. after FaceFusion).

    Extracts frames, detects face per frame, applies cloth recolor, merges back.
    """
    from backend.core.extractor import extract_frames, get_frame_count
    from backend.core.merger import merge_frames_to_video, get_video_fps
    from backend.core.face_swap import FaceSwapper

    if isinstance(target_color, str) and parse_color_hex(target_color) is None:
        raise ValueError(f"Invalid cloth color: {target_color!r}")
    frames_dir, audio_path = extract_frames(
        video_path, output_dir=temp_dir, image_format="png"
    )
    frame_count = get_frame_count(frames_dir)
    if frame_count == 0:
        raise ValueError("No frames extracted from video")
    fps = get_video_fps(video_path)
    swapper = FaceSwapper(swap_model="inswapper", det_size=(640, 640))
    frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
    for frame_path in frame_files:
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue
        bbox = swapper.get_primary_face_bbox(frame_bgr)
        if bbox is not None:
            frame_bgr = apply_cloth_color_change(
                frame_bgr, bbox, target_color, strength=strength
            )
        cv2.imwrite(str(frame_path), frame_bgr)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    merge_frames_to_video(
        frames_dir, output_path, fps=fps, audio_path=audio_path, input_format="png"
    )
    if temp_dir is None and ("tmp" in frames_dir or "temp" in frames_dir):
        try:
            shutil.rmtree(frames_dir, ignore_errors=True)
        except Exception:
            pass
    return output_path
