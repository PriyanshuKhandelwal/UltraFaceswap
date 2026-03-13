"""Hair color matching - transfer source hair color to swapped frames."""

import numpy as np
import cv2
from typing import Optional, Tuple


def _get_hair_region_mask(
    img_shape: Tuple[int, int],
    face_bbox: Tuple[int, int, int, int],
    expand_above: float = 1.2,
    expand_sides: float = 0.15,
) -> np.ndarray:
    """Create a soft mask for the hair region above the face.

    Args:
        img_shape: (height, width) of image
        face_bbox: (x1, y1, x2, y2) face bounding box
        expand_above: How much above face (in face heights) to include
        expand_sides: Horizontal expansion as fraction of face width

    Returns:
        Float mask [0, 1] of shape (H, W)
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = face_bbox
    fw = x2 - x1
    fh = y2 - y1

    # Hair region: above and slightly overlapping top of face
    hair_h = int(fh * expand_above)
    hair_y1 = max(0, y1 - hair_h)
    hair_y2 = y1 + int(fh * 0.15)
    pad_x = int(fw * expand_sides)
    hair_x1 = max(0, x1 - pad_x)
    hair_x2 = min(w, x2 + pad_x)

    mask = np.zeros((h, w), dtype=np.float32)
    if hair_y2 <= hair_y1:
        return mask

    # Elliptical mask for soft blending (hair is typically oval above face)
    sub_h = hair_y2 - hair_y1
    sub_w = hair_x2 - hair_x1
    if sub_h < 2 or sub_w < 2:
        return mask

    ell_center = (sub_w // 2, sub_h // 2)
    ell_axes = (sub_w // 2, sub_h // 2)
    ell_mask = np.zeros((sub_h, sub_w), dtype=np.float32)
    cv2.ellipse(ell_mask, ell_center, ell_axes, 0, 0, 360, 1.0, -1)
    # Soften edges
    ell_mask = cv2.GaussianBlur(ell_mask, (max(3, sub_w // 4 | 1), max(3, sub_h // 4 | 1)), 0)
    mask[hair_y1:hair_y2, hair_x1:hair_x2] = ell_mask
    return mask


def _extract_mean_lab(img_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Extract mean LAB color from masked region."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    valid = mask > 0.1
    if not np.any(valid):
        return None
    pixels = lab[valid]
    mean_lab = np.mean(pixels, axis=0)
    return mean_lab.astype(np.float32)


def _recolor_region_lab(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    target_mean_lab: np.ndarray,
    strength: float = 0.6,
) -> np.ndarray:
    """Shift color of masked region toward target LAB mean.

    Uses simple mean/std matching in LAB space for natural recoloring.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    valid = mask > 0.01
    if not np.any(valid):
        return img_bgr
    pixels = lab[valid]
    src_mean = np.mean(pixels, axis=0)
    src_std = np.std(pixels, axis=0) + 1e-6
    # Scale by source std to preserve local variation
    target_std = src_std
    shift = (target_mean_lab - src_mean) * strength
    mask_3d = np.stack([mask] * 3, axis=-1)
    lab_out = lab + mask_3d * shift
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


def apply_hair_color_matching(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    source_bbox: Tuple[int, int, int, int],
    target_bbox: Tuple[int, int, int, int],
    strength: float = 0.55,
) -> np.ndarray:
    """
    Transfer hair color from source to target image.

    Uses face bboxes to define hair regions, extracts mean LAB from source
    and recolors target's hair region.

    Args:
        source_bgr: Source face image (BGR)
        target_bgr: Target/swapped frame (BGR)
        source_bbox: Face bbox (x1,y1,x2,y2) in source
        target_bbox: Face bbox (x1,y1,x2,y2) in target
        strength: Blending strength 0..1 (0.5–0.7 typical)

    Returns:
        Target image with source hair color applied
    """
    src_mask = _get_hair_region_mask(source_bgr.shape, source_bbox)
    src_lab = _extract_mean_lab(source_bgr, src_mask)
    if src_lab is None:
        return target_bgr
    tgt_mask = _get_hair_region_mask(target_bgr.shape, target_bbox)
    return _recolor_region_lab(target_bgr, tgt_mask, src_lab, strength=strength)
