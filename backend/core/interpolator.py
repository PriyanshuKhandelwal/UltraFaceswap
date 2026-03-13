"""Frame interpolation for smoother video output (temporal smoothing)."""

import os
from pathlib import Path
from typing import List

import cv2
import numpy as np


def interpolate_frames(
    frames_dir: str,
    factor: int = 2,
    input_format: str = "png",
) -> str:
    """
    Insert interpolated frames between consecutive frames for smoother motion.

    Uses linear blend between adjacent frames. Factor 2 doubles the frame count.
    Factor 4 inserts 3 blended frames between each pair (4x total).

    Args:
        frames_dir: Directory with frame_000001.png, frame_000002.png, ...
        factor: 2 or 4 (2x or 4x frame count)
        input_format: png or jpg

    Returns:
        frames_dir (frames are updated in place, then renumbered)
    """
    if factor not in (2, 4):
        return frames_dir

    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob(f"frame_*.{input_format}"))
    if len(frame_files) < 2:
        return str(frames_dir)

    # Load all frames
    frames = []
    for fp in frame_files:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(img)

    if len(frames) < 2:
        return str(frames_dir)

    # Generate interpolated frames
    new_frames = []
    for i in range(len(frames) - 1):
        f1 = frames[i]
        f2 = frames[i + 1]
        new_frames.append(f1)
        for k in range(1, factor):
            alpha = k / factor
            blended = (f1.astype(np.float32) * (1 - alpha) + f2.astype(np.float32) * alpha).astype(np.uint8)
            new_frames.append(blended)
    new_frames.append(frames[-1])

    # Write back with new numbering
    for fp in frame_files:
        try:
            os.remove(fp)
        except OSError:
            pass

    for i, img in enumerate(new_frames):
        out_path = frames_dir / f"frame_{i + 1:06d}.{input_format}"
        cv2.imwrite(str(out_path), img)

    return str(frames_dir)
