#!/usr/bin/env python3
"""
UltraFaceswap CLI - Swap face from photo onto video.

Usage:
    python swap.py --source photo.jpg --target video.mp4 --output out.mp4
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
from tqdm import tqdm

from backend.core.extractor import extract_frames, get_frame_count
from backend.core.face_swap import FaceSwapper, load_source_face
from backend.core.enhancer import enhance_face
from backend.core.upscaler import upscale_image
from backend.core.merger import merge_frames_to_video, get_video_fps as _get_fps


def get_video_fps(video_path: str) -> float:
    """Get video FPS for merging."""
    try:
        return _get_fps(video_path)
    except Exception:
        return 30.0


def run_swap(
    source_path: str,
    target_path: str,
    output_path: str,
    *,
    use_enhancer: bool = False,
    swap_model: str = "inswapper",
    det_size: int = 640,
    upscale: int = 1,
    temp_dir: Optional[str] = None,
):
    """
    Run full face swap pipeline.

    Args:
        source_path: Path to source face photo
        target_path: Path to target video
        output_path: Path for output video
        use_enhancer: Use GFPGAN for face restoration
        temp_dir: Directory for temp frames (default: system temp)
    """
    print("Extracting frames...")
    frames_dir, audio_path = extract_frames(
        target_path,
        output_dir=temp_dir,
        image_format="png",
    )

    frame_count = get_frame_count(frames_dir)
    if frame_count == 0:
        raise ValueError("No frames extracted from video")

    print(f"Processing {frame_count} frames...")
    fps = get_video_fps(target_path)

    # Load source face
    source_bgr = load_source_face(source_path)

    # Initialize face swapper
    swapper = FaceSwapper(
        swap_model=swap_model,
        det_size=(det_size, det_size),
    )

    # Process each frame
    frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
    for frame_path in tqdm(frame_files, desc="Face swap"):
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        try:
            result = swapper.process_frame(source_bgr, frame_bgr)
            if use_enhancer:
                result = enhance_face(result, use_gfpgan=True)
            if upscale in (2, 4):
                result = upscale_image(result, scale=upscale)
            cv2.imwrite(str(frame_path), result)
        except Exception as e:
            # Log but continue - some frames may have no faces
            print(f"Warning: Frame {frame_path.name}: {e}")

    print("Merging video...")
    merge_frames_to_video(
        frames_dir,
        output_path,
        fps=fps,
        audio_path=audio_path,
        input_format="png",
    )

    # Cleanup temp if we created it
    if temp_dir is None and frames_dir.startswith("/tmp") or "tmp" in frames_dir:
        import shutil
        try:
            shutil.rmtree(frames_dir)
        except OSError:
            pass

    print(f"Done. Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="UltraFaceswap - Swap face from photo onto video"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Path to source face photo",
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Path to target video",
    )
    parser.add_argument(
        "--output", "-o",
        default="output.mp4",
        help="Output video path (default: output.mp4)",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Use GFPGAN for face restoration (optional)",
    )
    parser.add_argument(
        "--swap-model",
        choices=["inswapper", "simswap"],
        default="inswapper",
        help="InSwapper (faster) or SimSwap (sharper)",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        choices=[320, 640],
        default=640,
        help="Face detection size (640 for HD)",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="Upscale factor 1x, 2x, or 4x",
    )
    parser.add_argument(
        "--temp-dir",
        help="Directory for temporary frames (default: system temp)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source image not found: {args.source}")
        sys.exit(1)
    if not os.path.exists(args.target):
        print(f"Error: Target video not found: {args.target}")
        sys.exit(1)

    run_swap(
        args.source,
        args.target,
        args.output,
        use_enhancer=args.enhance,
        swap_model=args.swap_model,
        det_size=args.det_size,
        upscale=args.upscale,
        temp_dir=args.temp_dir,
    )


if __name__ == "__main__":
    main()
