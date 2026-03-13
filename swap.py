#!/usr/bin/env python3
"""
UltraFaceswap CLI - Swap face from photo onto video.

Usage:
    python swap.py --source photo.jpg --target video.mp4 --output out.mp4
    python swap.py --source photo.jpg --target video.mp4 --output-dir results --test-all
"""

import argparse
import itertools
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
from backend.core.interpolator import interpolate_frames
from backend.core.merger import merge_frames_to_video, get_video_fps as _get_fps
from backend.core.hair import apply_hair_color_matching


def settings_suffix(swap_model: str, det_size: int, upscale: int, interpolate: int, enhance: bool, hair_match: bool) -> str:
    """Short settings string for filenames: inswapper_d640_u1_i1_enh0_hair1"""
    return f"{swap_model}_d{det_size}_u{upscale}_i{interpolate}_enh{1 if enhance else 0}_hair{1 if hair_match else 0}"


def settings_summary(swap_model: str, det_size: int, upscale: int, interpolate: int, enhance: bool, hair_match: bool) -> str:
    """Human-readable settings summary."""
    parts = [
        f"model={swap_model}",
        f"det={det_size}",
        f"upscale={upscale}x",
        f"interpolate={interpolate}x",
        f"enhance={'on' if enhance else 'off'}",
        f"hair={'on' if hair_match else 'off'}",
    ]
    return " · ".join(parts)


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
    interpolate: int = 1,
    hair_match: bool = True,
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
    source_bbox = swapper.get_primary_face_bbox(source_bgr)

    # Process each frame
    frame_files = sorted(Path(frames_dir).glob("frame_*.png"))
    for frame_path in tqdm(frame_files, desc="Face swap"):
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
            print(f"Warning: Frame {frame_path.name} swap failed: {e}")
            traceback.print_exc()
            cv2.imwrite(str(frame_path), frame_bgr)

    output_fps = fps
    if interpolate in (2, 4):
        interpolate_frames(frames_dir, factor=interpolate, input_format="png")
        output_fps = fps * interpolate

    print("Merging video...")
    merge_frames_to_video(
        frames_dir,
        output_path,
        fps=output_fps,
        audio_path=audio_path,
        input_format="png",
    )

    # Cleanup temp if we created it
    if temp_dir is None and (frames_dir.startswith("/tmp") or "tmp" in frames_dir):
        import shutil
        try:
            shutil.rmtree(frames_dir)
        except OSError:
            pass

    summary = settings_summary(swap_model, det_size, upscale, interpolate, use_enhancer, hair_match)
    print(f"Done. Output saved to {output_path}")
    print(f"Settings used: {summary}")


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
        "--interpolate",
        type=int,
        choices=[1, 2, 4],
        default=1,
        help="Motion smoothing 1x, 2x, or 4x",
    )
    parser.add_argument(
        "--no-hair-match",
        action="store_true",
        help="Disable hair color matching",
    )
    parser.add_argument(
        "--temp-dir",
        help="Directory for temporary frames (default: system temp)",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Run all setting combinations and save each to a separate file (use with --output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for test-all mode (default: current dir)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source image not found: {args.source}")
        sys.exit(1)
    if not os.path.exists(args.target):
        print(f"Error: Target video not found: {args.target}")
        sys.exit(1)

    if args.test_all:
        # All combinations: swap_model, det_size, enhance, hair_match (16 total)
        # Use upscale=1, interpolate=1 for speed
        out_dir = Path(args.output_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        grid = list(itertools.product(
            ["inswapper", "simswap"],
            [320, 640],
            [False, True],   # enhance
            [True, False],   # hair_match
        ))
        print(f"Running {len(grid)} setting combinations...")
        for i, (swap_model, det_size, enhance, hair_match) in enumerate(grid):
            suffix = settings_suffix(swap_model, det_size, 1, 1, enhance, hair_match)
            output_path = str(out_dir / f"out_{suffix}.mp4")
            print(f"\n[{i+1}/{len(grid)}] {settings_summary(swap_model, det_size, 1, 1, enhance, hair_match)}")
            run_swap(
                args.source,
                args.target,
                output_path,
                use_enhancer=enhance,
                swap_model=swap_model,
                det_size=det_size,
                upscale=1,
                interpolate=1,
                hair_match=hair_match,
                temp_dir=args.temp_dir,
            )
        print(f"\nAll {len(grid)} outputs saved to {out_dir.absolute()}")
        return

    run_swap(
        args.source,
        args.target,
        args.output,
        use_enhancer=args.enhance,
        swap_model=args.swap_model,
        det_size=args.det_size,
        upscale=args.upscale,
        interpolate=args.interpolate,
        hair_match=not args.no_hair_match,
        temp_dir=args.temp_dir,
    )


if __name__ == "__main__":
    main()
