#!/usr/bin/env python3
"""
Run all Pro face-swap combinations using a source image and target video.
Usage:
  python scripts/run_pro_combinations.py [--source photo.png] [--target video4.mp4] [--api http://localhost:8000]
  python scripts/run_pro_combinations.py   # uses photo.png + video4.mp4, API at localhost:8000
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

# Project root
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "photo.png"
DEFAULT_TARGET = ROOT / "video4.mp4"
OUTPUT_DIR = ROOT / "pro_test_results"

# All Pro combinations to test
PIXEL_BOOSTS = ["256", "512", "768"]
FACE_DETECTORS = ["", "retinaface", "scrfd", "yoloface", "many"]  # "" = default
FACE_MASK_BLURS = [0.3, 0.35]
# Optional: reduce to fewer combos for a quick test (e.g. 3x3x2 = 18)
# PIXEL_BOOSTS = ["256", "512", "768"]
# FACE_DETECTORS = ["", "retinaface", "scrfd"]
# FACE_MASK_BLURS = [0.35]


def run_one(api_base: str, source_path: Path, target_path: Path, combo: dict) -> tuple[str, str]:
    """
    Submit one Pro job, poll until done, download result.
    Returns (job_id, output_path) or raises on failure.
    """
    url = f"{api_base.rstrip('/')}/api/swap-pro"
    with open(source_path, "rb") as f0, open(target_path, "rb") as f1:
        files = {"source": (source_path.name, f0, "image/png"), "target": (target_path.name, f1, "video/mp4")}
        data = {
            "facefusion_model": combo.get("facefusion_model", "inswapper_128_fp16"),
            "facefusion_pixel_boost": combo["facefusion_pixel_boost"],
            "facefusion_face_enhancer": "false",
            "facefusion_face_enhancer_blend": "0.5",
            "facefusion_lip_sync": "false",
        }
        if combo.get("face_detector_model"):
            data["face_detector_model"] = combo["face_detector_model"]
        if combo.get("face_detector_size"):
            data["face_detector_size"] = combo["face_detector_size"]
        if combo.get("face_mask_blur") is not None:
            data["face_mask_blur"] = str(combo["face_mask_blur"])

        r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    body = r.json()
    job_id = body["job_id"]

    status_url = f"{api_base.rstrip('/')}/api/status/{job_id}"
    result_url = f"{api_base.rstrip('/')}/api/result/{job_id}"
    while True:
        s = requests.get(status_url, timeout=10).json()
        st = s.get("status")
        if st == "completed":
            break
        if st == "failed":
            raise RuntimeError(f"Job {job_id} failed: {s.get('error', 'unknown')}")
        time.sleep(5)

    out_name = (
        f"pro_p{combo['facefusion_pixel_boost']}_det{combo.get('face_detector_model') or 'default'}_blur{combo.get('face_mask_blur', '')}.mp4"
    )
    out_path = OUTPUT_DIR / out_name
    resp = requests.get(result_url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return job_id, str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Run all Pro face-swap combinations")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source face image")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Target video")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--quick", action="store_true", help="Fewer combinations (3 pixel x 3 detector x 1 blur)")
    args = parser.parse_args()

    if not args.source.is_file():
        print(f"Source not found: {args.source}")
        sys.exit(1)
    if not args.target.is_file():
        print(f"Target not found: {args.target}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Source: {args.source}, Target: {args.target}, API: {args.api}")

    pixel_boosts = PIXEL_BOOSTS
    detectors = FACE_DETECTORS
    blurs = FACE_MASK_BLURS
    if args.quick:
        pixel_boosts = ["256", "512", "768"]
        detectors = ["", "retinaface", "scrfd"]
        blurs = [0.35]

    combinations = [
        {
            "facefusion_pixel_boost": p,
            "face_detector_model": d or None,
            "face_mask_blur": b,
        }
        for p in pixel_boosts
        for d in detectors
        for b in blurs
    ]
    # Normalize empty detector to missing key for API
    for c in combinations:
        if c.get("face_detector_model") == "" or c.get("face_detector_model") is None:
            c.pop("face_detector_model", None)
        if c.get("face_detector_model") is not None and c["face_detector_model"] == "":
            c.pop("face_detector_model", None)

    print(f"Running {len(combinations)} combinations...")
    failed = []
    for i, combo in enumerate(combinations):
        label = f"p{combo['facefusion_pixel_boost']}_det{combo.get('face_detector_model') or 'default'}_blur{combo.get('face_mask_blur')}"
        print(f"[{i+1}/{len(combinations)}] {label} ... ", end="", flush=True)
        try:
            jid, out = run_one(args.api, args.source, args.target, combo)
            print(f"ok -> {out}")
        except Exception as e:
            print(f"FAIL: {e}")
            failed.append((label, str(e)))

    if failed:
        print("\nFailed:", len(failed))
        for label, err in failed:
            print(f"  {label}: {err}")
        sys.exit(1)
    print("\nAll combinations completed.")


if __name__ == "__main__":
    main()
