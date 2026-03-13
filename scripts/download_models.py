#!/usr/bin/env python3
"""Download required models for UltraFaceswap."""

import os
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "inswapper_128.onnx": (
        "https://huggingface.co/crw-dev/Deepinsightinswapper/resolve/26470fec58658f1abefe9fb8ae0cfd3f804701dd/inswapper_128.onnx"
    ),
    "simswap_256.onnx": (
        "https://huggingface.co/Patil/inswapper/resolve/"
        "c4dae4118487411d40639ad36bc842c30d1a8452/simswap_256.onnx"
    ),
    "GFPGANv1.4.pth": (
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    ),
    "RealESRGAN_x2plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    ),
    "RealESRGAN_x4plus.pth": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    ),
}


def get_models_dir() -> str:
    models_dir = os.environ.get(
        "ULTRAFACESWAP_MODELS",
        str(Path(__file__).parent.parent / "models"),
    )
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def download(name: str, url: str, models_dir: str) -> bool:
    path = os.path.join(models_dir, name)
    if os.path.exists(path):
        print(f"  {name} - already exists")
        return True
    print(f"  Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"  {name} - done")
        return True
    except Exception as e:
        print(f"  {name} - failed: {e}")
        return False


def main():
    models_dir = get_models_dir()
    print(f"Models directory: {models_dir}")
    for name, url in MODELS.items():
        download(name, url, models_dir)
    print("Done.")


if __name__ == "__main__":
    main()
