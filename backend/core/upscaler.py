"""Real-ESRGAN upscaling for improved output quality."""

import os
import numpy as np

_UPSCALERS = {}  # scale -> RealESRGANer


def upscale_image(img_bgr: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Upscale image using Real-ESRGAN.

    Args:
        img_bgr: BGR numpy array
        scale: 2 or 4 (2x or 4x upscaling)

    Returns:
        Upscaled BGR image, or original if upscaling unavailable
    """
    if scale not in (2, 4):
        return img_bgr

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
    except ImportError:
        return img_bgr

    models_dir = os.environ.get(
        "ULTRAFACESWAP_MODELS",
        os.path.join(os.path.dirname(__file__), "..", "..", "models"),
    )

    if scale == 2:
        model_name = "RealESRGAN_x2plus.pth"
        arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        model_name = "RealESRGAN_x4plus.pth"
        arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        return img_bgr

    if scale not in _UPSCALERS:
        try:
            _UPSCALERS[scale] = RealESRGANer(scale=scale, model_path=model_path, model=arch)
        except Exception:
            return img_bgr

    try:
        output, _ = _UPSCALERS[scale].enhance(img_bgr, outscale=scale)
        return output if output is not None else img_bgr
    except Exception:
        return img_bgr
