"""Face restoration/enhancement using GFPGAN (optional)."""

import os
import numpy as np

_RESTORER = None


def enhance_face(
    img_bgr: np.ndarray,
    use_gfpgan: bool = True,
) -> np.ndarray:
    """
    Enhance/restore face in image using GFPGAN when available.

    When GFPGAN is not installed or model is missing, returns image unchanged.

    Args:
        img_bgr: BGR numpy array
        use_gfpgan: If True, attempt GFPGAN enhancement

    Returns:
        Enhanced BGR image (or original if GFPGAN unavailable)
    """
    global _RESTORER

    global _RESTORER

    if not use_gfpgan:
        return img_bgr

    try:
        from gfpgan import GFPGANer
    except ImportError:
        return img_bgr

    models_dir = os.environ.get(
        "ULTRAFACESWAP_MODELS",
        os.path.join(os.path.dirname(__file__), "..", "..", "models"),
    )
    model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
    if not os.path.exists(model_path):
        return img_bgr

    if _RESTORER is None:
        try:
            _RESTORER = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
            )
        except Exception:
            return img_bgr

    try:
        _, _, restored = _RESTORER.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        return restored if restored is not None else img_bgr
    except Exception:
        return img_bgr
