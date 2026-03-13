"""Face swap using InsightFace, InSwapper, or SimSwap model."""

import copy
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# Model URLs
INSWAPPER_MODEL_NAME = "inswapper_128.onnx"
INSWAPPER_MODEL_URL = (
    "https://huggingface.co/crw-dev/Deepinsightinswapper/resolve/"
    "26470fec58658f1abefe9fb8ae0cfd3f804701dd/inswapper_128.onnx"
)
SIMSWAP_MODEL_NAME = "simswap_256.onnx"
SIMSWAP_MODEL_URL = (
    "https://huggingface.co/Patil/inswapper/resolve/"
    "c4dae4118487411d40639ad36bc842c30d1a8452/simswap_256.onnx"
)
BUFFALO_L = "buffalo_l"

# Face alignment utility (from insightface)
def get_models_dir() -> str:
    """Get or create models directory."""
    models_dir = os.environ.get(
        "ULTRAFACESWAP_MODELS",
        str(Path(__file__).parent.parent.parent / "models"),
    )
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _download_model(url: str, path: str, name: str) -> None:
    import urllib.request
    if os.path.exists(path):
        return
    print(f"Downloading {name} to {path}...")
    try:
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {name}. Download manually from {url} to {path}"
        ) from e


def ensure_inswapper_model() -> str:
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, INSWAPPER_MODEL_NAME)
    _download_model(INSWAPPER_MODEL_URL, model_path, "InSwapper")
    return model_path


def ensure_simswap_model() -> str:
    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, SIMSWAP_MODEL_NAME)
    _download_model(SIMSWAP_MODEL_URL, model_path, "SimSwap")
    return model_path


class FaceSwapper:
    """Face swap engine using InsightFace InSwapper or SimSwap."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
        swap_model: str = "inswapper",
        det_size: Tuple[int, int] = (640, 640),
    ):
        """
        Initialize face swapper.

        Args:
            model_path: Path to model file (auto-downloaded if None)
            providers: ONNX runtime providers
            swap_model: "inswapper" (128px, faster) or "simswap" (256px, higher quality)
            det_size: Face detection size (320,640) - larger = better detection on HD
        """
        import onnxruntime
        import insightface
        from insightface.app import FaceAnalysis

        if providers is None:
            providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" not in providers and "CPUExecutionProvider" in providers:
                providers = ["CPUExecutionProvider"]

        models_dir = get_models_dir()
        self.swap_model = swap_model
        self.det_size = det_size

        self.face_analysis = FaceAnalysis(
            name=BUFFALO_L,
            root=models_dir,
            providers=providers,
        )
        self.face_analysis.prepare(ctx_id=0, det_size=det_size)

        if swap_model == "simswap":
            if model_path is None:
                model_path = ensure_simswap_model()
            self.face_swapper = self._load_simswap(model_path, providers)
        else:
            if model_path is None:
                model_path = ensure_inswapper_model()
            self.face_swapper = insightface.model_zoo.get_model(model_path, providers=providers)

    def _load_simswap(self, model_path: str, providers: List[str]):
        """Load SimSwap ONNX (256px). model_zoo maps simswap->ArcFace (wrong API), use _SimSwap256 first."""
        import insightface
        try:
            return _SimSwap256(model_path, providers)
        except Exception:
            pass
        try:
            m = insightface.model_zoo.get_model(model_path, providers=providers)
            import inspect
            sig = inspect.signature(m.get)
            if "source_face" in str(sig) or "target_face" in str(sig):
                return m
        except Exception:
            pass
        ensure_inswapper_model()
        return insightface.model_zoo.get_model(
            os.path.join(get_models_dir(), INSWAPPER_MODEL_NAME),
            providers=providers,
        )

    def _call_swapper_get(self, img: np.ndarray, target_face, source_face) -> np.ndarray:
        """Call face_swapper.get() - compatible with both old/new InsightFace API (paste_back)."""
        try:
            return self.face_swapper.get(img, target_face, source_face, paste_back=True)
        except TypeError:
            return self.face_swapper.get(img, target_face, source_face)

    def get_primary_face_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box (x1, y1, x2, y2) of primary face, or None."""
        faces = self._get_faces(img_bgr, many=False)
        if not faces:
            return None
        bbox = faces[0].bbox
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

    def _get_faces(self, img_bgr: np.ndarray, many: bool = True) -> List:
        faces = self.face_analysis.get(img_bgr)
        if not faces:
            return []
        if many:
            return sorted(faces, key=lambda x: x.bbox[0])
        return [min(faces, key=lambda x: x.bbox[0])]

    def swap_face(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        source_index: int = 0,
        target_index: int = 0,
    ) -> np.ndarray:
        source_faces = self._get_faces(source_img, many=True)
        target_faces = self._get_faces(target_img, many=True)
        if not source_faces:
            raise ValueError("No face found in source image")
        if not target_faces:
            return target_img
        source_face = source_faces[min(source_index, len(source_faces) - 1)]
        target_face = target_faces[min(target_index, len(target_faces) - 1)]
        return self._call_swapper_get(target_img, target_face, source_face)

    def swap_all_faces(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
    ) -> np.ndarray:
        source_faces = self._get_faces(source_img, many=True)
        target_faces = self._get_faces(target_img, many=True)
        if not source_faces:
            raise ValueError("No face found in source image")
        if not target_faces:
            return target_img
        result = copy.deepcopy(target_img)
        source_face = source_faces[0]
        for target_face in target_faces:
            result = self._call_swapper_get(result, target_face, source_face)
        return result

    def process_frame(
        self,
        source_bgr: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> np.ndarray:
        return self.swap_all_faces(source_bgr, frame_bgr)


class _SimSwap256:
    """SimSwap 256 ONNX - InsightFace model_zoo may route 256 to ArcFace; this handles it."""

    def __init__(self, model_path: str, providers: List[str]):
        import onnxruntime
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )
        inp = self.session.get_inputs()
        self.input_names = [i.name for i in inp]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = 256
        from insightface.utils import face_align
        self.face_align = face_align

    def get(self, img: np.ndarray, target_face, source_face, paste_back: bool = True):
        aimg, M = self.face_align.norm_crop2(img, target_face.kps, self.input_size)
        blob = (aimg.astype(np.float32) / 255.0 - 0.5) / 0.5
        blob = blob.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        latent = source_face.normed_embedding.reshape(1, -1).astype(np.float32)
        out = self.session.run(
            self.output_names,
            {self.input_names[0]: blob, self.input_names[1]: latent},
        )[0]
        img_fake = out.transpose(0, 2, 3, 1)[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return bgr_fake, M
        IM = cv2.invertAffineTransform(M)
        bgr_fake = cv2.warpAffine(bgr_fake, IM, (img.shape[1], img.shape[0]), borderValue=0.0)
        mask = np.ones((self.input_size, self.input_size), dtype=np.float32)
        mask = cv2.warpAffine(mask, IM, (img.shape[1], img.shape[0]), borderValue=0.0)
        # Strengthen mask to avoid ghosting (warp interpolation can dilute values)
        mask = np.clip(mask * 1.3, 0, 1)
        mask = np.expand_dims(mask, 2)
        img_float = img.astype(np.float32)
        result = (mask * bgr_fake.astype(np.float32) + (1 - mask) * img_float).astype(np.uint8)
        return result


def load_source_face(source_path: str) -> np.ndarray:
    """Load source image as BGR numpy array."""
    img = Image.open(source_path)
    img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
