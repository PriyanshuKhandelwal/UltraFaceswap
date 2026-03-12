"""Face swap using InsightFace and InSwapper model."""

import copy
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image


# Model URLs and paths (Hugging Face mirror - GitHub release URL often fails)
INSWAPPER_MODEL_NAME = "inswapper_128.onnx"
INSWAPPER_MODEL_URL = (
    "https://huggingface.co/crw-dev/Deepinsightinswapper/resolve/"
    "26470fec58658f1abefe9fb8ae0cfd3f804701dd/inswapper_128.onnx"
)
BUFFALO_L = "buffalo_l"


def get_models_dir() -> str:
    """Get or create models directory."""
    models_dir = os.environ.get(
        "ULTRAFACESWAP_MODELS",
        str(Path(__file__).parent.parent.parent / "models"),
    )
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def ensure_inswapper_model() -> str:
    """Ensure inswapper model exists, download if needed."""
    import urllib.request

    models_dir = get_models_dir()
    model_path = os.path.join(models_dir, INSWAPPER_MODEL_NAME)

    if not os.path.exists(model_path):
        print(f"Downloading InSwapper model to {model_path}...")
        try:
            urllib.request.urlretrieve(INSWAPPER_MODEL_URL, model_path)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download InSwapper model. "
                f"Download manually from {INSWAPPER_MODEL_URL} to {model_path}"
            ) from e

    return model_path


class FaceSwapper:
    """Face swap engine using InsightFace InSwapper."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize face swapper.

        Args:
            model_path: Path to inswapper_128.onnx (auto-downloaded if None)
            providers: ONNX runtime providers (default: CUDA, CPU)
        """
        import onnxruntime
        import insightface
        from insightface.app import FaceAnalysis

        if model_path is None:
            model_path = ensure_inswapper_model()

        if providers is None:
            providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" not in providers and "CPUExecutionProvider" in providers:
                providers = ["CPUExecutionProvider"]

        models_dir = get_models_dir()

        self.face_analysis = FaceAnalysis(
            name=BUFFALO_L,
            root=models_dir,
            providers=providers,
        )
        self.face_analysis.prepare(ctx_id=0, det_size=(320, 320))

        self.face_swapper = insightface.model_zoo.get_model(model_path, providers=providers)

    def _get_faces(self, img_bgr: np.ndarray, many: bool = True) -> List:
        """Detect faces in BGR image."""
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
        """
        Swap one face from source onto target.

        Args:
            source_img: BGR numpy array (source face image)
            target_img: BGR numpy array (target image/video frame)
            source_index: Index of face in source (0 = leftmost)
            target_index: Index of face in target to replace (0 = leftmost)

        Returns:
            BGR numpy array with swapped face
        """
        source_faces = self._get_faces(source_img, many=True)
        target_faces = self._get_faces(target_img, many=True)

        if not source_faces:
            raise ValueError("No face found in source image")
        if not target_faces:
            return target_img  # No target face, return unchanged

        source_face = source_faces[min(source_index, len(source_faces) - 1)]
        target_face = target_faces[min(target_index, len(target_faces) - 1)]

        result = self.face_swapper.get(
            target_img, target_face, source_face, paste_back=True
        )
        return result

    def swap_all_faces(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
    ) -> np.ndarray:
        """
        Replace all faces in target with the first face from source.

        Args:
            source_img: BGR numpy array (source face image)
            target_img: BGR numpy array (target frame)

        Returns:
            BGR numpy array with all faces swapped
        """
        source_faces = self._get_faces(source_img, many=True)
        target_faces = self._get_faces(target_img, many=True)

        if not source_faces:
            raise ValueError("No face found in source image")
        if not target_faces:
            return target_img

        result = copy.deepcopy(target_img)
        source_face = source_faces[0]

        for target_face in target_faces:
            result = self.face_swapper.get(
                result, target_face, source_face, paste_back=True
            )

        return result

    def process_frame(
        self,
        source_bgr: np.ndarray,
        frame_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Process a single video frame: swap all target faces with source face.

        Args:
            source_bgr: Source face image (BGR)
            frame_bgr: Video frame (BGR)

        Returns:
            Processed frame (BGR)
        """
        return self.swap_all_faces(source_bgr, frame_bgr)


def load_source_face(source_path: str) -> np.ndarray:
    """Load source image as BGR numpy array."""
    img = Image.open(source_path)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
