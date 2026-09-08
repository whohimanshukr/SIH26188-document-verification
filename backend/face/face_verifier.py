"""Face verification module using OpenCV YuNet detector and SFace recognizer."""

from pathlib import Path
from typing import Any, TypedDict
import cv2
import numpy as np


class FaceVerificationResult(TypedDict):
    """Normalized output of face verification."""

    match: bool
    similarity_score: float
    document_face_detected: bool
    selfie_face_detected: bool
    signals: list[str]


class FaceVerifier:
    """Verify facial identity between a document image and a reference/selfie image."""

    DEFAULT_DETECTION_MODEL = "face_detection_yunet_2023mar.onnx"
    DEFAULT_RECOGNITION_MODEL = "face_recognition_sface_2021dec.onnx"

    def __init__(
        self,
        threshold: float = 0.363,
        models_dir: str | Path | None = None,
    ) -> None:
        """Initialize FaceVerifier with YuNet and SFace models.

        Args:
            threshold: Cosine similarity cutoff for a positive match (default 0.363 for SFace).
            models_dir: Directory containing YuNet and SFace ONNX models.

        Raises:
            FileNotFoundError: If either required ONNX model file is missing.
        """
        self.threshold = threshold

        if models_dir is None:
            models_dir = Path(__file__).resolve().parent / "models"
        else:
            models_dir = Path(models_dir)

        self.models_dir = models_dir
        self.det_model_path = models_dir / self.DEFAULT_DETECTION_MODEL
        self.rec_model_path = models_dir / self.DEFAULT_RECOGNITION_MODEL

        # Validate that model files are present on disk; do NOT download automatically.
        if not self.det_model_path.is_file():
            raise FileNotFoundError(
                f"Face detection model missing: '{self.det_model_path}'. "
                f"Please place '{self.DEFAULT_DETECTION_MODEL}' in '{self.models_dir}'."
            )
        if not self.rec_model_path.is_file():
            raise FileNotFoundError(
                f"Face recognition model missing: '{self.rec_model_path}'. "
                f"Please place '{self.DEFAULT_RECOGNITION_MODEL}' in '{self.models_dir}'."
            )

        # Initialize YuNet detector.
        # input_size is dynamic and will be updated per image using setInputSize().
        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.det_model_path),
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

        # Initialize SFace recognizer for feature extraction and matching.
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=str(self.rec_model_path),
            config="",
        )

    def verify(
        self,
        document_image_path: str | Path,
        selfie_image_path: str | Path,
    ) -> FaceVerificationResult:
        """Compare face in document image against face in reference/selfie image.

        Args:
            document_image_path: Path to ID/passport image containing a portrait face.
            selfie_image_path: Path to user's reference/selfie image.

        Returns:
            FaceVerificationResult with match flag, similarity score, detection flags, and signals.

        Raises:
            FileNotFoundError: If either image file does not exist.
            ValueError: If either path is not a file or image cannot be decoded.
        """
        doc_path = Path(document_image_path)
        selfie_path = Path(selfie_image_path)

        if not doc_path.exists():
            raise FileNotFoundError(f"Document image not found: {doc_path}")
        if not doc_path.is_file():
            raise ValueError(f"Document image path is not a file: {doc_path}")

        if not selfie_path.exists():
            raise FileNotFoundError(f"Selfie image not found: {selfie_path}")
        if not selfie_path.is_file():
            raise ValueError(f"Selfie image path is not a file: {selfie_path}")

        # Load images with OpenCV
        doc_bgr = self._load_image(doc_path)
        selfie_bgr = self._load_image(selfie_path)

        # Detect faces in both images
        doc_face = self._detect_face(doc_bgr)
        selfie_face = self._detect_face(selfie_bgr)

        doc_detected = doc_face is not None
        selfie_detected = selfie_face is not None
        signals: list[str] = []

        if not doc_detected and not selfie_detected:
            return {
                "match": False,
                "similarity_score": 0.0,
                "document_face_detected": False,
                "selfie_face_detected": False,
                "signals": ["No face detected in document image or selfie image"],
            }

        if not doc_detected:
            return {
                "match": False,
                "similarity_score": 0.0,
                "document_face_detected": False,
                "selfie_face_detected": selfie_detected,
                "signals": ["Face could not be detected in the document image"],
            }

        if not selfie_detected:
            return {
                "match": False,
                "similarity_score": 0.0,
                "document_face_detected": doc_detected,
                "selfie_face_detected": False,
                "signals": ["Face could not be detected in the selfie image"],
            }

        # Both faces detected: align, crop, and extract 128-d SFace feature vectors
        doc_aligned = self.recognizer.alignCrop(doc_bgr, doc_face)
        selfie_aligned = self.recognizer.alignCrop(selfie_bgr, selfie_face)

        doc_feature = self.recognizer.feature(doc_aligned)
        selfie_feature = self.recognizer.feature(selfie_aligned)

        # Compare using cosine similarity (FaceRecognizerSF_FR_COSINE)
        raw_cosine: float = float(
            self.recognizer.match(
                doc_feature, selfie_feature, cv2.FaceRecognizerSF_FR_COSINE
            )
        )

        # Clamp similarity score to [0.0, 1.0] range
        similarity_score = max(0.0, min(1.0, raw_cosine))
        is_match = similarity_score >= self.threshold

        if is_match:
            signals.append(
                f"Faces match successfully (similarity: {similarity_score:.3f} >= {self.threshold:.3f})"
            )
        else:
            signals.append(
                f"Faces do not match (similarity: {similarity_score:.3f} < {self.threshold:.3f})"
            )

        return {
            "match": is_match,
            "similarity_score": round(similarity_score, 4),
            "document_face_detected": True,
            "selfie_face_detected": True,
            "signals": signals,
        }

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as standard 3-channel BGR array."""
        # cv2.imdecode with np.fromfile supports non-ASCII paths on Windows reliably
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"OpenCV failed to decode image: {path}")
        return img

    def _detect_face(self, bgr_image: np.ndarray) -> np.ndarray | None:
        """Detect face in BGR image using YuNet and return the highest-confidence face."""
        h, w = bgr_image.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(bgr_image)
        if faces is None or len(faces) == 0:
            return None

        # If multiple faces are detected, choose the face with the highest detection score (column index 14)
        # to focus on the primary subject/document portrait rather than background faces.
        if len(faces) == 1:
            return faces[0]

        best_face_index = int(np.argmax(faces[:, 14]))
        return faces[best_face_index]

