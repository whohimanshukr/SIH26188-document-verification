"""Reusable OCR engine for document images."""

from pathlib import Path
from typing import Any, TypedDict

from paddleocr import PaddleOCR


class OCRItem(TypedDict):
    """One detected text item and its optional metadata."""

    text: str
    confidence: float | None
    box: list[Any] | None


class OCRResult(TypedDict):
    """Project-owned normalized OCR output."""

    text: str
    items: list[OCRItem]


class OCREngine:
    """Run document OCR with the verified CPU PaddleOCR configuration."""

    def __init__(self) -> None:
        # Explicit mobile models keep this first production path small and CPU-friendly.
        # MKL-DNN is disabled because the verified Windows CPU setup failed in its oneDNN path.
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            device="cpu",
            engine="paddle",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def extract_text(self, image_path: str | Path) -> OCRResult:
        """Extract and normalize text from an existing local image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Image path is not a file: {path}")

        # PaddleOCR is initialized once in __init__ so model loading is not repeated per image.
        results = list(self.ocr.predict(input=str(path)))
        if not results:
            return {"text": "", "items": []}

        result = results[0]
        texts = _as_list(result.get("rec_texts", []))
        scores = _as_list(result.get("rec_scores", []))
        boxes = _as_list(result.get("rec_boxes", result.get("dt_polys", [])))

        items: list[OCRItem] = []
        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()
            if not text:
                continue
            raw_score = scores[index] if index < len(scores) else None
            raw_box = boxes[index] if index < len(boxes) else None
            items.append(
                {
                    "text": text,
                    "confidence": float(raw_score) if raw_score is not None else None,
                    "box": _as_list(raw_box) if raw_box is not None else None,
                }
            )

        return {
            "text": "\n".join(item["text"] for item in items),
            "items": items,
        }


def _as_list(value: Any) -> list[Any]:
    """Convert PaddleOCR or NumPy containers into ordinary Python lists."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]