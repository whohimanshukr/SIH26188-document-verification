'''Image metadata forensics for detecting editing software and extracting EXIF data.'''

from pathlib import Path
from typing import Any, TypedDict
from PIL import ExifTags, Image, UnidentifiedImageError


class MetadataAnalysisResult(TypedDict):
    '''Normalized result of image metadata forensics.'''

    format: str | None
    width: int | None
    height: int | None
    metadata_present: bool
    software: str | None
    camera_make: str | None
    camera_model: str | None
    datetime: str | None
    suspicious: bool
    signals: list[str]


class MetadataAnalyzer:
    '''Analyze document images for EXIF metadata and image-editing software signatures.'''

    # Obvious editing software signatures that warrant suspicion.
    # Note: benign document processing software or absence of EXIF does not indicate forgery.
    _EDITING_KEYWORDS: tuple[str, ...] = (
        "photoshop",
        "gimp",
        "paint.net",
        "adobe",
    )

    def analyze(self, image_path: str | Path) -> MetadataAnalysisResult:
        '''Extract metadata from an image file and check for editing signatures.

        Args:
            image_path: Path to the image file.

        Returns:
            MetadataAnalysisResult dictionary with extracted properties and signals.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the path is not a file or Pillow cannot decode it.
        '''
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Image path is not a file: {path}")

        try:
            with Image.open(path) as img:
                img_format = img.format
                width, height = img.size

                # Extract raw EXIF if present.
                raw_exif = img.getexif()
                exif_dict: dict[str, Any] = {}

                if raw_exif:
                    for tag_id, value in raw_exif.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        exif_dict[tag_name] = value

        except UnidentifiedImageError as err:
            raise ValueError(f"Cannot identify image file (corrupted or unsupported format): {path}") from err
        except Exception as err:
            raise ValueError(f"Failed to read image file: {path}. Error: {err}") from err

        # Determine EXIF presence: whether any recognized tag was extracted
        metadata_present = bool(exif_dict)

        # helper to convert tag value safely to a clean string
        def _get_str(key: str) -> str | None:
            val = exif_dict.get(key)
            if val is None:
                return None
            val_str = str(val).strip()
            return val_str if val_str else None

        software = _get_str("Software")
        camera_make = _get_str("Make")
        camera_model = _get_str("Model")
        dt = _get_str("DateTime")

        signals: list[str] = []
        suspicious = False

        # Check for known image editing software in EXIF Software tag
        if software:
            software_lower = software.lower()
            for kw in self._EDITING_KEYWORDS:
                if kw in software_lower:
                    signals.append(f"Image editing software detected: {software}")
                    suspicious = True
                    break

        return {
            "format": img_format,
            "width": width,
            "height": height,
            "metadata_present": metadata_present,
            "software": software,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "datetime": dt,
            "suspicious": suspicious,
            "signals": signals,
        }
