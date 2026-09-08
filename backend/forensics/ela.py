'''Error Level Analysis (ELA) forensics module for detecting digital image manipulation.'''

import io
from pathlib import Path
from typing import TypedDict
import numpy as np
from PIL import Image, UnidentifiedImageError


class ELAResult(TypedDict):
    '''Normalized result from Error Level Analysis.'''

    mean_error: float
    max_error: int
    suspicious_ratio: float
    suspicious: bool
    signals: list[str]


class ELAAnalyzer:
    '''Analyze compression artifacts via Error Level Analysis (ELA).

    ELA highlights differences in compression levels across an image.
    When an image is saved as a JPEG, each 8x8 block is compressed. If a portion
    of an image has been pasted or modified at a different time or compression level,
    it often shows a higher or noticeably different error upon recompression.
    '''

    def __init__(
        self,
        recompression_quality: int = 90,
        pixel_error_threshold: float = 20.0,
        suspicious_ratio_threshold: float = 0.08,
    ) -> None:
        '''Initialize ELA parameters.

        Args:
            recompression_quality: JPEG quality factor for the recompressed comparison image (default 90).
            pixel_error_threshold: Minimum per-pixel average RGB absolute error to count as high error (default 20.0).
            suspicious_ratio_threshold: Proportion of high-error pixels needed to flag suspicious (default 0.08 , 8%).
        '''
        self.recompression_quality = recompression_quality
        self.pixel_error_threshold = pixel_error_threshold
        self.suspicious_ratio_threshold = suspicious_ratio_threshold

    def analyze(self, image_path: str | Path) -> ELAResult:
        '''Perform ELA on a local document image.

        Args:
            image_path: Path to the target image file.

        Returns:
            ELAResult dictionary with numeric error statistics and forensic signals.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path is not a file or Pillow cannot decode it.
        '''
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Image path is not a file: {path}")

        try:
            with Image.open(path) as img:
                orig_rgb = img.convert("RGB")
        except UnidentifiedImageError as err:
            raise ValueError(f"Cannot identify image file: {path}") from err
        except Exception as err:
            raise ValueError(f"Failed to open image file: {path}. Error: {err}") from err

        # Recompress in memory using BytesIO to avoid saving temporary files on disk.
        # JPEG recompression introduces generation loss at a uniform rate for untouched images;
        # spliced or edited regions will usually exhibit higher disparity against the newly compressed base.
        buffer = io.BytesIO()
        orig_rgb.save(buffer, format="JPEG", quality=self.recompression_quality)
        buffer.seek(0)

        with Image.open(buffer) as recompressed_img:
            recompressed_rgb = recompressed_img.convert("RGB")

            # np float32 prevents underflow or overflow during absolute difference calculation
            orig_arr = np.asarray(orig_rgb, dtype=np.float32)
            recomp_arr = np.asarray(recompressed_rgb, dtype=np.float32)

        buffer.close()

        # Calculate absolute pixel difference across all color channels
        diff = np.abs(orig_arr - recomp_arr)

        # Mean and max error across all pixel values
        mean_error = float(np.mean(diff))
        max_error = int(np.max(diff))

        # Per-pixel mean RGB error
        pixel_errors = np.mean(diff, axis=2)

        # Fraction of pixels whose error exceeds the pixel_error_threshold
        high_error_pixels = np.count_nonzero(pixel_errors > self.pixel_error_threshold)
        total_pixels = pixel_errors.size
        suspicious_ratio = float(high_error_pixels / total_pixels) if total_pixels > 0 else 0.0

        signals: list[str] = []
        suspicious = False

        # Flag suspicion only when high-error pixel density exceeds prototype threshold.
        # Note: ELA is an explainable forensic signal, not proof of document forgery.
        if suspicious_ratio > self.suspicious_ratio_threshold:
            suspicious = True
            signals.append(
                f"High compression discrepancy detected: {suspicious_ratio:.2%} of pixels exceed error threshold"
            )

        return {
            "mean_error": round(mean_error, 4),
            "max_error": max_error,
            "suspicious_ratio": round(suspicious_ratio, 4),
            "suspicious": suspicious,
            "signals": signals,
        }
