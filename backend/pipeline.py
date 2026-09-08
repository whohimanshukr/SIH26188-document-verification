"""Document verification orchestration pipeline combining OCR, extraction, MRZ, and forensics."""

from pathlib import Path
import re
from typing import Any, TypedDict

from backend.extraction.field_extractor import FieldExtractor
from backend.forensics.ela import ELAAnalyzer, ELAResult
from backend.forensics.metadata import MetadataAnalyzer, MetadataAnalysisResult
from backend.mrz.mrz_parser import MRZParser, ParsedMRZ
from backend.ocr.ocr_engine import OCREngine, OCRResult


class MRZSection(TypedDict):
    """Normalized representation of MRZ detection, parsing, and consistency checks."""

    available: bool
    parsed: ParsedMRZ | None
    matches: dict[str, bool | None]
    signals: list[str]


class ForensicsSection(TypedDict):
    """Combined forensics analysis results."""

    metadata: MetadataAnalysisResult
    ela: ELAResult


class VerificationPipelineResult(TypedDict):
    """Top-level response structure for document verification pipeline."""

    status: str
    document: dict[str, Any]
    ocr: OCRResult
    mrz: MRZSection
    forensics: ForensicsSection
    signals: list[str]


class DocumentVerificationPipeline:
    """Deterministic orchestration pipeline for document verification demo."""

    # Regex matching typical TD3 MRZ characters (uppercase letters, digits, and filler '<')
    _MRZ_CHAR_PATTERN = re.compile(r"^[A-Z0-9<]+$")

    def __init__(self) -> None:
        # Initialize heavy engines and analyzers once to avoid reloading models on every call
        self.ocr_engine = OCREngine()
        self.field_extractor = FieldExtractor()
        self.mrz_parser = MRZParser()
        self.metadata_analyzer = MetadataAnalyzer()
        self.ela_analyzer = ELAAnalyzer()

    def verify(self, image_path: str | Path) -> VerificationPipelineResult:
        """Run full verification pipeline on a document image.

        Args:
            image_path: Path to the target document image.

        Returns:
            VerificationPipelineResult dictionary with OCR, extraction, MRZ, and forensics data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If image_path is not a file.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Document image not found: {path}")
        if not path.is_file():
            raise ValueError(f"Document image path is not a file: {path}")

        # 1. OCR text extraction
        ocr_result: OCRResult = self.ocr_engine.extract_text(path)

        # 2. Deterministic field extraction from OCR text
        extracted_fields = self.field_extractor.extract_fields(ocr_result["text"])

        # 3. Forensics: Metadata analysis
        metadata_result: MetadataAnalysisResult = self.metadata_analyzer.analyze(path)

        # 4. Forensics: Error Level Analysis
        ela_result: ELAResult = self.ela_analyzer.analyze(path)

        # 5. MRZ extraction & cross-field consistency
        mrz_section = self._process_mrz(ocr_result["text"], extracted_fields)

        # Aggregate overall informational and forensic signals
        pipeline_signals: list[str] = []
        if metadata_result.get("signals"):
            pipeline_signals.extend(metadata_result["signals"])
        if ela_result.get("signals"):
            pipeline_signals.extend(ela_result["signals"])
        if mrz_section.get("signals"):
            pipeline_signals.extend(mrz_section["signals"])

        return {
            "status": "success",
            "document": {
                "fields": extracted_fields,
            },
            "ocr": ocr_result,
            "mrz": mrz_section,
            "forensics": {
                "metadata": metadata_result,
                "ela": ela_result,
            },
            "signals": pipeline_signals,
        }

    def _process_mrz(
        self, ocr_text: str, extracted_fields: dict[str, str | None]
    ) -> MRZSection:
        """Detect, parse, and cross-reference MRZ lines with OCR-extracted fields."""
        candidate_lines = self._detect_mrz_lines(ocr_text)

        if not candidate_lines:
            return {
                "available": False,
                "parsed": None,
                "matches": {},
                "signals": ["MRZ could not be detected from OCR output"],
            }

        try:
            parsed_mrz = self.mrz_parser.parse(candidate_lines)
        except ValueError as err:
            return {
                "available": False,
                "parsed": None,
                "matches": {},
                "signals": [f"MRZ detected but failed parsing: {err}"],
            }

        matches, consistency_signals = self._compare_fields(extracted_fields, parsed_mrz)

        return {
            "available": True,
            "parsed": parsed_mrz,
            "matches": matches,
            "signals": consistency_signals,
        }

    def _detect_mrz_lines(self, ocr_text: str) -> list[str] | None:
        """Scan OCR lines for two consecutive TD3 MRZ candidate strings."""
        raw_lines = [line.strip().replace(" ", "") for line in ocr_text.splitlines()]

        # Scan for consecutive pairs in the candidates or original lines meeting length & char pattern
        for i in range(len(raw_lines) - 1):
            line_a = raw_lines[i]
            line_b = raw_lines[i + 1]
            if (
                40 <= len(line_a) <= 44
                and 40 <= len(line_b) <= 44
                and self._MRZ_CHAR_PATTERN.match(line_a)
                and self._MRZ_CHAR_PATTERN.match(line_b)
            ):
                return [line_a, line_b]

        # Fallback: find any two candidate lines in document order
        mrz_candidates = [
            line for line in raw_lines
            if 40 <= len(line) <= 44 and self._MRZ_CHAR_PATTERN.match(line)
        ]
        if len(mrz_candidates) >= 2:
            return [mrz_candidates[0], mrz_candidates[1]]

        return None

    def _compare_fields(
        self, ocr_fields: dict[str, str | None], mrz: ParsedMRZ
    ) -> tuple[dict[str, bool | None], list[str]]:
        """Compare OCR extracted visual inspection zone fields with parsed MRZ fields."""
        matches: dict[str, bool | None] = {}
        signals: list[str] = []

        # 1. Passport Number comparison
        ocr_pass = ocr_fields.get("passport_number")
        mrz_pass = mrz.get("passport_number")
        if ocr_pass and mrz_pass:
            clean_ocr_pass = re.sub(r"[^A-Z0-9]", "", ocr_pass.upper())
            clean_mrz_pass = re.sub(r"[^A-Z0-9]", "", mrz_pass.upper())
            is_match = clean_ocr_pass == clean_mrz_pass
            matches["passport_number"] = is_match
            if not is_match:
                signals.append(
                    f"Passport number mismatch: Visual '{ocr_pass}' vs MRZ '{mrz_pass}'"
                )
        else:
            matches["passport_number"] = None

        # 2. Nationality comparison (supports ISO code or partial containment)
        ocr_nat = ocr_fields.get("nationality")
        mrz_nat = mrz.get("nationality")
        if ocr_nat and mrz_nat:
            clean_ocr_nat = re.sub(r"[^A-Z]", "", ocr_nat.upper())
            clean_mrz_nat = re.sub(r"[^A-Z]", "", mrz_nat.upper())
            is_match = (
                (clean_ocr_nat == clean_mrz_nat)
                or (clean_ocr_nat in clean_mrz_nat)
                or (clean_mrz_nat in clean_ocr_nat)
            )
            matches["nationality"] = is_match
            if not is_match:
                signals.append(
                    f"Nationality mismatch: Visual '{ocr_nat}' vs MRZ '{mrz_nat}'"
                )
        else:
            matches["nationality"] = None

        # 3. Date of Birth comparison
        # OCR is typically YYYY-MM-DD or DD/MM/YYYY; MRZ is 6 digits (YYMMDD)
        ocr_dob = ocr_fields.get("date_of_birth")
        mrz_dob = mrz.get("date_of_birth")
        if ocr_dob and mrz_dob:
            digits_ocr = re.sub(r"\D", "", ocr_dob)
            digits_mrz = re.sub(r"\D", "", mrz_dob)
            is_match = (
                (digits_mrz in digits_ocr)
                or (digits_ocr[-6:] == digits_mrz if len(digits_ocr) >= 6 else False)
            )
            matches["date_of_birth"] = is_match
            if not is_match:
                signals.append(
                    f"Date of birth mismatch: Visual '{ocr_dob}' vs MRZ '{mrz_dob}'"
                )
        else:
            matches["date_of_birth"] = None

        # 4. Loose Name comparison
        ocr_name = ocr_fields.get("name")
        mrz_surname = mrz.get("surname", "")
        mrz_given = mrz.get("given_names", "")
        if ocr_name and (mrz_surname or mrz_given):
            full_mrz_name = f"{mrz_given} {mrz_surname}".strip().upper()
            clean_ocr_tokens = set(re.sub(r"[^A-Z ]", "", ocr_name.upper()).split())
            clean_mrz_tokens = set(re.sub(r"[^A-Z ]", "", full_mrz_name).split())

            is_match = bool(clean_ocr_tokens.intersection(clean_mrz_tokens))
            matches["name"] = is_match
            if not is_match:
                signals.append(
                    f"Name mismatch: Visual '{ocr_name}' vs MRZ '{mrz_surname} {mrz_given}'"
                )
        else:
            matches["name"] = None

        return matches, signals

