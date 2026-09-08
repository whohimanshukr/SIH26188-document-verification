"""Deterministic extraction of common document fields from OCR text."""

import re
from typing import Pattern


class FieldExtractor:
    """Extract labeled identity-document fields using regular expressions."""

    _LABELS: dict[str, Pattern[str]] = {
        "name": re.compile(r"^(?:full\s+name|name)\s*:\s*(.+?)\s*$", re.IGNORECASE),
        "passport_number": re.compile(
            r"^(?:passport\s+(?:no|number))\s*:\s*([A-Z]\d{7,9})\s*$",
            re.IGNORECASE,
        ),
        "date_of_birth": re.compile(
            r"^(?:dob|date\s+of\s+birth)\s*:\s*(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})\s*$",
            re.IGNORECASE,
        ),
        "nationality": re.compile(
            r"^nationality\s*:\s*([A-Za-z]{2,3})\s*$", re.IGNORECASE
        ),
        "expiry_date": re.compile(
            r"^(?:expiry|date\s+of\s+expiry)\s*:\s*(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})\s*$",
            re.IGNORECASE,
        ),
    }

    def extract_fields(self, ocr_text: str) -> dict[str, str | None]:
        """Extract supported labeled fields from OCR text."""
        fields: dict[str, str | None] = {
            "name": None,
            "passport_number": None,
            "date_of_birth": None,
            "nationality": None,
            "expiry_date": None,
        }

        # OCR commonly emits one labeled value per line, so line matching keeps
        # extraction deterministic and prevents values from consuming neighbors.
        for line in ocr_text.splitlines():
            normalized_line = line.strip()
            for field_name, pattern in self._LABELS.items():
                match = pattern.match(normalized_line)
                if match:
                    fields[field_name] = match.group(1).strip()
                    break

        return fields