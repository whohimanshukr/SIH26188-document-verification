# SIH26188 Document Verification

A document verification system organized around OCR, data extraction, MRZ parsing, forensic checks, face verification, persistence, and risk scoring.

## Structure

- `backend/ocr/` - OCR processing
- `backend/extraction/` - Field extraction and normalization
- `backend/mrz/` - Machine-readable zone parsing and validation
- `backend/forensics/` - Document integrity and tamper checks
- `backend/face/` - Face detection and matching
- `backend/database/` - Persistence and data access
- `backend/risk/` - Risk scoring and decision rules
- `tests/` - Automated tests
- `uploads/` - Local upload area; files are ignored by Git

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend\main.py
```

The implementation modules are intentionally empty placeholders until the verification workflow and API contract are defined.
