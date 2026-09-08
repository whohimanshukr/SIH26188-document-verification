# SIH26188 — AI-Based Fake Identity & Document Screening System

**SIH Problem Statement:** SIH26188

---

## Current Prototype

The backend accepts a document image and a selfie and produces a structured JSON response containing:

- **OCR output** — raw text extracted from the document image
- **Extracted document fields** — name, date of birth, passport number, nationality (pattern-based)
- **MRZ analysis** — Machine Readable Zone parsing and cross-field consistency checking (for passport-style documents)
- **Metadata forensics** — EXIF/metadata inspection for editing software or suspicious tool signatures
- **ELA forensics** — Error Level Analysis to flag compression-level inconsistencies in the image
- **Face verification** — YuNet-based face detection and SFace-based biometric similarity comparison between the document portrait and the submitted selfie
- **Risk assessment** — deterministic scoring based on MRZ mismatches and forensic signals
- **Verification signals** — aggregated human-readable explanation strings from each module

---

## Current Verification Flow

```
Document Image + Selfie Image
        │
        ▼
    OCR (PaddleOCR)
        │
        ▼
 Field Extraction (regex-based)
        │
        ▼
   MRZ Detection & Parsing
        │
        ▼
 Metadata Forensics (EXIF)
        │
        ▼
    ELA Forensics
        │
        ▼
 Face Verification (YuNet + SFace)
        │
        ▼
   Risk Evaluation (RiskEngine)
        │
        ▼
     JSON Response
```

---

## Backend Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| API framework | FastAPI |
| OCR | PaddleOCR (PaddlePaddle) |
| Face detection | OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) |
| Face recognition | OpenCV SFace (`face_recognition_sface_2021dec.onnx`) |
| Image processing | OpenCV (`opencv-contrib-python`), Pillow |
| Numerics | NumPy |
| ASGI server | Uvicorn |

> **Python environment note:** The project uses a dedicated `.venv312` virtual environment.
> The OCR stack (PaddleOCR / PaddlePaddle) was validated with Python 3.12 on Windows.
> The commands below use the interpreter path directly to avoid PowerShell execution-policy
> issues with `Activate.ps1`.

---

## API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Liveness check — returns `{"message": "..."}` |
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `POST` | `/verify` | Full document + face verification |

### POST /verify

Accepts **multipart/form-data** with two image file fields:

| Field | Type | Description |
|-------|------|-------------|
| `document` | image file | Identity document image (passport, ID card, etc.) |
| `selfie` | image file | Live selfie / reference photo of the person |

**Supported extensions:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tiff`

Both files must be non-empty and carry a supported image extension.  
Temporary files are saved to `uploads/` with collision-safe UUID names and deleted immediately after processing — regardless of success or failure.

### Example Response Structure

> The following is an **illustrative structure only**, not a fixed sample with real data.

```json
{
  "status": "success",
  "document": {
    "fields": {
      "name": "...",
      "date_of_birth": "...",
      "passport_number": "...",
      "nationality": "..."
    }
  },
  "ocr": {
    "text": "...",
    "confidence": 0.95
  },
  "mrz": {
    "available": true,
    "parsed": { "surname": "...", "given_names": "...", "passport_number": "...", "nationality": "...", "date_of_birth": "..." },
    "matches": { "passport_number": true, "nationality": true, "date_of_birth": true, "name": true },
    "signals": []
  },
  "forensics": {
    "metadata": { "suspicious": false, "signals": [] },
    "ela": { "suspicious": false, "signals": [] }
  },
  "signals": [],
  "face": {
    "match": true,
    "similarity_score": 0.646,
    "document_face_detected": true,
    "selfie_face_detected": true,
    "signals": ["Faces match successfully (similarity: 0.646 >= 0.363)"]
  },
  "risk": {
    "risk_score": 0,
    "decision": "LOW",
    "reasons": ["All available verification checks passed"]
  }
}
```

---

## Modules

| Module | Path | Description |
|--------|------|-------------|
| OCR engine | `backend/ocr/ocr_engine.py` | Wraps PaddleOCR; extracts raw text and confidence from a document image |
| Field extractor | `backend/extraction/field_extractor.py` | Regex-based extraction of name, date of birth, passport number, and nationality from OCR text |
| MRZ parser | `backend/mrz/mrz_parser.py` | Parses TD3 (passport) Machine Readable Zone lines; returns structured field dict |
| Metadata forensics | `backend/forensics/metadata.py` | Reads EXIF/image metadata; flags known editing-software signatures and suspicious tool strings |
| ELA forensics | `backend/forensics/ela.py` | Performs Error Level Analysis; computes compression-level heat-map and flags high-discrepancy regions |
| Face verifier | `backend/face/face_verifier.py` | YuNet face detection + SFace feature extraction; computes cosine similarity and returns match result |
| Pipeline | `backend/pipeline.py` | Orchestrates OCR → field extraction → MRZ → forensics in one deterministic call; returns structured dict |
| Risk engine | `backend/risk/risk_engine.py` | Deterministic scoring from MRZ mismatches and forensic flags; maps total score to LOW / MEDIUM / HIGH |
| FastAPI app | `backend/main.py` | Exposes `GET /`, `GET /health`, `POST /verify`; handles file upload, validation, temp-file cleanup, error responses |

---

## Face Verification

- **Detection:** OpenCV YuNet (`cv2.FaceDetectorYN`) detects facial bounding boxes and landmarks in both the document image and the selfie.
- **Recognition:** OpenCV SFace (`cv2.FaceRecognizerSF`) aligns and crops the detected face, then extracts a 128-dimensional feature embedding.
- **Comparison:** Embeddings are compared using cosine similarity (`cv2.FaceRecognizerSF_FR_COSINE`).
- **Threshold:** Current default is **0.363** (the recommended SFace cosine threshold). Scores ≥ 0.363 are classified as a match.
- **Output:** The API returns `match` (bool), `similarity_score` (float, clamped to [0, 1]), detection flags, and human-readable signals.

> ⚠️ Face similarity is biometric **evidence**, not definitive proof of identity. This system does not claim to constitute legal or government-grade identity verification.

---

## Risk Engine

The `RiskEngine` uses a deterministic weighted scoring model. Scores accumulate additively; the final score is capped at 100.

| Check | Condition | Weight |
|-------|-----------|--------|
| MRZ unavailable | MRZ not detected or unparseable | +10 |
| Passport number mismatch | Visual zone ≠ MRZ passport number | +25 |
| Date of birth mismatch | Visual zone ≠ MRZ date of birth | +20 |
| Name mismatch | Visual zone name tokens ∩ MRZ name = ∅ | +20 |
| Nationality mismatch | Visual zone ≠ MRZ nationality | +15 |
| Suspicious metadata | EXIF flags editing software | +10 |
| Suspicious ELA | High compression discrepancy regions | +10 |

**Decision thresholds:**

| Score range | Decision |
|-------------|----------|
| 0 – 20 | `LOW` |
| 21 – 50 | `MEDIUM` |
| 51 – 100 | `HIGH` |

> **Note:** Face verification evidence is currently returned by the API under the `face` key but is **not yet used as a RiskEngine scoring factor**. The face result is available in the response for downstream consumption, but it does not affect `risk_score` or `decision` in the current implementation.

---

## Limitations

- **Prototype only.** This is not a production-grade identity verification system. No real government database or national ID registry is queried.
- **OCR accuracy.** PaddleOCR can make recognition errors on low-resolution, skewed, or degraded document scans.
- **Field extraction is not universal.** The current regex patterns are primarily designed around passport-style field layouts. Non-standard document formats may produce incomplete or missing fields.
- **Aadhaar-specific extraction is not implemented.** Aadhaar card field extraction requires a separate layout-aware strategy that is not yet built. The current extractor will return `null` for most Aadhaar-specific fields.
- **MRZ applies to passport-style documents only.** Aadhaar cards do not contain a passport MRZ. An `mrz.available: false` result is the correct and expected behavior for Aadhaar.
- **Forensic signals are indicators, not proof.** Metadata flags and ELA results suggest possible manipulation but cannot definitively prove tampering.
- **Face similarity is evidence, not identity proof.** A high similarity score increases confidence but is not a biometric guarantee.
- **No external database verification.** There is no real-time lookup against any external government, passport, or identity database.
- **Risk engine is deterministic and limited.** Risk scoring covers only the signals implemented above. Face mismatch evidence is not currently a scoring factor.
- **Not hardened for adversarial inputs.** No rate limiting, authentication, or adversarial-image resistance is implemented at the prototype stage.

---

## Setup

### Requirements

- Python 3.12
- The `.venv312` virtual environment (created once in the project root)

### Install Dependencies

```bat
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

### Run the API Server

```bat
.\.venv312\Scripts\python.exe -m uvicorn backend.main:app --reload
```

The server starts at:

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:8000` | API root |
| `http://127.0.0.1:8000/docs` | Swagger interactive UI |
| `http://127.0.0.1:8000/health` | Health check |

---

## Model Files

The face verification module requires two official OpenCV Zoo ONNX models placed at:

```
backend/face/models/
├── face_detection_yunet_2023mar.onnx    (232 KB)  — YuNet face detector
└── face_recognition_sface_2021dec.onnx (38.7 MB) — SFace face recognizer
```

Both files are tracked in this repository. They are the official models from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo).

`FaceVerifier` raises `FileNotFoundError` at startup if either model file is missing. Do **not** delete them.

---

## Testing

The following tests were actually performed during development:

| Test | Method | Result |
|------|--------|--------|
| Syntax check | `python -m py_compile backend/main.py` | ✅ No errors |
| Health endpoint | `GET /health` | ✅ `{"status": "ok"}` |
| Same-person match | `POST /verify` — same face as document and selfie | ✅ `face.match: true`, `similarity_score ≈ 0.646`, HTTP 200 |
| Different-person mismatch | `POST /verify` — different faces for document and selfie | ✅ `face.match: false`, `similarity_score ≈ 0.193`, threshold 0.363, HTTP 200 |
| Temporary file cleanup | Inspect `uploads/` after request | ✅ Only `.gitkeep` remains; both temp files removed |
| Swagger UI | Manual test via `http://127.0.0.1:8000/docs` | ✅ Both `document` and `selfie` fields accepted |

> Tests marked ✅ were manually executed and verified. No automated test suite is currently implemented.

---

## Project Structure

```
SIH26188-document-verification/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── pipeline.py                # Verification orchestration pipeline
│   ├── ocr/
│   │   └── ocr_engine.py          # PaddleOCR wrapper
│   ├── extraction/
│   │   └── field_extractor.py     # Regex field extractor
│   ├── mrz/
│   │   └── mrz_parser.py          # TD3 MRZ parser
│   ├── forensics/
│   │   ├── metadata.py            # EXIF metadata forensics
│   │   └── ela.py                 # Error Level Analysis
│   ├── face/
│   │   ├── face_verifier.py       # YuNet + SFace face verifier
│   │   └── models/
│   │       ├── face_detection_yunet_2023mar.onnx
│   │       └── face_recognition_sface_2021dec.onnx
│   └── risk/
│       └── risk_engine.py         # Deterministic risk scoring
├── uploads/                       # Temporary upload area (files gitignored)
├── tests/                         # Test directory (placeholder)
├── frontend/                      # Frontend placeholder
├── requirements.txt
└── .gitignore
```
