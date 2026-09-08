"""Application entry point for document verification services."""

from pathlib import Path
import shutil
import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile, status

from backend.face.face_verifier import FaceVerifier
from backend.pipeline import DocumentVerificationPipeline
from backend.risk.risk_engine import RiskEngine

app = FastAPI(title="SIH26188 Document Verification")

# Initialize pipeline, risk engine, and face verifier once at application startup
# so heavy models and weights are not reloaded on every single HTTP request.
pipeline = DocumentVerificationPipeline()
risk_engine = RiskEngine()
face_verifier = FaceVerifier()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Allowed standard image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _validate_image_file(file: UploadFile, field_name: str) -> str:
    """Validate that an uploaded file has a filename and supported image extension."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded {field_name} must have a filename",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}' for {field_name}. Expected an image file.",
        )
    return ext


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Document verification API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/verify")
def verify_document(
    document: UploadFile = File(...),
    selfie: UploadFile = File(...),
) -> dict:
    """Verify an identity document image and compare with reference selfie.

    Processes OCR, visual field extraction, TD3 MRZ parsing, image forensics,
    and biometric face comparison between the document portrait and selfie image.
    Both uploaded files are saved temporarily and cleaned up immediately after verification.
    """
    doc_ext = _validate_image_file(document, "document")
    selfie_ext = _validate_image_file(selfie, "selfie")

    # Save to collision-safe unique temporary files in uploads/
    doc_temp_path = UPLOAD_DIR / f"doc_{uuid.uuid4().hex}{doc_ext}"
    selfie_temp_path = UPLOAD_DIR / f"selfie_{uuid.uuid4().hex}{selfie_ext}"

    try:
        with doc_temp_path.open("wb") as buffer:
            shutil.copyfileobj(document.file, buffer)

        if doc_temp_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded document file is empty",
            )

        with selfie_temp_path.open("wb") as buffer:
            shutil.copyfileobj(selfie.file, buffer)

        if selfie_temp_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded selfie file is empty",
            )

        # 1. Run document pipeline verification (OCR, fields, MRZ, forensics)
        try:
            result = pipeline.verify(doc_temp_path)
        except (ValueError, FileNotFoundError) as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document processing failed: {err}",
            )
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during document verification",
            ) from err

        # 2. Run facial biometric verification
        try:
            face_result = face_verifier.verify(doc_temp_path, selfie_temp_path)
            result["face"] = face_result
        except (ValueError, FileNotFoundError) as err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Face verification failed: {err}",
            )
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during face verification",
            ) from err

        # 3. Evaluate risk using RiskEngine
        # Note: RiskEngine currently evaluates MRZ and forensics signals.
        # Face evidence is preserved in the response under result['face'].
        risk_result = risk_engine.evaluate(result)
        result["risk"] = risk_result

        return result

    finally:
        # Always remove both temporary files from disk to prevent storage leaks and protect privacy
        for path in (doc_temp_path, selfie_temp_path):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass


