"""Application entry point for document verification services."""

from fastapi import FastAPI


app = FastAPI(title="SIH26188 Document Verification")


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Document verification API is running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
