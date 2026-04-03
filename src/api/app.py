from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from src.ner import extract_entities
from src.explainer import MedicalExplainer
from src.phi_deidentifier import detect_phi, deidentify_text
from src.ocr import extract_text_from_bytes

app = FastAPI(title="Clinical NER & Patient Summarization API")

# Cache explainer instances per backend to avoid re-initialization
_explainers: Dict[str, MedicalExplainer] = {}

def get_explainer(backend: str = "openai") -> MedicalExplainer:
    if backend not in _explainers:
        _explainers[backend] = MedicalExplainer(backend=backend)
    return _explainers[backend]

class ClinicalNoteRequest(BaseModel):
    text: str
    llm_backend: Optional[str] = "openai"  # "openai" or "ollama"
    model_type: Optional[str] = "custom"   # "custom" (SpaCy) or "pretrained" (d4data)

class Entity(BaseModel):
    text: str
    label: str
    score: float
    start_char: int
    end_char: int

class PHIEntity(BaseModel):
    entity_type: str
    text: str
    start: int
    end: int
    score: float
    description: str

class SummaryResponse(BaseModel):
    original_text: dict
    phi_detected: List[PHIEntity]
    deidentified_text: str
    entities: List[Entity]
    explanations: Dict[str, str]
    patient_summary: str
    llm_backend: str
    phi_deidentification_applied: bool

class OCRResult(BaseModel):
    extracted_text: str
    ocr_metadata: dict

class OCRSummaryResponse(SummaryResponse):
    ocr_metadata: dict

@app.post("/summarize", response_model=SummaryResponse)
def process_clinical_note(request: ClinicalNoteRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    backend = (request.llm_backend or "openai").lower().strip()
    
    try:
        explainer = get_explainer(backend)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    try:
        # 0. Detect and de-identify PHI
        phi_entities = detect_phi(request.text)
        safe_text, _ = deidentify_text(request.text)
                
        # 1. Extract entities (local model — safe to use original text)
        entities = extract_entities(request.text, model_type=request.model_type)
        
        # 2. Explain terms (de-identifies internally if cloud backend)
        explanations = explainer.explain_terms(entities, request.text)
        
        # 3. Generate patient-friendly summary (de-identifies internally if cloud backend)
        summary = explainer.generate_summary(request.text, explanations)
        
        return SummaryResponse(
            original_text={"text": request.text},
            phi_detected=phi_entities,
            deidentified_text=safe_text,
            entities=entities,
            explanations=explanations,
            patient_summary=summary,
            llm_backend=backend,
            phi_deidentification_applied=explainer.needs_deidentification,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ── OCR Endpoints ─────────────────────────────────────────────────────────────

ALLOWED_OCR_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/tiff",
    "image/bmp", "image/gif", "image/webp", "application/pdf",
}

@app.post("/ocr/extract", response_model=OCRResult)
async def ocr_extract(file: UploadFile = File(...)):
    """Extract text from an uploaded image or PDF using OCR only."""
    if file.content_type and file.content_type not in ALLOWED_OCR_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Supported: {', '.join(sorted(ALLOWED_OCR_TYPES))}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    text, metadata = extract_text_from_bytes(
        file_bytes, filename=file.filename or "",
    )

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="OCR could not extract any text from this file. "
                   "Try a higher resolution scan.",
        )

    return OCRResult(extracted_text=text, ocr_metadata=metadata)


@app.post("/ocr/summarize", response_model=OCRSummaryResponse)
async def ocr_summarize(
    file: UploadFile = File(...),
    llm_backend: str = Form(default="openai"),
    model_type: str = Form(default="custom"),
):
    """
    Full pipeline: OCR → NER → PHI Detection → Explain → Summarize.
    Accepts an image or PDF upload.
    """
    if file.content_type and file.content_type not in ALLOWED_OCR_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Supported: {', '.join(sorted(ALLOWED_OCR_TYPES))}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Step 0: OCR
    extracted_text, ocr_meta = extract_text_from_bytes(
        file_bytes, filename=file.filename or "",
    )

    if not extracted_text.strip():
        raise HTTPException(
            status_code=422,
            detail="OCR could not extract any text from this file. "
                   "Try a higher resolution scan.",
        )

    backend = (llm_backend or "openai").lower().strip()

    try:
        explainer = get_explainer(backend)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # Step 1: PHI detection
        phi_entities = detect_phi(extracted_text)
        safe_text, _ = deidentify_text(extracted_text)

        # Step 2: NER (local model)
        entities = extract_entities(extracted_text, model_type=model_type)

        # Step 3: Explain terms
        explanations = explainer.explain_terms(entities, extracted_text)

        # Step 4: Generate summary
        summary = explainer.generate_summary(extracted_text, explanations)

        return OCRSummaryResponse(
            original_text={"text": extracted_text},
            phi_detected=phi_entities,
            deidentified_text=safe_text,
            entities=entities,
            explanations=explanations,
            patient_summary=summary,
            llm_backend=backend,
            phi_deidentification_applied=explainer.needs_deidentification,
            ocr_metadata=ocr_meta,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
