# 🩺 ClinicalNER — Patient-Friendly Clinical Note Summarizer

An AI-powered NLP pipeline that takes complex clinical notes, extracts medical entities, detects and redacts PHI (Protected Health Information), explains medical jargon in plain language, and generates patient-friendly summaries.

Built with privacy as a first-class concern — supports both cloud (OpenAI) and fully local (Ollama) LLM backends. Features a **custom-trained SpaCy NER model** (trained from scratch on MedMentions) alongside a pre-trained HuggingFace model, with a UI toggle to switch between them. Includes OCR for processing scanned documents, handwritten notes, and faxed records.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Biomedical NER** | Dual-model support: a **custom SpaCy model** trained from scratch on MedMentions (10 entity categories) and a pre-trained HuggingFace model (`d4data/biomedical-ner-all`), selectable via UI toggle |
| **PHI De-identification** | Detects and redacts all 18 HIPAA Safe Harbor identifiers (names, dates, SSN, MRN, phone numbers, etc.) using Microsoft Presidio |
| **Plain-Language Explanations** | Translates medical jargon into 5th-grade reading level explanations via LLM |
| **Patient-Friendly Summaries** | Generates structured visit summaries (Reason for Visit, Conditions, Medications, Next Steps) |
| **Local LLM Support** | Run entirely offline with Ollama (Llama 3, Mistral, etc.) — no data ever leaves your machine |
| **OCR / Image Upload** | Extract text from scanned clinical notes, faxes, handwritten prescriptions, and PDFs via Tesseract |
| **REST API** | FastAPI backend with documented endpoints for integration |
| **Web Interface** | Streamlit-based UI with tabbed input, PHI shield display, and backend selector |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (ui.py)                        │
│   ┌──────────────────┐    ┌────────────────────────────────────┐    │
│   │  Paste Text   │    │  Upload Image / PDF (OCR)      │    │
│   └────────┬─────────┘    └──────────────┬─────────────────────┘    │
│            │                             │                          │
│            │    Sidebar: Backend Selector (OpenAI / Ollama)      │
└────────────┼─────────────────────────────┼──────────────────────────┘
             │                             │
             ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (app.py)                         │
│                                                                     │
│  POST /summarize          POST /ocr/extract    POST /ocr/summarize  │
│       │                        │                     │              │
│       │                        ▼                     │              │
│       │               ┌──────────────┐               │              │
│       │               │   OCR Engine  │◄─────────────┘              │
│       │               │   (ocr.py)   │                              │
│       │               │  Tesseract   │                              │
│       │               └──────┬───────┘                              │
│       │                      │ extracted text                       │
│       ▼                      ▼                                      │
│  ┌──────────────────────────────────────────┐                       │
│  │  Step 0: PHI Detection & De-identification │                     │
│  │          (phi_deidentifier.py)             │                     │
│  │          Microsoft Presidio               │                     │
│  └────────────────────┬─────────────────────┘                       │
│                       │                                             │
│  ┌────────────────────▼─────────────────────┐                       │
│  │  Step 1: Named Entity Recognition (NER)   │                     │
│  │          (ner.py)                         │                     │
│  │          Custom SpaCy / d4data model   │                     │
│  │          Runs locally — no PHI risk     │                     │
│  └────────────────────┬─────────────────────┘                       │
│                       │                                             │
│  ┌────────────────────▼─────────────────────┐                       │
│  │  Step 2: Medical Term Explanations        │                     │
│  │  Step 3: Patient-Friendly Summary         │                     │
│  │          (explainer.py)                   │                     │
│  │                                           │                     │
│  │  OpenAI  ──► de-identified text sent   │                     │
│  │  Ollama  ──► original text stays local │                     │
│  └──────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ClinicalNER/
├── app.py                 # FastAPI backend — REST API with all endpoints
├── ner.py                 # Dual-model NER (custom SpaCy + pre-trained HuggingFace)
├── explainer.py           # Dual-backend LLM explainer (OpenAI / Ollama)
├── phi_deidentifier.py    # PHI detection & redaction using Presidio
├── ocr.py                 # OCR engine using Tesseract (images + PDFs)
├── ui.py                  # Streamlit frontend with tabbed interface
├── train_ner.py           # Train custom SpaCy NER model locally
├── train_ner_colab.ipynb  # Colab notebook for GPU-accelerated training
├── evaluate_ner.py        # Evaluate trained model with seqeval metrics
├── requirements.txt       # Python dependencies
├── models/
│   └── custom_ner_model/  # Trained SpaCy NER model (trained on MedMentions)
└── venv/                  # Python virtual environment
```

### Module Details

#### `ner.py` — Named Entity Recognition
- **Dual-model architecture** — switch between models via API parameter or UI toggle:
  - **Custom SpaCy model** (default): Trained from scratch on [MedMentions](https://huggingface.co/datasets/Aremaki/MedMentions) (4,392 PubMed abstracts, 350K+ entity mentions). 10 entity categories mapped from 127 UMLS Semantic Types.
  - **Pre-trained model**: [`d4data/biomedical-ner-all`](https://huggingface.co/d4data/biomedical-ner-all) (HuggingFace transformer)
- **Custom model entity categories**: `DISEASE`, `CHEMICAL_DRUG`, `PROCEDURE`, `ANATOMY`, `SIGN_SYMPTOM`, `ORGANISM`, `GENE_PROTEIN`, `MEDICAL_DEVICE`, `LAB_TEST`, `OTHER`
- **Deduplication**: Filters short tokens (<3 chars) and removes duplicate mentions
- **Runs locally** — no network calls, no PHI risk

#### `train_ner.py` / `train_ner_colab.ipynb` — Model Training
- **Dataset**: [MedMentions](https://huggingface.co/datasets/Aremaki/MedMentions) — 2,635 train / 878 validation / 879 test documents
- **UMLS type mapping**: 127 fine-grained UMLS Semantic Types → 10 practical NER categories
- **Architecture**: SpaCy Tok2Vec + Transition-based NER (no transformer dependency)
- **Training**: Colab notebook recommended (~25 min on CPU, ~5-10 min with GPU)
- **Best dev F1**: 41.0% | **Test weighted F1**: 39.5%
- **Usage**: `python train_ner.py --dry-run` (validate) or upload `train_ner_colab.ipynb` to Colab

#### `phi_deidentifier.py` — PHI De-identification
- **Engine**: [Microsoft Presidio](https://github.com/microsoft/presidio) with spaCy `en_core_web_lg`
- **Custom recognizers** for healthcare-specific PHI:
  - Medical Record Numbers (MRN) — various formats
  - Health Plan Beneficiary Numbers / Member IDs
  - Account Numbers
  - Age ≥ 90 (HIPAA considers ages over 89 as PHI)
  - Social Security Numbers
- **Built-in Presidio recognizers**: Person names, dates, phone numbers, email addresses, locations, IP addresses, URLs, driver's licenses, passport numbers, credit card numbers
- **Configurable sensitivity**: `score_threshold` parameter (default 0.4 — aggressive redaction for safety)

#### `explainer.py` — Dual-Backend Medical Explainer
- **OpenAI mode**: Uses `gpt-4o-mini` for term explanations + `gpt-4o` for summaries. PHI is de-identified before sending.
- **Ollama mode**: Uses any local model (default: `llama3`). No API key required. PHI de-identification is skipped since data never leaves the machine.
- **Backend selection** via:
  1. Constructor: `MedicalExplainer(backend="ollama")`
  2. Environment variable: `LLM_BACKEND=ollama`
  3. API request parameter: `llm_backend`
- **Smart JSON parsing**: Handles both strict JSON mode (OpenAI) and freeform responses with markdown code fences (Ollama)
- **In-memory caching**: Avoids redundant LLM calls for previously explained terms

#### `ocr.py` — OCR Engine
- **Engine**: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) v5.5+ with LSTM neural net
- **Image preprocessing** pipeline optimized for clinical documents:
  1. Grayscale conversion
  2. Contrast boost (2x — helps with faded faxes)
  3. Sharpening (helps with blurry scans)
  4. Binarization (clean background noise)
- **PDF support**: Converts each page to an image at 300 DPI via `pdf2image`
- **Confidence scoring**: Reports average OCR confidence per document
- **Text cleanup**: Removes OCR artifacts, collapses blank lines, fixes common misreads
- **Formats**: PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP, PDF

#### `app.py` — FastAPI Backend
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/summarize` | POST | Full text pipeline: PHI → NER → Explain → Summary |
| `/ocr/extract` | POST | OCR only — extract text from uploaded image/PDF |
| `/ocr/summarize` | POST | Full pipeline from image: OCR → PHI → NER → Explain → Summary |
| `/health` | GET | Health check |

#### `ui.py` — Streamlit Frontend
- **Sidebar**: LLM backend selector (OpenAI Cloud / Ollama Local), **NER model selector** (Custom SpaCy / Pre-trained), model configuration
- **Tab 1 — Paste Text**: Text area with sample clinical note, "Generate Summary" button
- **Tab 2 — Upload Image/PDF**: Drag-and-drop file upload with image preview, two action buttons:
  - **Extract Text Only** (OCR)
  - **Full Pipeline** (OCR → NER → Summary)
- **PHI Privacy Shield**: Visual display of detected PHI with confidence scores
- **Results**: Side-by-side entity explanations and patient-friendly summary

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Tesseract OCR** (for image/PDF processing)
- **Ollama** (optional — for local LLM mode)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd ClinicalNER

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download the spaCy model (required by Presidio)
python -m spacy download en_core_web_lg

# 5. Install Tesseract OCR (macOS)
brew install tesseract

# 6. (Optional) Install Ollama for local LLM
brew install ollama
ollama pull llama3
```

### Running the Application

You need **two terminals** (and optionally a third for Ollama):

```bash
# Terminal 1 — Start the FastAPI backend
cd ~/Downloads/ClinicalNER
source venv/bin/activate
export OPENAI_API_KEY='sk-...'   # Only if using OpenAI mode
python app.py
# Server starts at http://localhost:8000

# Terminal 2 — Start the Streamlit UI
cd ~/Downloads/ClinicalNER
source venv/bin/activate
streamlit run ui.py
# Opens browser at http://localhost:8501

# Terminal 3 (Optional) — Start Ollama for local LLM
ollama serve
# Runs at http://localhost:11434
```

---

## Privacy & HIPAA Considerations

### PHI De-identification
When using the **OpenAI (cloud)** backend, all clinical text is de-identified before being sent to external APIs:

```
Original:  "Patient John Smith (DOB: 03/15/1958, MRN: 00384721)..."
Sent to AI: "Patient [PERSON] (DOB: [DATE_TIME], [MEDICAL_RECORD_NUMBER])..."
```

### HIPAA Safe Harbor Identifiers Detected

| # | Identifier | Detection Method |
|---|-----------|-----------------|
| 1 | Names | spaCy NER (PERSON) |
| 2 | Dates | Pattern + NER |
| 3 | Phone/Fax numbers | Pattern matching |
| 4 | Email addresses | Pattern matching |
| 5 | Social Security Numbers | Custom pattern recognizer |
| 6 | Medical Record Numbers | Custom pattern recognizer |
| 7 | Health Plan IDs | Custom pattern recognizer |
| 8 | Account Numbers | Custom pattern recognizer |
| 9 | Geographic data | spaCy NER (LOCATION) |
| 10 | Ages over 89 | Custom pattern recognizer |
| 11 | URLs | Pattern matching |
| 12 | IP addresses | Pattern matching |
| 13 | Driver's license numbers | Built-in Presidio |
| 14 | Passport numbers | Built-in Presidio |
| 15 | Credit card numbers | Built-in Presidio |
| 16 | Bank account numbers | Built-in Presidio |

### Fully Local Mode (Ollama)
When using the **Ollama** backend:
- **Zero data leaves your machine** — all processing is local
- No API keys required
- PHI de-identification is skipped (not needed)
- Works offline once the model is downloaded

---

## API Usage Examples

### Summarize a Clinical Note (Text)
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient presents with severe dyspnea and hypertension. Prescribed Lisinopril 20mg.",
    "llm_backend": "ollama",
    "model_type": "custom"
  }'
```

> **`model_type`** options: `"custom"` (SpaCy, default) or `"pretrained"` (d4data HuggingFace)

### OCR — Extract Text from Image
```bash
curl -X POST http://localhost:8000/ocr/extract \
  -F "file=@clinical_note_scan.png"
```

### OCR — Full Pipeline from Image
```bash
curl -X POST http://localhost:8000/ocr/summarize \
  -F "file=@clinical_note_scan.png" \
  -F "llm_backend=ollama"
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for OpenAI backend |
| `LLM_BACKEND` | `openai` | Default LLM backend (`openai` or `ollama`) |
| `OLLAMA_MODEL` | `llama3` | Which Ollama model to use |
| `MODEL_TYPE` | `custom` | Default NER model (`custom` or `pretrained`) |

### Recommended Ollama Models

| Model | Size | Best For |
|-------|------|----------|
| `llama3` | 4.7 GB | General-purpose, good quality |
| `llama3:70b` | 40 GB | Highest quality (needs 64GB+ RAM) |
| `mistral` | 4.1 GB | Fast, efficient |
| `medllama2` | 4.1 GB | Medical-specialized |
| `gemma2` | 5.4 GB | Good multilingual support |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| NER (Custom) | SpaCy EntityRecognizer trained on MedMentions |
| NER (Pre-trained) | HuggingFace Transformers (`d4data/biomedical-ner-all`) |
| NER Training | SpaCy `train` + MedMentions dataset (Colab-ready) |
| PHI Detection | Microsoft Presidio + spaCy `en_core_web_lg` |
| LLM (Cloud) | OpenAI GPT-4o / GPT-4o-mini |
| LLM (Local) | Ollama (Llama 3, Mistral, etc.) |
| OCR | Tesseract 5.5+ with LSTM engine |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Image Processing | Pillow (PIL) |
| PDF Processing | pdf2image + Poppler |

---

