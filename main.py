"""
ClinicalNER — Main Entry Point

Starts the FastAPI backend server.

Usage:
    python main.py                          # Start API server
    streamlit run src/ui/ui.py              # Start Streamlit UI (separate terminal)
    python -m training.train_ner            # Train NER model
    python -m training.evaluate_ner         # Evaluate NER model
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env before anything else

import uvicorn
from src.api.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
