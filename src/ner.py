import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from transformers import pipeline
import spacy

# ── Model Configuration ───────────────────────────────────────────────────────
CUSTOM_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "custom_ner_model"

# Lazy-loaded model singletons
_pretrained_pipeline = None
_custom_nlp = None


def _load_pretrained():
    """Load the pre-trained HuggingFace NER pipeline (d4data/biomedical-ner-all)."""
    global _pretrained_pipeline
    if _pretrained_pipeline is None:
        try:
            _pretrained_pipeline = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
            )
            print("[NER] Loaded pre-trained model: d4data/biomedical-ner-all")
        except Exception as e:
            print(f"[NER] Error loading pre-trained model: {e}")
    return _pretrained_pipeline


def _load_custom():
    """Load the custom SpaCy NER model trained on MedMentions."""
    global _custom_nlp
    if _custom_nlp is None:
        if not CUSTOM_MODEL_PATH.exists():
            print(f"[NER] Custom model not found at: {CUSTOM_MODEL_PATH}")
            print("      Train one first: python -m training.train_ner")
            return None
        try:
            _custom_nlp = spacy.load(CUSTOM_MODEL_PATH)
            labels = _custom_nlp.get_pipe("ner").labels
            print(f"[NER] Loaded custom SpaCy model from {CUSTOM_MODEL_PATH}")
            print(f"      Labels: {labels}")
        except Exception as e:
            print(f"[NER] Error loading custom model: {e}")
    return _custom_nlp


def _extract_pretrained(text: str) -> List[Dict[str, Any]]:
    """Extract entities using the pre-trained HuggingFace model."""
    ner_pipe = _load_pretrained()
    if ner_pipe is None:
        raise RuntimeError("Pre-trained NER pipeline is not loaded.")

    extracted_raw = ner_pipe(text)
    entities = []
    seen_texts = set()

    for ent in extracted_raw:
        label = ent.get("entity_group", "")
        word = ent.get("word", "").strip()

        lower_word = word.lower()
        if len(word) < 3 or lower_word in seen_texts:
            continue

        seen_texts.add(lower_word)

        entities.append({
            "text": word,
            "label": label,
            "score": float(ent.get("score", 0.0)),
            "start_char": ent.get("start"),
            "end_char": ent.get("end"),
        })

    return entities


def _extract_custom(text: str) -> List[Dict[str, Any]]:
    """Extract entities using the custom SpaCy model."""
    nlp = _load_custom()
    if nlp is None:
        raise RuntimeError(
            "Custom NER model is not loaded. "
            "Train one first: python -m training.train_ner"
        )

    doc = nlp(text)
    entities = []
    seen_texts = set()

    for ent in doc.ents:
        word = ent.text.strip()
        lower_word = word.lower()

        if len(word) < 3 or lower_word in seen_texts:
            continue

        seen_texts.add(lower_word)

        entities.append({
            "text": word,
            "label": ent.label_,
            "score": 1.0,  # SpaCy NER doesn't produce per-entity scores
            "start_char": ent.start_char,
            "end_char": ent.end_char,
        })

    return entities


def extract_entities(
    text: str,
    model_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract medical entities from a clinical note.

    Args:
        text: The clinical note text.
        model_type: "custom" (SpaCy, trained on MedMentions) or
                    "pretrained" (HuggingFace d4data model).
                    Defaults to MODEL_TYPE env var, then "custom".

    Returns:
        A list of entity dicts with keys: text, label, score, start_char, end_char.
    """
    if model_type is None:
        model_type = os.environ.get("MODEL_TYPE", "custom").lower().strip()

    if model_type == "pretrained":
        return _extract_pretrained(text)
    else:
        return _extract_custom(text)


# ── Eager-load whichever model is configured ──────────────────────────────────
_default_model = os.environ.get("MODEL_TYPE", "custom").lower().strip()
if _default_model == "pretrained":
    _load_pretrained()
else:
    _load_custom()


if __name__ == "__main__":
    sample_text = (
        "The patient is a 65-year-old male presenting with severe dyspnea "
        "and hypertension. He was prescribed Lisinopril 20mg."
    )

    for mtype in ["custom", "pretrained"]:
        print(f"\n{'=' * 50}")
        print(f"Model: {mtype}")
        print("=" * 50)
        try:
            extracted = extract_entities(sample_text, model_type=mtype)
            print(f"Input: {sample_text}\n")
            for e in extracted:
                print(f"  - {e['text']} ({e['label']}) [Score: {e['score']:.2f}]")
            if not extracted:
                print("  (no entities found)")
        except RuntimeError as err:
            print(f"  ⚠️  {err}")
