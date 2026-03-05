from transformers import pipeline
from typing import List, Dict, Any

# Load the HuggingFace NER pipeline
try:
    # Using aggregation_strategy="simple" to combine B/I tags
    ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
except Exception as e:
    print(f"Error loading model: {e}")
    ner_pipeline = None

def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract medical entities from a clinical note using HuggingFace.
    
    Args:
        text: The clinical note text.
        
    Returns:
        A list of dictionaries containing the extracted entities and their metadata.
    """
    if ner_pipeline is None:
        raise RuntimeError("NER pipeline is not loaded.")
        
    extracted_raw = ner_pipeline(text)
    
    entities = []
    seen_texts = set()
    
    for ent in extracted_raw:
        # ent typically has: entity_group, score, word, start, end
        label = ent.get('entity_group', '')
        word = ent.get('word', '').strip()
        
        # We only really care about medications, procedures, diseases, symptoms
        lower_word = word.lower()
        if len(word) < 3 or lower_word in seen_texts:
            continue
            
        seen_texts.add(lower_word)
        
        entities.append({
            "text": word,
            "label": label,
            "score": float(ent.get("score", 0.0)),
            "start_char": ent.get("start"),
            "end_char": ent.get("end")
        })
        
    return entities

if __name__ == "__main__":
    # Test sample
    sample_text = "The patient is a 65-year-old male presenting with severe dyspnea and hypertension. He was prescribed Lisinopril 20mg."
    extracted = extract_entities(sample_text)
    print("Input Text:", sample_text)
    print("\nExtracted Entities:")
    for e in extracted:
        print(f" - {e['text']} ({e['label']}) [Score: {e['score']:.2f}]")
