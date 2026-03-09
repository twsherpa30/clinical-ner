"""
Evaluate the custom SpaCy NER model on the MedMentions test split.

Usage:
    python evaluate_ner.py
    python evaluate_ner.py --model ./models/custom_ner/model-best
"""

import argparse
from pathlib import Path
from collections import defaultdict

import spacy
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score

from train_ner import load_and_convert_medmentions, NER_LABELS


def tokens_to_bio(doc, gold_entities):
    """
    Convert SpaCy Doc + gold entity spans into BIO tag sequences.

    Returns:
        (gold_tags, pred_tags) — lists of BIO tag strings
    """
    # Build gold BIO tags from character offsets
    gold_tags = ["O"] * len(doc)
    for start_char, end_char, label in gold_entities:
        span = doc.char_span(start_char, end_char, label=label, alignment_mode="contract")
        if span is None:
            continue
        for i, token in enumerate(span):
            tag_prefix = "B" if token == span[0] else "I"
            gold_tags[span.start + i - 0] = f"{tag_prefix}-{label}"
            # Fix: use absolute token index
        for i in range(span.start, span.end):
            tag_prefix = "B" if i == span.start else "I"
            gold_tags[i] = f"{tag_prefix}-{label}"

    # Build predicted BIO tags from model output
    pred_tags = ["O"] * len(doc)
    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            tag_prefix = "B" if i == ent.start else "I"
            pred_tags[i] = f"{tag_prefix}-{ent.label_}"

    return gold_tags, pred_tags


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpaCy NER on MedMentions")
    parser.add_argument("--model", type=str, default="./models/custom_ner/model-best",
                        help="Path to trained SpaCy model directory")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of test documents (0 = all)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("   Train a model first: python train_ner.py")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    nlp = spacy.load(model_path)
    print(f"  Pipeline: {nlp.pipe_names}")
    print(f"  NER labels: {nlp.get_pipe('ner').labels}")

    # Load test data
    print("\nLoading MedMentions test split...")
    test_data = load_and_convert_medmentions("test")

    if args.limit > 0:
        test_data = test_data[:args.limit]
        print(f"  (Limited to {args.limit} documents)")

    # Evaluate
    print(f"\nEvaluating on {len(test_data)} documents...")
    all_gold = []
    all_pred = []
    entity_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for text, annotations in test_data:
        doc = nlp(text)
        gold_tags, pred_tags = tokens_to_bio(doc, annotations["entities"])
        all_gold.append(gold_tags)
        all_pred.append(pred_tags)

    # Print classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Entity-Level)")
    print("=" * 60)
    print(classification_report(all_gold, all_pred, zero_division=0))

    overall_f1 = f1_score(all_gold, all_pred, average="weighted", zero_division=0)
    print(f"Weighted F1: {overall_f1:.4f}")


if __name__ == "__main__":
    main()
