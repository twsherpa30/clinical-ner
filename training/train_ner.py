"""
Train a SpaCy NER model from scratch on the MedMentions dataset.

Usage:
    python -m training.train_ner                 # Full training run
    python -m training.train_ner --dry-run       # Validate data pipeline only
    python -m training.train_ner --epochs 20     # Override number of epochs
    python -m training.train_ner --gpu 0         # Train on GPU 0

Output:
    ./models/custom_ner/model-best/     # Best model checkpoint
    ./models/custom_ner/model-last/     # Last model checkpoint
    ./data/train.spacy                  # Training data (DocBin)
    ./data/dev.spacy                    # Validation data (DocBin)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import spacy
from spacy.tokens import DocBin
from datasets import load_dataset

# ── UMLS Semantic Type → High-Level NER Category Mapping ──────────────────────
# MedMentions uses 127 fine-grained UMLS Semantic Type IDs (TUIs).
# We group them into 10 practical clinical NER categories.

UMLS_TYPE_TO_LABEL = {
    # DISEASE: Diseases, syndromes, pathologic functions, neoplasms, congenital abnormalities
    "T047": "DISEASE", "T019": "DISEASE", "T046": "DISEASE", "T191": "DISEASE",
    "T048": "DISEASE", "T049": "DISEASE", "T050": "DISEASE", "T190": "DISEASE",
    "T037": "DISEASE",  # Injury or Poisoning

    # CHEMICAL_DRUG: Pharmacologic substances, clinical drugs, antibiotics
    "T121": "CHEMICAL_DRUG", "T200": "CHEMICAL_DRUG", "T195": "CHEMICAL_DRUG",
    "T109": "CHEMICAL_DRUG", "T114": "CHEMICAL_DRUG", "T131": "CHEMICAL_DRUG",
    "T125": "CHEMICAL_DRUG", "T129": "CHEMICAL_DRUG", "T130": "CHEMICAL_DRUG",
    "T118": "CHEMICAL_DRUG", "T119": "CHEMICAL_DRUG", "T110": "CHEMICAL_DRUG",
    "T111": "CHEMICAL_DRUG", "T196": "CHEMICAL_DRUG", "T127": "CHEMICAL_DRUG",
    "T123": "CHEMICAL_DRUG", "T122": "CHEMICAL_DRUG", "T120": "CHEMICAL_DRUG",
    "T126": "CHEMICAL_DRUG", "T103": "CHEMICAL_DRUG", "T104": "CHEMICAL_DRUG",

    # PROCEDURE: Therapeutic/diagnostic procedures, lab procedures
    "T061": "PROCEDURE", "T060": "PROCEDURE", "T065": "PROCEDURE",
    "T058": "PROCEDURE", "T059": "PROCEDURE", "T063": "PROCEDURE",

    # ANATOMY: Body parts, tissues, cells, body systems
    "T023": "ANATOMY", "T024": "ANATOMY", "T017": "ANATOMY",
    "T029": "ANATOMY", "T030": "ANATOMY", "T025": "ANATOMY",
    "T018": "ANATOMY", "T021": "ANATOMY", "T022": "ANATOMY",
    "T026": "ANATOMY", "T031": "ANATOMY",

    # SIGN_SYMPTOM: Signs, symptoms, clinical findings
    "T184": "SIGN_SYMPTOM", "T033": "SIGN_SYMPTOM", "T034": "SIGN_SYMPTOM",

    # ORGANISM: Organisms, bacteria, viruses, fungi
    "T001": "ORGANISM", "T002": "ORGANISM", "T004": "ORGANISM",
    "T005": "ORGANISM", "T007": "ORGANISM", "T008": "ORGANISM",
    "T010": "ORGANISM", "T011": "ORGANISM", "T012": "ORGANISM",
    "T013": "ORGANISM", "T014": "ORGANISM", "T015": "ORGANISM",
    "T016": "ORGANISM", "T096": "ORGANISM", "T101": "ORGANISM",

    # GENE_PROTEIN: Genes, proteins, enzymes, amino acid sequences
    "T028": "GENE_PROTEIN", "T116": "GENE_PROTEIN", "T126": "GENE_PROTEIN",
    "T192": "GENE_PROTEIN", "T087": "GENE_PROTEIN", "T088": "GENE_PROTEIN",

    # MEDICAL_DEVICE: Devices, instruments, manufactured objects
    "T074": "MEDICAL_DEVICE", "T075": "MEDICAL_DEVICE", "T203": "MEDICAL_DEVICE",
    "T073": "MEDICAL_DEVICE",

    # LAB_TEST: Lab or test results, quantitative/qualitative concepts
    "T034": "LAB_TEST", "T201": "LAB_TEST",

    # OTHER will catch anything not explicitly mapped
}

# All unique labels we support
NER_LABELS = sorted(set(UMLS_TYPE_TO_LABEL.values())) + ["OTHER"]


def get_label_for_types(semantic_types: list) -> str:
    """Map a list of UMLS semantic type IDs to a single NER label."""
    for st in semantic_types:
        # Extract the TUI (e.g. "T047") — handle both "T047" and full URIs
        tui = st.split("/")[-1] if "/" in st else st
        if tui in UMLS_TYPE_TO_LABEL:
            return UMLS_TYPE_TO_LABEL[tui]
    return "OTHER"


def load_and_convert_medmentions(split: str = "train"):
    """
    Load a MedMentions split and extract (text, entities) pairs.

    Returns:
        List of (text, {"entities": [(start, end, label), ...]}) tuples
    """
    print(f"Loading MedMentions '{split}' split...")
    ds = load_dataset(
        "Aremaki/MedMentions",
        name="Original",
        split=split,
    )
    print(f"  Loaded {len(ds)} documents")

    training_data = []
    label_counts = Counter()
    skipped_overlaps = 0

    for example in ds:
        # Each example has 'passages' (list of text passages) and 'entities'
        passages = example.get("passages", [])
        entities = example.get("entities", [])

        # Reconstruct the full text from passages
        if not passages:
            continue

        # Use the first passage (title + abstract combined)
        full_text = ""
        passage_offsets = []
        for passage in passages:
            for text_segment in passage.get("text", []):
                offset_start = len(full_text)
                full_text += text_segment + " "
                passage_offsets.append((offset_start, len(full_text)))

        full_text = full_text.strip()
        if not full_text:
            continue

        # Extract entity spans
        doc_entities = []
        occupied = set()

        for ent in entities:
            offsets = ent.get("offsets", [])
            ent_type = ent.get("type", "")

            # The 'type' field contains the UMLS Semantic Type ID (TUI)
            # e.g., "T047" for Disease, "T103" for Chemical
            # Use it directly for label mapping
            label = get_label_for_types([ent_type]) if ent_type else "OTHER"

            for start, end in offsets:
                # Clamp to text length
                start = max(0, min(start, len(full_text)))
                end = max(start, min(end, len(full_text)))

                if start == end:
                    continue

                # Check for overlapping entities (SpaCy doesn't support them)
                char_range = set(range(start, end))
                if char_range & occupied:
                    skipped_overlaps += 1
                    continue

                occupied.update(char_range)
                doc_entities.append((start, end, label))
                label_counts[label] += 1

        # Sort by start position
        doc_entities.sort(key=lambda x: x[0])
        training_data.append((full_text, {"entities": doc_entities}))

    print(f"  Extracted {sum(label_counts.values())} entities from {len(training_data)} documents")
    print(f"  Skipped {skipped_overlaps} overlapping entities")
    print(f"  Label distribution: {dict(label_counts.most_common())}")

    return training_data


def create_docbin(training_data, nlp, output_path: Path):
    """Convert (text, entities) pairs to a SpaCy DocBin file."""
    db = DocBin()
    skipped = 0

    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        ents = []
        seen_tokens = set()

        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                skipped += 1
                continue

            # Avoid token-level overlaps
            span_tokens = set(range(span.start, span.end))
            if span_tokens & seen_tokens:
                skipped += 1
                continue

            seen_tokens.update(span_tokens)
            ents.append(span)

        doc.ents = ents
        db.add(doc)

    db.to_disk(output_path)
    total_ents = sum(len(doc_data[1]["entities"]) for doc_data in training_data)
    print(f"  Saved {output_path} — {len(training_data)} docs, "
          f"{total_ents - skipped} entities ({skipped} skipped due to alignment)")

    return len(training_data), total_ents - skipped


def generate_config(config_path: Path, labels: list, gpu_id: int = -1):
    """Generate a SpaCy training config file."""
    label_entries = "\n".join(f'    "{lbl}",' for lbl in sorted(labels))

    config_text = f"""[system]
gpu_allocator = null
seed = 42

[nlp]
lang = "en"
pipeline = ["tok2vec","ner"]
batch_size = 1000
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {{"@tokenizers":"spacy.Tokenizer.v1"}}

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 128
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
rows = [5000,2500,2500,2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 128
depth = 4
window_size = 1
maxout_pieces = 3

[components.ner]
factory = "ner"
incorrect_spans_key = null
moves = null
scorer = {{"@scorers":"spacy.ner_scorer.v1"}}
update_with_oracle_cut_size = 100

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${{components.tok2vec.model.encode.width}}
upstream = "*"

[paths]
train = null
dev = null
vectors = null
init_tok2vec = null

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${{system.seed}}
gpu_allocator = ${{system.gpu_allocator}}
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = true

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.00005

[training.score_weights]
ents_per_type = null
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0

[pretraining]

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[initialize]
vectors = null
init_tok2vec = ${{paths.init_tok2vec}}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""
    config_path.write_text(config_text)
    print(f"  Config saved to {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SpaCy NER on MedMentions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate data pipeline without training")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Max training epochs (0 = use max_steps instead)")
    parser.add_argument("--max-steps", type=int, default=20000,
                        help="Max training steps (default: 20000)")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--output", type=str, default="./models/custom_ner",
                        help="Output directory for trained model")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    output_dir = Path(args.output)

    # ── Step 1: Load and convert data ─────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading MedMentions dataset")
    print("=" * 60)

    train_data = load_and_convert_medmentions("train")
    dev_data = load_and_convert_medmentions("validation")

    # ── Step 2: Create DocBin files ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Converting to SpaCy DocBin format")
    print("=" * 60)

    nlp = spacy.blank("en")

    train_path = data_dir / "train.spacy"
    dev_path = data_dir / "dev.spacy"

    n_train_docs, n_train_ents = create_docbin(train_data, nlp, train_path)
    n_dev_docs, n_dev_ents = create_docbin(dev_data, nlp, dev_path)

    # ── Step 3: Generate config ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Generating SpaCy config")
    print("=" * 60)

    config_path = project_root / "training" / "config.cfg"
    generate_config(config_path, NER_LABELS, gpu_id=args.gpu)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Training docs:    {n_train_docs} ({n_train_ents} entities)")
    print(f"  Validation docs:  {n_dev_docs} ({n_dev_ents} entities)")
    print(f"  Labels:           {NER_LABELS}")
    print(f"  Train file:       {train_path}")
    print(f"  Dev file:         {dev_path}")
    print(f"  Config file:      {config_path}")

    if args.dry_run:
        print("\n✅ Dry run complete — data pipeline validated successfully!")
        print("   Run without --dry-run to start training.")
        return

    # ── Step 4: Train ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Training SpaCy NER model")
    print("=" * 60)

    train_cmd = (
        f"{sys.executable} -m spacy train {config_path} "
        f"--output {output_dir} "
        f"--paths.train {train_path} "
        f"--paths.dev {dev_path} "
    )

    if args.gpu >= 0:
        train_cmd += f"--gpu-id {args.gpu} "

    if args.epochs > 0:
        train_cmd += f"--training.max_epochs {args.epochs} "

    if args.max_steps > 0:
        train_cmd += f"--training.max_steps {args.max_steps} "

    print(f"  Command: {train_cmd}")
    print()

    exit_code = os.system(train_cmd)

    if exit_code == 0:
        print("\n✅ Training complete!")
        print(f"   Best model: {output_dir / 'model-best'}")
        print(f"   Last model: {output_dir / 'model-last'}")
    else:
        print(f"\n❌ Training failed with exit code {exit_code}")
        sys.exit(1)


if __name__ == "__main__":
    main()
