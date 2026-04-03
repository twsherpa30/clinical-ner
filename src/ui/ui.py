import streamlit as st
import requests
import json
import os

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Patient-Friendly Clinical Notes", layout="wide")

# ── Sidebar: LLM Backend Selection ────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    llm_backend = st.radio(
        "LLM Backend",
        options=["openai", "ollama"],
        format_func=lambda x: {
            "openai": "☁️  OpenAI (Cloud)",
            "ollama": "🏠 Ollama (Local)",
        }[x],
        index=0,
        help=(
            "**OpenAI**: Uses GPT-4o via the cloud. Requires OPENAI_API_KEY. "
            "PHI is automatically de-identified before sending.\n\n"
            "**Ollama**: Runs a local LLM (e.g. Llama 3, Mistral). "
            "No API key needed. Data never leaves your machine."
        ),
    )

    if llm_backend == "ollama":
        st.success("🔒 **HIPAA-safe mode** — all data stays on your machine.")
        ollama_model = st.text_input(
            "Ollama model name",
            value="llama3",
            help="Must be already pulled via `ollama pull <model>`.",
        )
        st.caption(
            "Make sure Ollama is running: `ollama serve`"
        )
    else:
        if "OPENAI_API_KEY" not in os.environ:
            st.warning(
                "⚠️ OPENAI_API_KEY is not set in the environment."
            )
        st.info(
            "PHI is automatically de-identified before data is sent to OpenAI."
        )

    st.divider()

    # ── NER Model Selection ────────────────────────────────────────────────
    st.header("🧠 NER Model")
    model_type = st.radio(
        "Entity Recognition Model",
        options=["custom", "pretrained"],
        format_func=lambda x: {
            "custom": "🎯 Custom SpaCy (MedMentions)",
            "pretrained": "📦 Pre-trained (d4data)",
        }[x],
        index=0,
        help=(
            "**Custom**: SpaCy NER model trained from scratch on MedMentions. "
            "Must be trained first via `python -m training.train_ner`.\n\n"
            "**Pre-trained**: Uses the `d4data/biomedical-ner-all` HuggingFace model."
        ),
    )

    st.divider()
    st.caption("ClinicalNER v2.0 — NER + Explainer + PHI Shield + OCR")

# ── Main Content ──────────────────────────────────────────────────────────────
st.title("🩺 Patient-Friendly Clinical Note Summarizer")
st.markdown(
    "This tool takes a complex clinical note, extracts medical terms, explains them "
    "in simple language, and provides a structured summary for the patient. "
    "**Upload an image/PDF** or paste text directly."
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: render results (shared between text and OCR tabs)
# ═══════════════════════════════════════════════════════════════════════════════
def render_results(data: dict):
    """Display PHI detection, entities, and summary from API response."""
    used_backend = data.get("llm_backend", llm_backend)

    # ── PHI Detection ─────────────────────────────────────────────────────
    phi_list = data.get("phi_detected", [])
    safe_text = data.get("deidentified_text", "")

    st.header("🔒 PHI Privacy Shield")

    if used_backend == "ollama":
        st.success(
            "🏠 **Local LLM mode** — your data never left this machine. "
            "PHI de-identification was not needed."
        )
        if phi_list:
            with st.expander(f"PHI scan found {len(phi_list)} identifier(s) (informational only)"):
                for phi in phi_list:
                    score_pct = int(phi["score"] * 100)
                    st.markdown(
                        f"- 🛡️ **`{phi['entity_type']}`** — "
                        f"*\"{phi['text']}\"* → {phi['description']}  "
                        f"(confidence: {score_pct}%)"
                    )
    elif phi_list:
        st.warning(
            f"**{len(phi_list)} PHI element(s) detected** and redacted "
            "before sending to OpenAI."
        )
        for phi in phi_list:
            score_pct = int(phi["score"] * 100)
            st.markdown(
                f"- 🛡️ **`{phi['entity_type']}`** — "
                f"*\"{phi['text']}\"* → {phi['description']}  "
                f"(confidence: {score_pct}%)"
            )
        with st.expander("View de-identified text sent to AI"):
            st.code(safe_text, language="text")
    else:
        st.success("✅ No PHI detected in this note.")

    # ── Entities & Summary ────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📋 Extracted Entities & Explanations")
        if data.get("explanations"):
            for term, explanation in data["explanations"].items():
                st.markdown(f"**{term}**: {explanation}")
        else:
            st.write("No distinct medical terms found to explain.")

    with col2:
        st.header("💬 Patient-Friendly Summary")
        st.markdown(
            data.get("patient_summary", "Summary generation failed.")
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Input Tabs
# ═══════════════════════════════════════════════════════════════════════════════
st.header("1. Input Clinical Note")

tab_text, tab_ocr = st.tabs(["📝 Paste Text", "📷 Upload Image / PDF"])

# ── Tab 1: Paste Text ─────────────────────────────────────────────────────────
with tab_text:
    sample_note = (
        "Patient John Smith (DOB: 03/15/1958, MRN: 00384721) is a 65-year-old male "
        "presenting with severe dyspnea and hypertension. He was prescribed Lisinopril "
        "20mg daily and instructed to monitor his blood pressure. Echocardiogram showed "
        "mild LVH. Contact his wife Mary Smith at (555) 867-5309 if there are changes. "
        "Follow-up scheduled for 04/10/2026."
    )

    note_input = st.text_area(
        "Paste the clinical note here:", value=sample_note, height=200
    )

    if st.button("Generate Patient Summary", key="btn_text"):
        backend_label = "☁️ OpenAI" if llm_backend == "openai" else "🏠 Ollama"
        with st.spinner(f"Processing with {backend_label}..."):
            try:
                payload = {"text": note_input, "llm_backend": llm_backend, "model_type": model_type}
                response = requests.post(f"{API_URL}/summarize", json=payload)

                if response.status_code == 200:
                    render_results(response.json())
                else:
                    st.error(f"Error from API: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the backend API. Please make sure "
                    "the API server is running on port 8000."
                )

# ── Tab 2: Upload Image/PDF ───────────────────────────────────────────────────
with tab_ocr:
    st.markdown(
        "Upload a **scanned clinical note**, **handwritten prescription**, "
        "**faxed document**, or **PDF**. Text will be extracted via OCR "
        "and then processed through the full pipeline."
    )

    uploaded_file = st.file_uploader(
        "Choose an image or PDF",
        type=["png", "jpg", "jpeg", "tiff", "bmp", "gif", "webp", "pdf"],
        help="Supported: PNG, JPG, TIFF, BMP, GIF, WEBP, PDF",
    )

    if uploaded_file is not None:
        # Show preview for images
        if uploaded_file.type and uploaded_file.type.startswith("image/"):
            st.image(uploaded_file, caption=uploaded_file.name, width=400)
        else:
            st.info(f"📄 Uploaded: **{uploaded_file.name}** ({uploaded_file.type})")

        col_ocr_only, col_full = st.columns(2)

        with col_ocr_only:
            ocr_only = st.button("🔍 Extract Text Only (OCR)", key="btn_ocr_only")

        with col_full:
            full_pipeline = st.button(
                "🚀 Full Pipeline (OCR → NER → Summary)", key="btn_ocr_full"
            )

        if ocr_only:
            with st.spinner("Running OCR..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/ocr/extract", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        meta = result.get("ocr_metadata", {})

                        st.success(
                            f"✅ OCR complete — **{meta.get('word_count', '?')} words** extracted "
                            f"(avg confidence: {meta.get('avg_confidence', '?')}%)"
                        )

                        st.subheader("Extracted Text")
                        st.text_area(
                            "OCR Output (you can copy or edit this):",
                            value=result["extracted_text"],
                            height=300,
                            key="ocr_output",
                        )

                        with st.expander("📊 OCR Metadata"):
                            st.json(meta)
                    else:
                        detail = response.json().get("detail", response.text)
                        st.error(f"OCR failed: {detail}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the backend API. Please make sure "
                        "the API server is running on port 8000."
                    )

        if full_pipeline:
            backend_label = "☁️ OpenAI" if llm_backend == "openai" else "🏠 Ollama"
            with st.spinner(f"Running OCR + full pipeline with {backend_label}..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    form_data = {"llm_backend": llm_backend, "model_type": model_type}
                    response = requests.post(
                        f"{API_URL}/ocr/summarize",
                        files=files,
                        data=form_data,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        ocr_meta = data.get("ocr_metadata", {})

                        st.success(
                            f"✅ OCR extracted **{ocr_meta.get('word_count', ocr_meta.get('total_word_count', '?'))} words** "
                            f"(confidence: {ocr_meta.get('avg_confidence', '?')}%)"
                        )

                        with st.expander("📄 View extracted text from OCR"):
                            st.code(
                                data.get("original_text", {}).get("text", ""),
                                language="text",
                            )

                        render_results(data)
                    else:
                        detail = response.json().get("detail", response.text)
                        st.error(f"Pipeline failed: {detail}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the backend API. Please make sure "
                        "the API server is running on port 8000."
                    )
