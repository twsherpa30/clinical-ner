"""
Medical Explainer Module — Dual Backend (OpenAI Cloud / Ollama Local)

Supports two LLM backends:
  • "openai"  — Uses the OpenAI API (requires OPENAI_API_KEY).
                PHI is de-identified before sending to the cloud.
  • "ollama"  — Uses a local Ollama instance (http://localhost:11434).
                No API key needed. Data never leaves the machine,
                so PHI de-identification is skipped for performance.

The backend can be selected via:
  1. Constructor parameter: MedicalExplainer(backend="ollama")
  2. Environment variable:  LLM_BACKEND=ollama
  3. Default:               "openai"
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

from src.phi_deidentifier import deidentify_text


# ── Backend configuration ─────────────────────────────────────────────────────

BACKEND_CONFIGS = {
    "openai": {
        "base_url": None,                    # Uses default OpenAI endpoint
        "api_key": None,                     # Reads from OPENAI_API_KEY env var
        "explain_model": "gpt-4o-mini",      # Fast, cheap for term explanations
        "summary_model": "gpt-4o",           # Capable model for full summaries
        "supports_json_mode": True,          # OpenAI supports response_format
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",                 # Ollama ignores this but the SDK requires it
        "explain_model": "llama3",           # Override via OLLAMA_MODEL env var
        "summary_model": "llama3",           # Same model for both tasks
        "supports_json_mode": False,         # Most Ollama models don't support this
    },
}


class MedicalExplainer:
    """
    Generates plain-language explanations of medical terms and
    patient-friendly summaries using either OpenAI or a local Ollama model.
    """

    def __init__(self, backend: Optional[str] = None):
        """
        Args:
            backend: "openai" or "ollama". Falls back to LLM_BACKEND env var,
                     then defaults to "openai".
        """
        self.backend = (
            backend
            or os.environ.get("LLM_BACKEND", "openai")
        ).lower().strip()

        if self.backend not in BACKEND_CONFIGS:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                f"Choose from: {list(BACKEND_CONFIGS.keys())}"
            )

        config = BACKEND_CONFIGS[self.backend]

        # Allow overriding the Ollama model via env var
        if self.backend == "ollama":
            ollama_model = os.environ.get("OLLAMA_MODEL", config["explain_model"])
            config["explain_model"] = ollama_model
            config["summary_model"] = ollama_model

        self.explain_model = config["explain_model"]
        self.summary_model = config["summary_model"]
        self.supports_json_mode = config["supports_json_mode"]

        # Whether to de-identify text before sending to the LLM
        # Cloud backends: YES (data leaves the machine)
        # Local backends:  NO  (data stays on-premises)
        self.needs_deidentification = (self.backend == "openai")

        # Initialize the OpenAI-compatible client
        self.client = None
        self._init_error = None

        try:
            client_kwargs = {}
            if config["base_url"]:
                client_kwargs["base_url"] = config["base_url"]

            if self.backend == "openai":
                # OpenAI requires an API key — check explicitly
                api_key = os.environ.get("OPENAI_API_KEY", "").strip()
                if not api_key:
                    self._init_error = (
                        "OPENAI_API_KEY is not set. Please either:\n"
                        "  1. Set it:  export OPENAI_API_KEY='sk-...'  (in the terminal running app.py)\n"
                        "  2. Or switch to the local Ollama backend in the sidebar."
                    )
                    print(f"Warning: {self._init_error}")
                else:
                    client_kwargs["api_key"] = api_key
                    self.client = OpenAI(**client_kwargs)
            else:
                # Ollama and other local backends
                if config["api_key"]:
                    client_kwargs["api_key"] = config["api_key"]
                self.client = OpenAI(**client_kwargs)

        except Exception as e:
            self._init_error = f"{self.backend} client error: {e}"
            print(f"Warning: {self._init_error}")

        # Simple in-memory cache to avoid redundant LLM calls
        self.explanation_cache: Dict[str, str] = {}

        status = "✅ ready" if self.client else f"❌ {self._init_error}"
        print(f"[MedicalExplainer] Backend: {self.backend} | "
              f"Models: {self.explain_model} / {self.summary_model} | "
              f"PHI de-id: {'ON' if self.needs_deidentification else 'OFF (local)'} | "
              f"Status: {status}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _safe_text(self, text: str) -> str:
        """De-identify text if using a cloud backend; return as-is for local."""
        if self.needs_deidentification:
            safe, _ = deidentify_text(text)
            return safe
        return text

    def _parse_json_response(self, content: str) -> dict:
        """
        Parse JSON from LLM response. Handles both strict JSON mode
        (OpenAI) and freeform responses where JSON may be wrapped in
        markdown code fences (common with Ollama/local models).
        """
        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in the text
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    # ── Core Methods ───────────────────────────────────────────────────────────

    def explain_terms(self, entities: List[Dict[str, Any]], context: str) -> Dict[str, str]:
        """
        Takes a list of extracted entities and generates plain-language explanations.
        """
        if not self.client:
            err = self._init_error or "LLM client not initialized."
            return {ent['text']: err for ent in entities}

        explanations = {}
        terms_to_explain = []

        for ent in entities:
            term = ent['text']
            if len(term) <= 2:
                continue
            if term.lower() in self.explanation_cache:
                explanations[term] = self.explanation_cache[term.lower()]
            else:
                terms_to_explain.append(term)

        if not terms_to_explain:
            return explanations

        # De-identify context only if needed (cloud backend)
        safe_context = self._safe_text(context)

        prompt = (
            f"Context from clinical note: '{safe_context}'\n\n"
            "Explain the following medical terms in very simple, plain language "
            "(at a 5th-grade reading level) as if speaking directly to the patient.\n"
            "Return the result strictly as a JSON dictionary mapping the term to its explanation.\n"
            "Do not include any other text outside the JSON.\n"
            f"Terms to explain: {', '.join(terms_to_explain)}"
        )

        try:
            create_kwargs = {
                "model": self.explain_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful, empathetic doctor explaining medical jargon to a patient. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            }
            # Only use JSON mode if the backend supports it
            if self.supports_json_mode:
                create_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**create_kwargs)
            result_json = self._parse_json_response(response.choices[0].message.content)

            for term, explanation in result_json.items():
                matching_term = next(
                    (t for t in terms_to_explain if t.lower() == term.lower()), term
                )
                explanations[matching_term] = explanation
                self.explanation_cache[matching_term.lower()] = explanation

        except Exception as e:
            print(f"Error generating explanations ({self.backend}): {e}")
            for term in terms_to_explain:
                explanations[term] = "Could not generate explanation."

        return explanations

    def generate_summary(self, original_text: str, explanations: Dict[str, str]) -> str:
        """
        Generates a structured, patient-friendly summary of the visit.
        """
        if not self.client:
            return self._init_error or "LLM client not initialized."

        # De-identify only for cloud backend
        safe_text = self._safe_text(original_text)

        prompt = (
            f"Original clinical note:\n{safe_text}\n\n"
            f"Medical terms explained:\n{json.dumps(explanations, indent=2)}\n\n"
            "Please rewrite the clinical note into a structured, patient-friendly summary. "
            "Use an empathetic, encouraging tone. Address the patient directly as 'you'.\n"
            "Structure the output in Markdown with the following sections:\n"
            "- **Reason for Visit**\n"
            "- **Your Conditions (What they mean)**\n"
            "- **Your Medications (Why you are taking them)**\n"
            "- **Next Steps / Plan**"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": "You are a helpful, empathetic doctor providing a summary to a patient."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating summary ({self.backend}): {e}")
            return "Could not generate summary."


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Testing with backend: {backend or '(auto-detect)'}\n")

    explainer = MedicalExplainer(backend=backend)
    sample_text = (
        "The patient is a 65-year-old male presenting with severe dyspnea "
        "and hypertension. He was prescribed Lisinopril 20mg."
    )
    sample_entities = [
        {"text": "dyspnea", "label": "Sign_symptom"},
        {"text": "hypertension", "label": "Disease_disorder"},
        {"text": "Lisinopril", "label": "Medication"},
    ]

    can_run = (
        explainer.backend == "ollama"
        or os.environ.get("OPENAI_API_KEY")
    )

    if can_run:
        print("Explaining terms...")
        explanations = explainer.explain_terms(sample_entities, sample_text)
        print(json.dumps(explanations, indent=2))

        print("\nGenerating summary...")
        summary = explainer.generate_summary(sample_text, explanations)
        print(summary)
    else:
        print("OPENAI_API_KEY not set and not using Ollama. Skipping test.")
