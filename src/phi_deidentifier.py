"""
PHI De-identification Module
Detects and redacts Protected Health Information (PHI) from clinical notes
before they are sent to external APIs (e.g., OpenAI).

Uses Microsoft Presidio for entity detection and anonymization.
Includes custom recognizers for healthcare-specific PHI such as
Medical Record Numbers (MRN) and common clinical note PHI patterns.
"""

import re
from typing import List, Dict, Any, Tuple

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# ---------------------------------------------------------------------------
# Custom Healthcare-Specific Recognizers
# ---------------------------------------------------------------------------

# Medical Record Number (MRN) – common formats: MRN-123456, MRN: 123456, MR#123456
mrn_patterns = [
    Pattern("MRN_PATTERN_1", r"\bMRN[\s\-:#]*\d{4,10}\b", 0.9),
    Pattern("MRN_PATTERN_2", r"\bMR#?\s*\d{4,10}\b", 0.85),
    Pattern("MRN_PATTERN_3", r"\bMedical Record(?:\s*Number)?[\s:#]*\d{4,10}\b", 0.95),
]
mrn_recognizer = PatternRecognizer(
    supported_entity="MEDICAL_RECORD_NUMBER",
    name="MRN_Recognizer",
    patterns=mrn_patterns,
    supported_language="en",
)

# Health Plan Beneficiary Number
hpbn_patterns = [
    Pattern("HPBN_PATTERN", r"\bHPBN[\s:#]*\d{6,12}\b", 0.85),
    Pattern("MEMBER_ID_PATTERN", r"\bMember\s*ID[\s:#]*[A-Z0-9]{6,15}\b", 0.80),
    Pattern("POLICY_PATTERN", r"\bPolicy[\s:#]*[A-Z0-9]{6,15}\b", 0.75),
]
hpbn_recognizer = PatternRecognizer(
    supported_entity="HEALTH_PLAN_ID",
    name="HPBN_Recognizer",
    patterns=hpbn_patterns,
    supported_language="en",
)

# Account Numbers
account_patterns = [
    Pattern("ACCOUNT_PATTERN", r"\bAcct?[\s.#:]*\d{6,12}\b", 0.80),
    Pattern("ACCOUNT_NUM_PATTERN", r"\bAccount\s*(?:Number|No|#)[\s:#]*\d{6,12}\b", 0.90),
]
account_recognizer = PatternRecognizer(
    supported_entity="ACCOUNT_NUMBER",
    name="Account_Recognizer",
    patterns=account_patterns,
    supported_language="en",
)

# Social Security Number (various formats)
ssn_patterns = [
    Pattern("SSN_PATTERN_1", r"\b\d{3}-\d{2}-\d{4}\b", 0.95),
    Pattern("SSN_PATTERN_2", r"\bSSN[\s:#]*\d{3}-?\d{2}-?\d{4}\b", 0.99),
]
ssn_recognizer = PatternRecognizer(
    supported_entity="US_SSN",
    name="SSN_Recognizer",
    patterns=ssn_patterns,
    supported_language="en",
)

# Age over 89 (HIPAA considers ages > 89 as PHI)
age_over_89_patterns = [
    Pattern("AGE_OVER_89", r"\b(9[0-9]|1[0-9]{2})\s*[-–]?\s*year[\s-]*old\b", 0.85),
    Pattern("AGE_OVER_89_SHORT", r"\bage\s*(9[0-9]|1[0-9]{2})\b", 0.80),
]
age_over_89_recognizer = PatternRecognizer(
    supported_entity="AGE_OVER_89",
    name="Age_Over_89_Recognizer",
    patterns=age_over_89_patterns,
    supported_language="en",
)


# ---------------------------------------------------------------------------
# HIPAA Safe-Harbor PHI categories (18 identifiers)
# ---------------------------------------------------------------------------
HIPAA_PHI_CATEGORIES = {
    "PERSON":                   "Patient/Person Name",
    "DATE_TIME":                "Date (admission, discharge, DOB, etc.)",
    "PHONE_NUMBER":             "Phone/Fax Number",
    "EMAIL_ADDRESS":            "Email Address",
    "US_SSN":                   "Social Security Number",
    "LOCATION":                 "Geographic Location / Address",
    "IP_ADDRESS":               "IP Address",
    "URL":                      "Web URL",
    "MEDICAL_RECORD_NUMBER":    "Medical Record Number (MRN)",
    "HEALTH_PLAN_ID":           "Health Plan Beneficiary Number",
    "ACCOUNT_NUMBER":           "Account Number",
    "AGE_OVER_89":              "Age ≥ 90 (HIPAA PHI)",
    "US_DRIVER_LICENSE":        "Driver's License Number",
    "US_PASSPORT":              "Passport Number",
    "CREDIT_CARD":              "Credit Card Number (if present)",
    "US_BANK_NUMBER":           "Bank Account Number",
    "IBAN_CODE":                "IBAN Code",
}


# ---------------------------------------------------------------------------
# Main De-identification Class
# ---------------------------------------------------------------------------

class PHIDeidentifier:
    """
    Detects and redacts Protected Health Information (PHI) from clinical text.
    
    Two modes:
      - detect():     returns detected PHI entities with positions
      - deidentify():  returns redacted text + list of what was removed
    """

    def __init__(self, score_threshold: float = 0.4):
        """
        Args:
            score_threshold: Minimum confidence score to consider a detection valid.
                             Lower = more aggressive redaction (safer for HIPAA).
        """
        self.score_threshold = score_threshold

        # Build the analyzer with custom recognizers
        self.analyzer = AnalyzerEngine()
        self.analyzer.registry.add_recognizer(mrn_recognizer)
        self.analyzer.registry.add_recognizer(hpbn_recognizer)
        self.analyzer.registry.add_recognizer(account_recognizer)
        self.analyzer.registry.add_recognizer(ssn_recognizer)
        self.analyzer.registry.add_recognizer(age_over_89_recognizer)

        self.anonymizer = AnonymizerEngine()

    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PHI entities in the text without modifying it.

        Returns:
            A list of dicts: {entity_type, text, start, end, score, description}
        """
        results: List[RecognizerResult] = self.analyzer.analyze(
            text=text,
            language="en",
            score_threshold=self.score_threshold,
        )

        detections = []
        for r in sorted(results, key=lambda x: x.start):
            detections.append({
                "entity_type": r.entity_type,
                "text": text[r.start:r.end],
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
                "description": HIPAA_PHI_CATEGORIES.get(
                    r.entity_type, r.entity_type
                ),
            })

        return detections

    def deidentify(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact PHI from the text, replacing detected entities with
        placeholder tags like [PERSON], [DATE_TIME], etc.

        Returns:
            (redacted_text, list_of_detected_phi)
        """
        # Detect
        results: List[RecognizerResult] = self.analyzer.analyze(
            text=text,
            language="en",
            score_threshold=self.score_threshold,
        )

        # Build a record of what was found (before anonymization)
        detections = []
        for r in sorted(results, key=lambda x: x.start):
            detections.append({
                "entity_type": r.entity_type,
                "original_text": text[r.start:r.end],
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
                "description": HIPAA_PHI_CATEGORIES.get(
                    r.entity_type, r.entity_type
                ),
            })

        # Anonymize — replace each entity type with a tag: [ENTITY_TYPE]
        operators = {}
        for r in results:
            operators[r.entity_type] = OperatorConfig(
                "replace", {"new_value": f"[{r.entity_type}]"}
            )

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )

        return anonymized.text, detections


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_deidentifier: PHIDeidentifier | None = None


def _get_default() -> PHIDeidentifier:
    global _default_deidentifier
    if _default_deidentifier is None:
        _default_deidentifier = PHIDeidentifier()
    return _default_deidentifier


def detect_phi(text: str) -> List[Dict[str, Any]]:
    """Convenience function: detect PHI in text."""
    return _get_default().detect(text)


def deidentify_text(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Convenience function: de-identify text and return (redacted_text, detections)."""
    return _get_default().deidentify(text)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Patient John Smith (DOB: 03/15/1958, SSN: 123-45-6789) was admitted on "
        "01/20/2026 for acute chest pain. MRN: 00384721. His wife, Mary Smith, "
        "was contacted at (555) 867-5309. Email: john.smith@email.com. "
        "He lives at 123 Oak Street, Springfield, IL 62704. "
        "The 95-year-old patient has a history of COPD and atrial fibrillation. "
        "Account Number: 8827364510. Member ID: BCBS12345678."
    )

    print("=" * 70)
    print("ORIGINAL TEXT:")
    print("=" * 70)
    print(sample)

    print("\n" + "=" * 70)
    print("DETECTED PHI:")
    print("=" * 70)
    detections = detect_phi(sample)
    for d in detections:
        print(f"  [{d['entity_type']}] \"{d['text']}\" "
              f"(score: {d['score']}, chars {d['start']}-{d['end']})")
        print(f"    → {d['description']}")

    print("\n" + "=" * 70)
    print("DE-IDENTIFIED TEXT:")
    print("=" * 70)
    redacted, _ = deidentify_text(sample)
    print(redacted)
