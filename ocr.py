"""
OCR Module for Clinical Note Image Processing

Extracts text from scanned clinical notes, handwritten prescriptions,
faxed documents, and PDF files using Tesseract OCR.

Supports: PNG, JPG, JPEG, TIFF, BMP, GIF, WEBP, PDF
"""

import io
import os
from typing import Tuple, Optional
from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance
import pytesseract


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply preprocessing to improve OCR accuracy on clinical documents.

    Steps:
      1. Convert to grayscale
      2. Increase contrast (helps with faded faxes)
      3. Sharpen (helps with blurry scans)
      4. Binarize with adaptive threshold (clean background noise)
    """
    # Convert to grayscale
    img = image.convert("L")

    # Boost contrast — faxed/scanned docs are often low-contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Sharpen — helps with slightly blurry images
    img = img.filter(ImageFilter.SHARPEN)

    # Binarize — convert to pure black/white for cleaner OCR
    threshold = 150
    img = img.point(lambda p: 255 if p > threshold else 0, mode="1")

    # Convert back to grayscale for Tesseract compatibility
    img = img.convert("L")

    return img


def extract_text_from_image(
    image: Image.Image,
    lang: str = "eng",
    preprocess: bool = True,
) -> Tuple[str, dict]:
    """
    Extract text from a PIL Image using Tesseract OCR.

    Args:
        image: PIL Image object.
        lang: Tesseract language code (default: English).
        preprocess: Whether to apply image preprocessing.

    Returns:
        (extracted_text, metadata_dict)
    """
    original_size = image.size

    if preprocess:
        processed = preprocess_image(image)
    else:
        processed = image.convert("L")

    # Use Tesseract with clinical-friendly config:
    #   --psm 6  = Assume a single uniform block of text
    #   --oem 3  = Use LSTM neural net engine
    custom_config = r"--oem 3 --psm 6"

    text = pytesseract.image_to_string(
        processed, lang=lang, config=custom_config
    )

    # Get confidence data
    try:
        data = pytesseract.image_to_data(
            processed, lang=lang, config=custom_config,
            output_type=pytesseract.Output.DICT,
        )
        confidences = [
            int(c) for c in data["conf"] if int(c) > 0
        ]
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0
        )
        word_count = len(confidences)
    except Exception:
        avg_confidence = 0
        word_count = 0

    # Clean up the extracted text
    cleaned = _clean_ocr_text(text)

    metadata = {
        "image_width": original_size[0],
        "image_height": original_size[1],
        "preprocessed": preprocess,
        "language": lang,
        "word_count": word_count,
        "avg_confidence": round(avg_confidence, 1),
        "raw_char_count": len(text),
        "cleaned_char_count": len(cleaned),
    }

    return cleaned, metadata


def extract_text_from_bytes(
    file_bytes: bytes,
    filename: str = "",
    lang: str = "eng",
    preprocess: bool = True,
) -> Tuple[str, dict]:
    """
    Extract text from raw file bytes. Handles images and PDFs.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used to detect format).
        lang: Tesseract language code.
        preprocess: Whether to apply image preprocessing.

    Returns:
        (extracted_text, metadata_dict)
    """
    ext = Path(filename).suffix.lower() if filename else ""

    # Handle PDFs
    if ext == ".pdf":
        return _extract_from_pdf(file_bytes, lang, preprocess)

    # Handle images
    image = Image.open(io.BytesIO(file_bytes))
    text, meta = extract_text_from_image(image, lang, preprocess)
    meta["source_type"] = "image"
    meta["source_format"] = image.format or ext.lstrip(".")
    return text, meta


def _extract_from_pdf(
    pdf_bytes: bytes,
    lang: str = "eng",
    preprocess: bool = True,
) -> Tuple[str, dict]:
    """Extract text from a PDF by converting pages to images first."""
    try:
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(pdf_bytes, dpi=300)
    except ImportError:
        return (
            "PDF support requires 'pdf2image' and 'poppler'. "
            "Install with: brew install poppler && pip install pdf2image",
            {"error": "pdf2image not available"},
        )
    except Exception as e:
        return (
            f"Failed to convert PDF to images: {e}",
            {"error": str(e)},
        )

    all_text = []
    total_words = 0
    total_confidence = 0
    confidence_count = 0

    for i, page_img in enumerate(images):
        page_text, page_meta = extract_text_from_image(
            page_img, lang, preprocess,
        )
        if page_text.strip():
            all_text.append(f"--- Page {i + 1} ---\n{page_text}")
            total_words += page_meta.get("word_count", 0)
            if page_meta.get("avg_confidence", 0) > 0:
                total_confidence += page_meta["avg_confidence"]
                confidence_count += 1

    combined_text = "\n\n".join(all_text)
    avg_conf = (
        total_confidence / confidence_count if confidence_count else 0
    )

    metadata = {
        "source_type": "pdf",
        "page_count": len(images),
        "total_word_count": total_words,
        "avg_confidence": round(avg_conf, 1),
        "cleaned_char_count": len(combined_text),
    }

    return combined_text, metadata


def _clean_ocr_text(text: str) -> str:
    """
    Clean OCR output to improve downstream NER accuracy.

    - Remove excessive blank lines
    - Strip trailing whitespace
    - Fix common OCR artifacts
    """
    lines = text.split("\n")
    cleaned_lines = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()

        # Skip consecutive blank lines
        if not stripped:
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
            continue

        prev_blank = False

        # Fix common OCR artifacts
        stripped = stripped.replace("|", "l")   # pipe → lowercase L
        stripped = stripped.replace("}", ")")   # common brace misread
        stripped = stripped.replace("{", "(")

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Verify Tesseract is accessible
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"❌ Tesseract not found: {e}")
        print("   Install with: brew install tesseract")
        exit(1)

    # Create a simple test image with text
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    test_text = "Patient has hypertension\nand was prescribed Lisinopril 20mg."

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font = ImageFont.load_default()

    draw.text((20, 40), test_text, fill="black", font=font)

    print("\n🔍 Running OCR on test image...")
    extracted, meta = extract_text_from_image(img, preprocess=True)
    print(f"\nExtracted text:\n  {extracted}")
    print(f"\nMetadata:\n  {meta}")
