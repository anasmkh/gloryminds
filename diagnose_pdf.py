"""
Diagnose why a PDF produced no extractable text: is it a scan (image-only),
encrypted, or something else?

Usage:
    python diagnose_pdf.py --input raw_pdfs --files "path/relative/to/input.pdf" "another.pdf"
"""

import argparse
from pathlib import Path

import fitz


def diagnose(pdf_path: Path):
    print(f"\n=== {pdf_path.name} ===")
    doc = fitz.open(str(pdf_path))

    print(f"  Encrypted: {doc.is_encrypted}")
    print(f"  Page count: {doc.page_count}")

    total_text_chars = 0
    total_images = 0
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        images = page.get_images()
        total_text_chars += len(text)
        total_images += len(images)
        if page_num <= 3:  # sample first few pages
            print(f"  Page {page_num}: {len(text)} text chars, {len(images)} image(s)")

    print(f"  TOTAL: {total_text_chars} text chars, {total_images} images across {doc.page_count} pages")

    if total_text_chars == 0 and total_images > 0:
        print("  -> Looks like a SCANNED/image-only PDF. Needs OCR.")
    elif total_text_chars == 0 and total_images == 0:
        print("  -> No text AND no images found. File may be corrupted or blank.")
    else:
        print("  -> Has some text; investigate further if extraction still failed.")

    doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw_pdfs folder")
    parser.add_argument("--files", nargs="+", required=True, help="Relative paths to the PDFs to check")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    for rel in args.files:
        diagnose(input_dir / rel)