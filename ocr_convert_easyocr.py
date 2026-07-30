import argparse
from pathlib import Path

import easyocr
import fitz  
import numpy as np

DPI = 300  


def ocr_pdf(pdf_path: Path, lang: str) -> str:
    reader = easyocr.Reader([lang], gpu=False) 

    doc = fitz.open(str(pdf_path))
    zoom = DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    pages_text = []
    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        results = reader.readtext(img, detail=0, paragraph=True)
        text = "\n".join(results)
        pages_text.append(f"## Page {page_num}\n\n{text}")
        print(f"  OCR page {page_num}/{doc.page_count} ({len(text)} chars)")

    doc.close()
    return "\n\n".join(pages_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw_pdfs folder")
    parser.add_argument("--output", required=True, help="Path to processed_data folder")
    parser.add_argument("--file", required=True, help="Relative path (from --input) to the PDF to OCR")
    parser.add_argument("--lang", required=True, help="EasyOCR language code: ar, ru, en, fr, etc.")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    pdf_path = input_dir / args.file
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {pdf_path}")

    md_relative = Path(args.file).with_suffix(".md")
    md_path = output_dir / md_relative
    md_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"OCR'ing {args.file} (lang={args.lang}, {DPI} DPI) with EasyOCR...")
    text = ocr_pdf(pdf_path, args.lang)
    md_path.write_text(text, encoding="utf-8")
    print(f"\nDone. Wrote {len(text)} chars to {md_path}")