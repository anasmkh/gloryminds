import argparse
from pathlib import Path

import fitz  
import pymupdf4llm


def extract_plain_text_fallback(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append(f"## Page {page_num}\n\n{text}")
    doc.close()
    return "\n\n".join(pages)


def convert_all(input_dir: Path, output_dir: Path, only: list[str] | None = None, force: bool = False) -> None:
    if only:
        pdf_files = [input_dir / rel for rel in only]
        missing = [str(p) for p in pdf_files if not p.exists()]
        if missing:
            raise SystemExit(f"These files were not found under {input_dir}:\n  " + "\n  ".join(missing))
    else:
        pdf_files = sorted(input_dir.rglob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found under {input_dir}. Check the path.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Converting...\n")

    successes, failures, skipped = 0, [], 0

    for pdf_path in pdf_files:
        relative = pdf_path.relative_to(input_dir) 
        md_relative = relative.with_suffix(".md")
        md_path = output_dir / md_relative

        if md_path.exists() and not force:
            print(f"  SKIP {relative}  (already converted, output exists)")
            skipped += 1
            continue

        md_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
            md_path.write_text(md_text, encoding="utf-8")
            print(f"  OK   {relative}  ->  {md_relative}")
            successes += 1
        except Exception as e:
            print(f"  WARN {relative}  (markdown extractor failed: {e}) -> trying plain-text fallback")
            try:
                fallback_text = extract_plain_text_fallback(pdf_path)
                if not fallback_text.strip():
                    raise ValueError("fallback extraction produced no text")
                md_path.write_text(fallback_text, encoding="utf-8")
                print(f"  OK   {relative}  ->  {md_relative}  (plain-text fallback)")
                successes += 1
            except Exception as e2:
                print(f"  FAIL {relative}  (fallback also failed: {e2})")
                failures.append(str(relative))

    print(f"\nDone. {successes} converted, {skipped} skipped (already done), {len(failures)} failed.")
    if failures:
        print("Failed files:")
        for f in failures:
            print(f"  - {f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert grade/subject PDFs to Markdown.")
    parser.add_argument(
        "--input",
        default="/Users/apple/Desktop/psy_bot_v2/raw_pdfs",
        help="Path to raw_pdfs folder",
    )
    parser.add_argument(
        "--output",
        default="/Users/apple/Desktop/psy_bot_v2/processed_data",
        help="Path to write processed_md folder",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional: only (re)convert these specific files, given as paths relative to --input. "
             "Use this to retry just the ones that failed, instead of the whole batch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert every file even if its .md output already exists. "
             "Without this flag, already-converted files are skipped automatically.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    convert_all(input_dir, output_dir, only=args.only, force=args.force)