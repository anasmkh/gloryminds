import argparse
import json
import re
import unicodedata
import uuid
from pathlib import Path


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)



GRADE_FOLDER_MAP = {
    "07-الصف السابع": "7th",
    "08-الصف الثامن": "8th",
    "09-الصف التاسع": "9th",
    "10-الصف الأول الثانوي": "10th",
    "11-الصف الثاني الثانوي": "11th",
    "12-الصف الثالث الثانوي": "12th",
}



SUBJECT_BASE_MAP = {
    "التاريخ سورية القديمة": "history",
    "التاريخ سورية الحضارة": "history",
    "التاريخ سورية الحديثة": "history",
    "التربية الدينية الإسلامية": "islamic_education",
    "التربية الدينية المسيحية": "christian_education",
    "التربية الفنية البصرية والجمالية": "visual_arts",
    "التربية الموسيقية": "music",
    "التربية الوطنية": "national_education",
    "الریاضیات": "math",
    "الفيزياء والكيمياء": "physics_chemistry",
    "اللغة الانكليزية": "english",
    "اللغة الروسية": "russian",
    "اللغة العربية": "arabic",
    "اللغة الفرنسية": "french",
    "تكنلوجيا المعلومات والاتصالات": "ict",
    "عالم الجغرافية": "geography",
    "علم الأحياء والأرض": "biology_earth_science",
    "عربي": "arabic", 
}

SUBTOPIC_MAP = {
    "الجبر": "algebra",
    "الهندسة": "geometry",
}
BOOK_TYPE_MAP = {
    "كتاب الأنشطة": "activity_book",
    "كتاب الطالب": "student_book",
}

GRADE_WORDS = ["السابع", "الثامن", "التاسع"]


GRADE_FOLDER_MAP = {nfc(k): v for k, v in GRADE_FOLDER_MAP.items()}
SUBJECT_BASE_MAP = {nfc(k): v for k, v in SUBJECT_BASE_MAP.items()}
SUBTOPIC_MAP = {nfc(k): v for k, v in SUBTOPIC_MAP.items()}
BOOK_TYPE_MAP = {nfc(k): v for k, v in BOOK_TYPE_MAP.items()}
GRADE_WORDS = [nfc(w) for w in GRADE_WORDS]


def normalize_subject(filename_stem: str) -> dict:
   
    raw = nfc(filename_stem.strip())

    cleaned = raw
    for gw in GRADE_WORDS:
        cleaned = re.sub(rf"الصف\s+{gw}", "", cleaned)
    cleaned = re.sub(r"\s*-\s*-\s*", " - ", cleaned)  
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -").strip()

    book_type = None
    for k, v in BOOK_TYPE_MAP.items():
        if k in cleaned:
            book_type = v
            cleaned = cleaned.replace(k, "").strip(" -").strip()

    subtopic = None
    for k, v in SUBTOPIC_MAP.items():
        if k in cleaned:
            subtopic = v
            cleaned = cleaned.replace(k, "").strip(" -").strip()

   
    chapter_hint = None
    m = re.search(r"الفصل\s+\S+", cleaned)
    if m:
        chapter_hint = m.group(0)
        cleaned = cleaned.replace(chapter_hint, "").strip(" -").strip()

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -").strip()

    subject_slug = None
    for key in sorted(SUBJECT_BASE_MAP.keys(), key=len, reverse=True):
        if cleaned.startswith(key):
            subject_slug = SUBJECT_BASE_MAP[key]
            break

    if subject_slug is None:
        print(f"  WARN: could not map subject for filename part: '{cleaned}' (raw: '{raw}')")
        subject_slug = "unknown"

    return {
        "subject": subject_slug,
        "subject_raw": raw,
        "subtopic": subtopic,
        "book_type": book_type,
        "chapter_hint": chapter_hint,
    }




MAX_CHARS = 1500  

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def split_into_sections(md_text: str):

    sections = []
    heading_stack = []  
    buffer_lines = []

    def flush():
        text = "\n".join(buffer_lines).strip()
        if text:
            path = [title for _, title in heading_stack]
            sections.append((path, text))
        buffer_lines.clear()

    for line in md_text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack = [(lvl, t) for lvl, t in heading_stack if lvl < level]
            heading_stack.append((level, title))
        else:
            buffer_lines.append(line)

    flush()
    return sections


def split_large_section(text: str, max_chars: int = MAX_CHARS):
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def process_file(md_path: Path, input_root: Path):
    relative = md_path.relative_to(input_root)
    grade_folder = nfc(relative.parts[0])
    grade = GRADE_FOLDER_MAP.get(grade_folder)
    if grade is None:
        print(f"  WARN: unrecognized grade folder '{grade_folder}', skipping {relative}")
        return []

    subject_info = normalize_subject(md_path.stem)
    md_text = md_path.read_text(encoding="utf-8")
    sections = split_into_sections(md_text)

    chunks = []
    for heading_path, section_text in sections:
        pieces = split_large_section(section_text)
        for i, piece in enumerate(pieces):
    
            id_source = f"{relative}::{'>'.join(heading_path)}::{i}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, id_source))

            chunk = {
                "id": chunk_id,
                "text": piece,
                "metadata": {
                    "grade": grade,
                    "subject": subject_info["subject"],
                    "subject_raw": subject_info["subject_raw"],
                    "subtopic": subject_info["subtopic"],
                    "book_type": subject_info["book_type"],
                    "chapter_hint": subject_info["chapter_hint"],
                    "heading_path": heading_path,
                    "chapter_title": heading_path[-1] if heading_path else None,
                    "source_file": str(relative),
                    "chunk_part": i,
                },
            }
            chunks.append(chunk)
    return chunks


def main(input_dir: Path, output_path: Path):
    md_files = sorted(input_dir.rglob("*.md"))
    if not md_files:
        print(f"No .md files found under {input_dir}.")
        return

    print(f"Found {len(md_files)} markdown file(s). Chunking...\n")

    all_chunks = []
    for md_path in md_files:
        file_chunks = process_file(md_path, input_dir)
        all_chunks.extend(file_chunks)
        print(f"  {md_path.relative_to(input_dir)}  ->  {len(file_chunks)} chunk(s)")

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    from collections import Counter
    counts = Counter((c["metadata"]["grade"], c["metadata"]["subject"]) for c in all_chunks)
    print(f"\nDone. {len(all_chunks)} total chunks written to {output_path}\n")
    print("Chunks per grade/subject:")
    for (grade, subject), n in sorted(counts.items()):
        print(f"  {grade:>4}  {subject:<25}  {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk markdown files and tag with grade/subject metadata.")
    parser.add_argument(
        "--input",
        default="/Users/apple/Desktop/psy_bot_v2/processed_data",
        help="Path to processed_data folder (output of Step 1)",
    )
    parser.add_argument(
        "--output",
        default="/Users/apple/Desktop/psy_bot_v2/chunks.jsonl",
        help="Path to write the chunks JSONL file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    main(input_dir, output_path)