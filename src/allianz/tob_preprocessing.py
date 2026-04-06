import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz
import pdfplumber

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "allianz"
CHUNKED_DIR = DATA_DIR / "chunked"

PLAN_NAMES = ["Care Base", "Care Enhanced", "Care Signature"]

SECTION_PATTERNS = [
    ("core_plans", re.compile(r"\bcore plans\b", re.I)),
    ("outpatient_plans", re.compile(r"\bout-?patient plans\b", re.I)),
    ("dental_plans", re.compile(r"\bdental plans\b", re.I)),
    ("area_of_cover", re.compile(r"\barea of cover\b", re.I)),
    ("deductibles", re.compile(r"\bdeductibles?\b", re.I)),
]


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_cell_text(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    text = " ".join(lines)
    text = re.sub(r"\s*/\s*", " / ", text)
    text = re.sub(r"\bCHF(?=\d)", "CHF ", text)
    text = re.sub(r"\bUS\$(?=\d)", "US$ ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def normalize_value(text: str) -> str:
    text = normalize_cell_text(text)
    if not text:
        return ""

    if text == "√":
        return "Covered in full"
    if text == "X":
        return "Not covered"

    text = text.replace("√", "Covered in full")
    text = re.sub(r"(?<![A-Za-z])X(?![A-Za-z])", "Not covered", text)
    return text.strip()


def extract_page_texts(pdf_path: str) -> Dict[int, str]:
    doc = fitz.open(pdf_path)
    page_texts = {}
    try:
        for i, page in enumerate(doc, start=1):
            page_texts[i] = clean_text(page.get_text("text"))
    finally:
        doc.close()
    return page_texts


def detect_section_from_page_text(page_text: str) -> str:
    txt = clean_text(page_text)
    for name, pattern in SECTION_PATTERNS:
        if pattern.search(txt):
            return name
    return "unknown"


def is_tob_section(section: str) -> bool:
    return section in {"core_plans", "outpatient_plans", "dental_plans"}


def is_subsection_text(text: str) -> bool:
    s = text.strip().lower()
    patterns = [
        r"^in-patient benefits$",
        r"^other benefits$",
        r"^additional core plan services$",
        r"^out-patient plan benefits$",
        r"^dental plan benefits$",
    ]
    return any(re.search(p, s) for p in patterns)


def is_noise_row(cells: List[str]) -> bool:
    merged = " ".join(cells).strip().lower()
    if not merged:
        return True

    noise_patterns = [
        r"^care base care enhanced care signature$",
        r"^core plans( \(cont\.\))?$",
        r"^out-patient plans( \(cont\.\))?$",
        r"^dental plans$",
        r"^maximum plan limit$",
        r"^core plans key to table of benefits",
        r"^looking for something specific\?$",
        r"^click here or press enter",
    ]
    return any(re.search(p, merged) for p in noise_patterns)


def looks_like_continuation_row(row: List[str]) -> bool:
    """
    benefit 없이 부가 설명/값만 이어지는 continuation row 판별
    """
    benefit = row[0].strip() if len(row) > 0 else ""
    condition = row[1].strip() if len(row) > 1 else ""
    base = row[2].strip() if len(row) > 2 else ""
    enhanced = row[3].strip() if len(row) > 3 else ""
    signature = row[4].strip() if len(row) > 4 else ""

    if benefit:
        return False

    return any([condition, base, enhanced, signature])


def merge_row_values(prev: Dict[str, Any], curr: List[str]) -> Dict[str, Any]:
    def join(a: str, b: str) -> str:
        a = a.strip()
        b = b.strip()
        if a and b:
            return f"{a} {b}".strip()
        return a or b

    prev["condition"] = join(prev["condition"], curr[1] if len(curr) > 1 else "")
    prev["plans"]["Care Base"] = join(prev["plans"]["Care Base"], curr[2] if len(curr) > 2 else "")
    prev["plans"]["Care Enhanced"] = join(prev["plans"]["Care Enhanced"], curr[3] if len(curr) > 3 else "")
    prev["plans"]["Care Signature"] = join(prev["plans"]["Care Signature"], curr[4] if len(curr) > 4 else "")
    return prev


def table_rows_from_pdfplumber(page) -> List[List[str]]:
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 4,
        "join_tolerance": 4,
        "edge_min_length": 15,
        "text_x_tolerance": 2,
        "text_y_tolerance": 2,
    }

    tables = page.extract_tables(table_settings=settings)
    if not tables:
        return []

    # 가장 큰 표 하나 사용
    best = max(tables, key=lambda t: len(t) * max((len(r) for r in t if r), default=0))

    rows = []
    for raw_row in best:
        if not raw_row:
            continue
        row = [normalize_cell_text(c) for c in raw_row]
        row = [c for c in row]  # keep length
        if any(c.strip() for c in row):
            rows.append(row)

    return rows


def normalize_to_5cols(row: List[str]) -> List[str]:
    """
    TOB는 기본적으로 5열 구조를 기대:
    benefit / condition / base / enhanced / signature
    """
    row = row[:5] + [""] * max(0, 5 - len(row))
    return row[:5]


def parse_tob_page(page, page_num: int, section: str) -> List[Dict[str, Any]]:
    rows = table_rows_from_pdfplumber(page)
    if not rows:
        return []

    parsed = []
    subsection = ""
    row_idx = 0

    for raw in rows:
        row = normalize_to_5cols(raw)

        if is_noise_row(row):
            continue

        # subsection row
        if row[0] and is_subsection_text(row[0]) and not any(row[1:]):
            subsection = row[0]
            continue

        # continuation row면 직전 row에 병합
        if parsed and looks_like_continuation_row(row):
            parsed[-1] = merge_row_values(parsed[-1], row)
            continue

        benefit = row[0].strip()
        condition = row[1].strip()

        # benefit이 없고 plan만 있으면 이전 row continuation 취급
        if not benefit and parsed:
            parsed[-1] = merge_row_values(parsed[-1], row)
            continue

        # 정상 row 생성
        record = {
            "page": page_num,
            "section": section,
            "subsection": subsection,
            "row_index_in_page": row_idx,
            "benefit": benefit,
            "notes": "",
            "condition": normalize_value(condition),
            "plans": {
                "Care Base": normalize_value(row[2]),
                "Care Enhanced": normalize_value(row[3]),
                "Care Signature": normalize_value(row[4]),
            },
        }

        # benefit 내부에 설명문 포함 시 분리
        benefit_lines = [ln.strip() for ln in benefit.split("\n") if ln.strip()]
        if len(benefit_lines) > 1:
            record["benefit"] = benefit_lines[0]
            record["notes"] = " ".join(benefit_lines[1:])

        parsed.append(record)
        row_idx += 1

    # 후처리: 비정상 row 제거
    cleaned = []
    for row in parsed:
        benefit = row["benefit"].strip().lower()
        if benefit in {"required", "covered in full", "not covered", "√", "x", ""}:
            continue
        cleaned.append(row)

    return cleaned


def structured_row_to_text(row: Dict[str, Any]) -> str:
    lines = []

    if row.get("subsection"):
        lines.append(f"Subsection: {row['subsection']}")

    lines.append(f"Benefit: {row['benefit']}")

    if row.get("notes"):
        lines.append(f"Notes: {row['notes']}")

    if row.get("condition"):
        lines.append(f"Condition: {row['condition']}")

    for plan in PLAN_NAMES:
        val = row["plans"].get(plan, "")
        if val:
            lines.append(f"{plan}: {val}")

    return "\n".join(lines)


def build_chunk_record(chunk_id: str, row: Dict[str, Any], doc_id: str, source_file: str) -> Dict[str, Any]:
    text = "\n".join([
        f"[SECTION] {row['section']}",
        f"[PAGE] {row['page']}",
        f"[SUBSECTION] {row.get('subsection', '')}",
        "",
        structured_row_to_text(row),
    ]).strip()

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "source_file": source_file,
        "page_start": row["page"],
        "page_end": row["page"],
        "section": row["section"],
        "subsection": row.get("subsection", ""),
        "row_count": 1,
        "benefit_names": [row["benefit"]],
        "text": text,
        "rows_structured": [row],
    }


def extract_and_chunk_tables_for_rag(pdf_path: str, output_dir: str = "./rag_chunks") -> List[Dict[str, Any]]:
    pdf_path = str(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_file = Path(pdf_path).name
    doc_id = Path(pdf_path).stem

    page_texts = extract_page_texts(pdf_path)
    chunk_records = []
    chunk_seq = 1

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            section = detect_section_from_page_text(page_texts.get(page_idx, ""))

            if not is_tob_section(section):
                continue

            rows = parse_tob_page(page, page_idx, section)

            for row in rows:
                if not row.get("benefit"):
                    continue

                chunk_id = f"{doc_id}_p{page_idx}_c{chunk_seq:04d}"
                chunk_records.append(
                    build_chunk_record(
                        chunk_id=chunk_id,
                        row=row,
                        doc_id=doc_id,
                        source_file=source_file,
                    )
                )
                chunk_seq += 1

    jsonl_path = output_dir / f"{doc_id}_table_chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in chunk_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] saved: {jsonl_path}")
    print(f"[INFO] total chunks: {len(chunk_records)}")
    return chunk_records


if __name__ == "__main__":
    pdf_path = DATA_DIR / "care-tob-en_보장금액.pdf"
    output_dir = CHUNKED_DIR

    chunks = extract_and_chunk_tables_for_rag(pdf_path, output_dir)

    for ch in chunks:
        if ch["page_start"] == 8:
            print("=" * 100)
            print(ch["chunk_id"])
            print(ch["text"])