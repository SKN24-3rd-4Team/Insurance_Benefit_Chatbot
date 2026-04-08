"""
tricare_ingest.py
─────────────────
TRICARE 벡터 DB 구축 전용 스크립트

위치: src/tricare/tricare_ingest.py

경로는 모두 이 파일 위치 기준으로 자동 계산됨.
DATA_DIR, DB 경로 등을 수동으로 수정할 필요 없음.

프로젝트 구조 (자동 인식):
    INSURANCE_BENEFIT_CHATBOT/     <- BASE_DIR (루트)
    ├── data/raw/tricare/          <- DATA_DIR  (PDF/CSV 여기)
    ├── src/tricare/
    │   └── tricare_ingest.py      <- 이 파일
    ├── vectordb/tricare/          <- DB 저장 위치
    └── .env

실행 방법 (src/tricare/ 폴더 안에서 OR 어디서든):
    python src/tricare/tricare_ingest.py --reload   # 전체 재구축 (권장)
    python src/tricare/tricare_ingest.py            # 증분 추가
"""

from __future__ import annotations

import os
import re
import csv
import shutil
import argparse
from math import ceil
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz          # PyMuPDF  (pip install pymupdf)
import pdfplumber    # (pip install pdfplumber)
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─── 경로 설정 (팀원 allianz/ingest.py와 동일한 패턴) ────────────────
# src/tricare/tricare_ingest.py
#   .parent       → src/tricare/
#   .parent.parent → src/
#   .parent.parent.parent → INSURANCE_BENEFIT_CHATBOT/  (루트)

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 프로젝트 루트

DATA_DIR     = BASE_DIR / "data" / "raw" / "tricare"      # PDF/CSV 위치
DB_TEXT_DIR  = BASE_DIR / "vectordb" / "tricare" / "tricare_text"   # 텍스트+CSV DB
DB_TABLE_DIR = BASE_DIR / "vectordb" / "tricare" / "tricare_table"  # 표 전용 DB

COLLECTION_TEXT  = "tricare_rag"
COLLECTION_TABLE = "tricare_cost_tables"

load_dotenv(dotenv_path=BASE_DIR / ".env")


# ─── 임베딩 설정 ──────────────────────────────────────────────────
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DEVICE     = os.getenv("EMBED_DEVICE", "cpu")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "100"))

MIN_CHUNK_CHARS = 80   # PDF 단락 최소 길이 — 헤더/페이지번호 노이즈 제거


# ─── 파일 목록 ────────────────────────────────────────────────────
# location : "OCONUS" → 전 페이지 사용
#            "BOTH"   → OCONUS_KEYWORDS 포함 페이지만 사용
# table    : True → chroma_db2(표 저장소)에도 pdfplumber로 추가

RAG_PDF_FILES: List[Dict[str, Any]] = [
    {
        "path": DATA_DIR / "Costs_Fees.pdf",
        "location": "BOTH",
        "table": True,
    },
    {
        "path": DATA_DIR / "Overseas_HB(해외 프로그램 안내서).pdf",
        "location": "OCONUS",
        "table": False,
    },
    {
        "path": DATA_DIR / "Pharmacy_HB(tricare 약국 프로그램 안내서).pdf",
        "location": "BOTH",
        "table": True,
    },
    {
        "path": DATA_DIR / "TOP_Handbook_AUG_2023_FINAL_092223_508 (1).pdf",
        "location": "OCONUS",
        "table": False,
    },
    {
        "path": DATA_DIR / "NGR_HB(국가방위군 및 예비군을 위한 트라이케어 안내서).pdf",
        "location": "BOTH",
        "table": True,
    },
    {
        "path": DATA_DIR / "TFL_HB(평생 트라이케어).pdf",
        "location": "BOTH",
        "table": True,
    },
    {
        "path": DATA_DIR / "TRICARE_ADDP_HB_FINAL_508c(현역 군인 치과 프로그램 안내서).pdf",
        "location": "BOTH",
        "table": False,
    },
    {
        "path": DATA_DIR / "ADDP_Brochure_FINAL_122624_508c.pdf",
        "location": "BOTH",
        "table": False,
    },
    {
        "path": DATA_DIR / "Maternity_Br (1).pdf",
        "location": "BOTH",
        "table": False,
    },
    {
        "path": DATA_DIR / "Retiring_NGR_Br.pdf",
        "location": "BOTH",
        "table": False,
    },
    {
        "path": DATA_DIR / "QLEs_FS (2).pdf",
        "location": "BOTH",
        "table": False,
    },
    {
        "path": DATA_DIR / "Medicare_Turning_65_Br.pdf",
        "location": "BOTH",
        "table": False,
    },
    {
        "path": DATA_DIR / "Medicare_Under_65_Br_7.pdf",
        "location": "BOTH",
        "table": False,
    },
    {
        "path": DATA_DIR / "Plans_Overview_FS_1.pdf",
        "location": "BOTH",
        "table": False,
    },
    {
        "path": DATA_DIR / "Retiring_AD_Br.pdf",
        "location": "BOTH",
        "table": False,
    },
]

CSV_FILES: Dict[str, Path] = {
    "mental":     DATA_DIR / "mental_health_services.csv",
    "costs":      DATA_DIR / "Health_Plan_Costs.csv",
    "plans":      DATA_DIR / "TricarePlans.csv",
    "exclusions": DATA_DIR / "tricare_exclusions.csv",
}


# ─── 전처리 유틸 ──────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """PDF 특수문자/중복 공백 정제"""
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_cell(cell: str) -> str:
    """표 셀 체크마크 → Covered / Not covered 변환"""
    if not cell:
        return "N/A"
    cell = cell.strip()
    if cell in ["✓", "√", "v", "V", "●", "Yes", "yes", "Y"]:
        return "Covered"
    if cell in ["✗", "×", "x", "X", "✘", "No", "no", "N"]:
        return "Not covered"
    return cell


def is_noise_line(line: str) -> bool:
    """반복 헤더/URL 등 노이즈 줄 판별"""
    noise_patterns = [
        r"^group a.*group b",
        r"^covered service.*group",
        r"^are you in group",
        r"^this is an overview",
        r"^visit www\.tricare",
        r"^for more information.*go to",
        r"^updated (january|february|march|april|may|june|july|august|"
        r"september|october|november|december)",
    ]
    line_lower = line.lower().strip()
    return any(re.match(pat, line_lower) for pat in noise_patterns)


OCONUS_KEYWORDS = [
    "overseas", "oconus", "outside the continental",
    "south korea", "korea", "usfk",
    "outside the u.s.", "outside the united states",
    "international", "host nation", "foreign country",
    "tricare prime overseas", "tricare select overseas",
    "tricare prime remote overseas",
    "near patient", "overseas claim",
]


def is_oconus_relevant(text: str) -> bool:
    """BOTH 문서에서 해외/한국 키워드 포함 페이지만 통과"""
    return any(kw in text.lower() for kw in OCONUS_KEYWORDS)


TRICARE_SEARCH_TAGS = [
    "coverage", "covered", "not covered", "benefit", "cost", "copay",
    "deductible", "enrollment fee", "cost-share", "prior authorization",
    "TRICARE Prime", "TRICARE Select", "TRICARE For Life",
    "Group A", "Group B", "active duty", "retiree", "reserve",
    "OCONUS", "overseas", "USFK", "주한미군",
    "보장", "비용", "본인부담금", "사전승인", "해외",
]


def enrich_tricare_text(text: str, location: str) -> str:
    """
    본문 뒤에 [search_tags] 블록 추가.
    임베딩엔 포함되지만 LLM 전달 시엔 tricare_core.format_docs()가 제거함.
    """
    tags = "\n".join([
        "[search_tags]",
        "insurer: TRICARE",
        f"location: {location}",
        "keywords: " + ", ".join(TRICARE_SEARCH_TAGS),
    ])
    return f"{text}\n\n{tags}"


# ─── 파일 존재 여부 사전 확인 ──────────────────────────────────────

def check_files() -> None:
    """실행 전 PDF/CSV 파일 존재 여부 출력"""
    print("=" * 60)
    print("[ 경로 확인 ]")
    print(f"  BASE_DIR : {BASE_DIR}")
    print(f"  DATA_DIR : {DATA_DIR}")
    print(f"  DB_TEXT  : {DB_TEXT_DIR}")
    print(f"  DB_TABLE : {DB_TABLE_DIR}")

    print("\n  [ RAG PDF ]")
    for f in RAG_PDF_FILES:
        status = "OK     " if f["path"].exists() else "MISSING"
        print(f"  [{status}] {f['path'].name}")

    print("\n  [ CSV ]")
    for key, path in CSV_FILES.items():
        status = "OK     " if path.exists() else "MISSING"
        print(f"  [{status}] {path.name}")
    print("=" * 60 + "\n")


# ─── PDF 로드 + 청킹 ──────────────────────────────────────────────

def load_pdf_chunks() -> List[Document]:
    all_chunks: List[Document] = []
    total_skipped = 0

    for file_info in RAG_PDF_FILES:
        path: Path    = file_info["path"]
        location: str = file_info["location"]
        fname         = path.name

        if not path.exists():
            print(f"[WARN] 파일 없음, 건너뜀: {fname}")
            continue

        doc_fitz   = fitz.open(str(path))
        kept_pages: List[Document] = []
        skipped    = 0

        for i, page in enumerate(doc_fitz):
            raw = clean_text(page.get_text("text"))
            if not raw:
                skipped += 1
                continue
            if location == "BOTH" and not is_oconus_relevant(raw):
                skipped += 1
                continue

            lines   = [ln for ln in raw.split("\n") if not is_noise_line(ln)]
            cleaned = "\n".join(lines)

            kept_pages.append(Document(
                page_content=cleaned,
                metadata={
                    "source_file": fname,
                    "source":      str(path),
                    "page":        i + 1,
                    "location":    location,
                    "insurer":     "TRICARE",
                }
            ))

        doc_fitz.close()
        total_skipped += skipped
        print(
            f"[INFO] {fname} [{location}]: "
            f"{len(kept_pages) + skipped}p → {len(kept_pages)}p 사용"
        )

        for doc in kept_pages:
            paragraphs = [
                p.strip() for p in doc.page_content.split("\n\n")
                if len(p.strip()) >= MIN_CHUNK_CHARS
            ]
            if not paragraphs and len(doc.page_content.strip()) >= MIN_CHUNK_CHARS:
                paragraphs = [doc.page_content.strip()]

            for para in paragraphs:
                all_chunks.append(Document(
                    page_content=enrich_tricare_text(para, location),
                    metadata=doc.metadata.copy()
                ))

    print(f"\n[INFO] PDF 총 청크: {len(all_chunks)}개 | 제외 페이지: {total_skipped}개")
    return all_chunks


# ─── CSV 로드 + 청킹 ─────────────────────────────────────────────

def _load_csv_mental(path: Path) -> List[Document]:
    docs: List[Document] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            service = row.get("service", row.get("서비스", "")).strip()
            content = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if not content:
                continue
            docs.append(Document(
                page_content=f"서비스: {service}\n내용: {content}",
                metadata={"source_file": path.name, "source": str(path),
                          "service_name": service, "type": "mental_health", "insurer": "TRICARE"}
            ))
    return docs


def _load_csv_costs(path: Path) -> List[Document]:
    docs: List[Document] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if not content:
                continue
            docs.append(Document(
                page_content=content,
                metadata={"source_file": path.name, "source": str(path),
                          "type": "cost", "insurer": "TRICARE"}
            ))
    return docs


def _load_csv_plans(path: Path) -> List[Document]:
    docs: List[Document] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            plan_name = row.get("plan_name", row.get("플랜명", "")).strip()
            content   = " | ".join(f"{k}: {v}" for k, v in row.items() if v and v.strip())
            if not content:
                continue
            docs.append(Document(
                page_content=f"플랜명: {plan_name}\n내용: {content}",
                metadata={"source_file": path.name, "source": str(path),
                          "plan_name": plan_name, "type": "plan", "insurer": "TRICARE"}
            ))
    return docs


def _load_csv_exclusions(path: Path) -> List[Document]:
    docs: List[Document] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name    = row.get("name", row.get("항목", "")).strip()
            content = row.get("content", row.get("내용", "")).strip()
            url     = row.get("url", "").strip()
            if not content:
                continue
            docs.append(Document(
                page_content=f"제외항목: {name}\n내용: {content}",
                metadata={"source_file": path.name, "source": str(path),
                          "exclusion_name": name, "url": url,
                          "type": "exclusion", "insurer": "TRICARE"}
            ))
    return docs


def load_csv_chunks() -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    loaders = {
        "mental":     (_load_csv_mental,     CSV_FILES["mental"]),
        "costs":      (_load_csv_costs,      CSV_FILES["costs"]),
        "plans":      (_load_csv_plans,      CSV_FILES["plans"]),
        "exclusions": (_load_csv_exclusions, CSV_FILES["exclusions"]),
    }

    all_chunks: List[Document] = []
    for key, (loader_fn, path) in loaders.items():
        if not path.exists():
            print(f"[WARN] CSV 없음, 건너뜀: {path.name}")
            continue
        raw_docs = loader_fn(path)
        chunks   = text_splitter.split_documents(raw_docs)
        all_chunks.extend(chunks)
        print(f"[INFO] {path.name}: {len(raw_docs)}행 → {len(chunks)}청크")

    print(f"[INFO] CSV 총 청크: {len(all_chunks)}개")
    return all_chunks


# ─── 표 전용 청킹 (pdfplumber) ────────────────────────────────────

def load_table_chunks() -> List[Document]:
    all_chunks: List[Document] = []

    for file_info in RAG_PDF_FILES:
        if not file_info.get("table"):
            continue

        path: Path = file_info["path"]
        fname      = path.name
        if not path.exists():
            print(f"[WARN] 파일 없음, 건너뜀: {fname}")
            continue

        table_count = 0
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for table in (page.extract_tables() or []):
                    if not table:
                        continue
                    rows_text = [
                        " | ".join(normalize_cell(str(c or "")) for c in row)
                        for row in table
                    ]
                    table_text = "\n".join(rows_text)
                    if len(table_text.strip()) < 30:
                        continue
                    all_chunks.append(Document(
                        page_content=table_text,
                        metadata={
                            "source_file": fname,
                            "source":      str(path),
                            "page":        page_num,
                            "type":        "table",
                            "location":    file_info["location"],
                            "insurer":     "TRICARE",
                        }
                    ))
                    table_count += 1

        print(f"[INFO] {fname}: 표 {table_count}개 추출")

    # Health_Plan_Costs.csv도 표 저장소에 추가
    costs_path = CSV_FILES["costs"]
    if costs_path.exists():
        costs_docs = _load_csv_costs(costs_path)
        all_chunks.extend(costs_docs)
        print(f"[INFO] Health_Plan_Costs.csv: {len(costs_docs)}행 → 표 저장소 추가")

    print(f"[INFO] 표 총 청크: {len(all_chunks)}개")
    return all_chunks


# ─── 임베딩 / 벡터 저장소 ─────────────────────────────────────────

def build_embeddings() -> HuggingFaceEmbeddings:
    print(f"[INFO] 임베딩 모델 로드: {EMBED_MODEL_NAME} (device={EMBED_DEVICE})")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBED_BATCH_SIZE},
    )


def reset_vectordbs() -> None:
    for db_dir in [DB_TEXT_DIR, DB_TABLE_DIR]:
        if db_dir.exists():
            print(f"[INFO] 기존 벡터DB 삭제: {db_dir}")
            shutil.rmtree(db_dir, ignore_errors=True)


def _index_to_store(
    documents: List[Document], persist_directory: Path,
    collection_name: str, embeddings: HuggingFaceEmbeddings, batch_size: int,
) -> Chroma | None:
    if not documents:
        print(f"[WARN] 인덱싱할 문서 없음: {collection_name}")
        return None

    vectordb    = None
    num_batches = ceil(len(documents) / batch_size)

    for i in range(num_batches):
        start = i * batch_size
        end   = min(start + batch_size, len(documents))
        batch = documents[start:end]
        print(f"[INFO] [{collection_name}] 배치 {i+1}/{num_batches} ({start}~{end-1})")

        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(persist_directory),
                collection_name=collection_name,
            )
        else:
            vectordb.add_documents(batch)

    if vectordb:
        try:
            vectordb.persist()
        except Exception:
            pass  # 최신 Chroma는 자동 persist

    return vectordb


def build_vectorstores(
    reload: bool = False, batch_size: int = INDEX_BATCH_SIZE,
) -> Tuple[Chroma | None, Chroma | None]:
    if reload:
        reset_vectordbs()

    pdf_chunks = load_pdf_chunks()
    csv_chunks = load_csv_chunks()
    text_docs  = pdf_chunks + csv_chunks
    table_docs = load_table_chunks()

    print(f"\n[INFO] 텍스트 저장소 총 청크: {len(text_docs)}개")
    print(f"[INFO] 표 저장소 총 청크:     {len(table_docs)}개")

    embeddings = build_embeddings()

    vs_text = _index_to_store(
        text_docs, DB_TEXT_DIR, COLLECTION_TEXT, embeddings, batch_size
    )
    vs_table = _index_to_store(
        table_docs, DB_TABLE_DIR, COLLECTION_TABLE, embeddings, batch_size
    )

    print("\n" + "=" * 60)
    print("[DONE] 벡터 저장소 구축 완료")
    if vs_text:
        print(f"  tricare_text  (tricare_rag):         {vs_text._collection.count()}개")
    if vs_table:
        print(f"  tricare_table (tricare_cost_tables): {vs_table._collection.count()}개")
    print(f"  저장 위치: {BASE_DIR / 'vectordb' / 'tricare'}")
    print("=" * 60)

    return vs_text, vs_table


# ─── 메인 ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TRICARE RAG 벡터스토어 구축")
    parser.add_argument("--reload", action="store_true",
                        help="기존 DB 삭제 후 전체 재구축 (권장)")
    args = parser.parse_args()

    check_files()
    build_vectorstores(reload=args.reload, batch_size=INDEX_BATCH_SIZE)


if __name__ == "__main__":
    main()
