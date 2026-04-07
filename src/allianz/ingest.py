from __future__ import annotations

import os
import re
import json
import shutil
import argparse
from math import ceil
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
import torch
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "raw" / "allianz"
CHUNKED_DIR = DATA_DIR / "chunked"

DB_LATEST_DIR = BASE_DIR / "vectordb" / 'allianz' / "allianz_latest"
DB_ALL_DIR = BASE_DIR / "vectordb" / 'allianz' / "allianz_all"

COLLECTION_LATEST = "allianz_latest"
COLLECTION_ALL = "allianz_all"

ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# 실행 옵션
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "100"))
RESET_VECTORDB = os.getenv("RESET_VECTORDB", "true").lower() == "true"

TORCH_NUM_THREADS = int(
    os.getenv("TORCH_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
)

torch.set_num_threads(TORCH_NUM_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# 파일 목록
# 같은 문서군(doc_group) 안에서는 최신 버전 1개만 is_latest=True가 되도록 관리
FILES: List[Dict[str, Any]] = [
    # 글로벌 공통
    {
        "path": DATA_DIR / "DOC-Care-IBG-EN-1125_개인고객용혜택가이드.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "global_benefit_guide",
        "doc_year": 2025,
        "region": "global",
        "product_family": "care_global",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "care-tob-en_보장금액.pdf",
        "doc_type": "tob",
        "doc_group": "global_tob",
        "doc_year": 2025,
        "region": "global",
        "product_family": "care_global",
        "is_latest": True,
        "chunked_path": CHUNKED_DIR / "care-tob-en_보장금액_table_chunks.jsonl",
    },
    {
        "path": DATA_DIR / "FRM-PreAuth-EN-0825_사전승인신청서.pdf",
        "doc_type": "preauth_form",
        "doc_group": "global_preauth_form",
        "doc_year": 2025,
        "region": "global",
        "product_family": "care_global",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "FRM-PCF-EN-1125_사후보험청구서.pdf",
        "doc_type": "claim_form",
        "doc_group": "global_claim_form",
        "doc_year": 2025,
        "region": "global",
        "product_family": "care_global",
        "is_latest": True,
    },

    # 지역별 코퍼스
    {
        "path": DATA_DIR / "DOC-Singapore-IBG-EN-0126_싱가포르.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "singapore_benefit_guide",
        "doc_year": 2026,
        "region": "singapore",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-IBG-Dubai-Northern-Emirates-EN-0126_두바이.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "dubai_northern_emirates_benefit_guide",
        "doc_year": 2026,
        "region": "dubai_northern_emirates",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-LEBANON-IBG-EN-0725_레바논.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "lebanon_benefit_guide",
        "doc_year": 2025,
        "region": "lebanon",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-IBG-Indonesia-en-UK-1123_인도네시아.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "indonesia_benefit_guide",
        "doc_year": 2024,
        "region": "indonesia",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-IBG-Vietnam-en-UK-0823_베트남.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "vietnam_benefit_guide",
        "doc_year": 2023,
        "region": "vietnam",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-IBG-HongKong-en-UK-2024_홍콩.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "hong_kong_benefit_guide",
        "doc_year": 2023,
        "region": "hong_kong",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-IBG-AZJD-en-UK-0824_중국.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "china_benefit_guide",
        "doc_year": 2024,
        "region": "china",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-SUISSE-IBG-KPT-EN-0624_스위스.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "switzerland_benefit_guide",
        "doc_year": 2022,
        "region": "switzerland",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-IBG-CARE-UK-EN-1125_영국.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "uk_benefit_guide",
        "doc_year": 2025,
        "region": "uk",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-IBG-FP-en-UK-1223_프랑스.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "france_benelux_monaco_benefit_guide",
        "doc_year": 2024,
        "region": "france_benelux_monaco",
        "product_family": "regional",
        "is_latest": True,
    },
    {
        "path": DATA_DIR / "DOC-Global-IBG-EN-0524_남미.pdf",
        "doc_type": "benefit_guide",
        "doc_group": "latin_america_benefit_guide",
        "doc_year": 2024,
        "region": "latin_america",
        "product_family": "regional",
        "is_latest": True,
    },
]


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_pdf_pages(pdf_path: Path) -> List[tuple[int, str]]:
    if not pdf_path.exists():
        print(f"[WARN] 파일이 없습니다: {pdf_path}")
        return []

    pages: List[tuple[int, str]] = []
    doc = fitz.open(pdf_path)

    try:
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append((i + 1, text))
    finally:
        doc.close()

    return pages


def build_common_metadata(
    file_info: Dict[str, Any],
    source_name: str,
    page_num: int,
    chunk_idx: int | None = None,
    section: str | None = None,
) -> Dict[str, Any]:
    metadata = {
        "source": source_name,
        "doc_type": file_info["doc_type"],
        "doc_group": file_info.get("doc_group", f'{file_info["region"]}_{file_info["doc_type"]}'),
        "doc_year": file_info["doc_year"],
        "region": file_info["region"],
        "product_family": file_info["product_family"],
        "page": page_num,
        "insurer": "Allianz",
        "is_latest": file_info.get("is_latest", False),
    }

    if chunk_idx is not None:
        metadata["chunk_idx"] = chunk_idx

    if section:
        metadata["section"] = section

    return metadata


REGION_ALIASES = {
    "global": ["global", "worldwide", "전세계", "글로벌", "공통"],
    "singapore": ["singapore", "싱가포르"],
    "dubai_northern_emirates": ["dubai", "northern emirates", "두바이", "북부에미리트", "uae", "아랍에미리트"],
    "lebanon": ["lebanon", "레바논"],
    "indonesia": ["indonesia", "인도네시아"],
    "vietnam": ["vietnam", "베트남"],
    "hong_kong": ["hong kong", "hk", "홍콩"],
    "china": ["china", "중국", "중화권"],
    "switzerland": ["switzerland", "suisse", "스위스"],
    "uk": ["uk", "united kingdom", "england", "britain", "영국"],
    "france_benelux_monaco": ["france", "benelux", "monaco", "프랑스", "모나코", "베네룩스"],
    "latin_america": ["latin america", "남미", "라틴아메리카"],
}

DOC_TYPE_ALIASES = {
    "benefit_guide": [
        "benefit guide", "coverage guide", "benefits", "혜택 가이드", "보장 안내", "보장", "혜택"
    ],
    "tob": [
        "table of benefits", "schedule of benefits", "benefit limits",
        "보장금액", "보장표", "한도표", "한도", "limit"
    ],
    "preauth_form": [
        "pre-authorisation form", "preauthorization form", "preauth form",
        "사전승인 신청서", "사전승인", "입원 전 승인", "직접청구 준비"
    ],
    "claim_form": [
        "claim form", "reimbursement form",
        "보험금 청구서", "청구서", "환급 청구", "사후 청구"
    ],
}

INSURANCE_SEARCH_TAGS = [
    "coverage", "covered", "benefit", "limit", "co-payment", "copay",
    "deductible", "waiting period", "exclusion", "outpatient", "inpatient",
    "maternity", "cancer", "chronic condition", "pre-existing condition",
    "pre-authorisation", "preauthorization", "planned hospitalisation",
    "direct billing", "claim", "reimbursement", "invoice", "receipt",
    "서류", "청구", "환급", "직접청구", "사전승인", "보장", "혜택",
    "한도", "면책", "제외사항", "외래", "입원", "출산", "기왕증"
]


def build_search_tags(file_info: Dict[str, Any]) -> str:
    region_aliases = REGION_ALIASES.get(file_info["region"], [])
    doc_type_aliases = DOC_TYPE_ALIASES.get(file_info["doc_type"], [])

    return "\n".join([
        "[search_tags]",
        f"region: {' | '.join(region_aliases)}",
        f"doc_type: {' | '.join(doc_type_aliases)}",
        f"product_family: {file_info['product_family']}",
        f"doc_year: {file_info['doc_year']}",
        f"is_latest: {file_info.get('is_latest', False)}",
        "insurer: Allianz 알리안츠",
        "keywords: " + ", ".join(INSURANCE_SEARCH_TAGS),
    ])


def enrich_text_for_multilingual_search(text: str, file_info: Dict[str, Any]) -> str:
    return f"{text}\n\n{build_search_tags(file_info)}"


def chunk_benefit_guide(
    pages: List[tuple[int, str]],
    source_name: str,
    file_info: Dict[str, Any],
) -> List[Document]:
    docs: List[Document] = []

    for page_num, text in pages:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]

        for idx, para in enumerate(paragraphs):
            docs.append(
                Document(
                    page_content=enrich_text_for_multilingual_search(para, file_info),
                    metadata=build_common_metadata(
                        file_info=file_info,
                        source_name=source_name,
                        page_num=page_num,
                        chunk_idx=idx,
                    ),
                )
            )
    return docs


def normalize_form_line(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*:\s*", ": ", text)
    return text.strip()


def is_form_noise_line(line: str) -> bool:
    s = line.strip().lower()
    if not s:
        return True

    noise_patterns = [
        r"^dd\s*/\s*mm\s*/\s*yyyy$",
        r"^d d\s*/\s*m m\s*/\s*y y y y$",
        r"^country code$",
        r"^area code$",
        r"^country\s+code$",
        r"^area\s+code$",
        r"^yes no$",
        r"^yes\s+no$",
        r"^mr\.?\s+mrs\.?\s+ms\.?\s+miss\.?\s+other$",
        r"^official stamp of medical provider$",
    ]
    return any(re.search(p, s) for p in noise_patterns)


def is_form_section_header(line: str) -> bool:
    s = line.strip().lower()

    section_patterns = [
        r"^\d+\s+patient[’']?s details",
        r"^\d+\s+medical details",
        r"^\d+\s+treatment details",
        r"^\d+\s+patient details",
        r"^\d+\s+your personal data",
        r"^\d+\s+declaration",
        r"^patient[’']?s details$",
        r"^medical details$",
        r"^treatment details$",
        r"^your personal data$",
        r"^declaration$",
        r"^medical provider details$",
        r"^treatment$",
        r"^costs$",
        r"^applicable to cases of pregnancy only:?$",
        r"^please also provide the following details for maternity cases$",
    ]
    return any(re.search(p, s) for p in section_patterns)


def clean_form_field(line: str) -> str:
    s = normalize_form_line(line)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" -•\t")


def summarize_form_section(section: str, fields: List[str], source_name: str) -> str:
    field_text = ", ".join(fields)
    section_l = section.lower()

    if "patient" in section_l and "detail" in section_l:
        summary = (
            "This section describes the patient information required in the form, "
            "including identity, policy, birth date, and contact details."
        )
    elif "medical" in section_l and "detail" in section_l:
        summary = (
            "This section describes the medical information required in the form, "
            "including symptoms, diagnosis, condition history, ICD/DSM code, recurrence, "
            "rehabilitation, permanence, monitoring needs, and related treatment details."
        )
    elif "treatment" in section_l:
        summary = (
            "This section describes treatment information required for review, "
            "including planned procedure, admission date, diagnosis-related codes, "
            "length of stay, and estimated treatment costs."
        )
    elif "medical provider details" in section_l:
        summary = (
            "This section describes the medical provider information required, "
            "including hospital or facility details, doctor details, and contact information."
        )
    elif "cost" in section_l:
        summary = (
            "This section describes cost-related information required in the form, "
            "including package price, hospital charges, doctor or anaesthetist fees, "
            "and total estimated costs."
        )
    elif "declaration" in section_l:
        summary = (
            "This section describes the declaration and signature requirements, "
            "including confirmation of accuracy, consent, and signing responsibilities."
        )
    elif "personal data" in section_l:
        summary = (
            "This section describes personal data and consent requirements, "
            "including privacy notice, consent for medical data processing, and related obligations."
        )
    elif "pregnancy" in section_l or "maternity" in section_l:
        summary = (
            "This section describes pregnancy or maternity-related information required, "
            "including delivery date, single or multiple birth status, and assisted reproduction details."
        )
    else:
        summary = (
            "This section describes information required in the form and the fields that must be completed."
        )

    return "\n".join([
        f"Form document: {source_name}",
        f"Section: {section}",
        summary,
        f"Fields included: {field_text}",
    ])


def chunk_form(
    pages: List[tuple[int, str]],
    source_name: str,
    file_info: Dict[str, Any],
) -> List[Document]:
    docs: List[Document] = []

    current_section = "General"
    current_fields: List[str] = []
    current_page = 1
    chunk_idx = 0

    def flush_section():
        nonlocal current_fields, chunk_idx, current_section, current_page

        unique_fields: List[str] = []
        seen = set()

        for field in current_fields:
            field_norm = field.strip().lower()
            if not field_norm:
                continue
            if field_norm in seen:
                continue
            seen.add(field_norm)
            unique_fields.append(field)

        if not unique_fields:
            current_fields = []
            return

        content = summarize_form_section(
            section=current_section,
            fields=unique_fields,
            source_name=source_name,
        )
        content += "\nForm purpose: insurance claim or pre-authorisation document."

        docs.append(
            Document(
                page_content=enrich_text_for_multilingual_search(content, file_info),
                metadata=build_common_metadata(
                    file_info=file_info,
                    source_name=source_name,
                    page_num=current_page,
                    chunk_idx=chunk_idx,
                    section=current_section,
                ),
            )
        )

        chunk_idx += 1
        current_fields = []

    for page_num, text in pages:
        lines = [normalize_form_line(line) for line in text.split("\n")]
        lines = [line for line in lines if line and not is_form_noise_line(line)]

        for line in lines:
            if is_form_section_header(line):
                flush_section()
                current_section = clean_form_field(line)
                current_page = page_num
                continue

            cleaned = clean_form_field(line)

            if len(cleaned) < 2:
                continue

            if len(cleaned) > 180:
                continue

            current_fields.append(cleaned)

    flush_section()
    return docs


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] JSONL 파일이 없습니다: {path}")
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSONL 파싱 실패: {path} line={line_no} | {e}")

    return rows


def chunk_tob_jsonl(
    jsonl_path: Path,
    source_name: str,
    file_info: Dict[str, Any],
) -> List[Document]:
    docs: List[Document] = []
    items = load_jsonl(jsonl_path)

    for idx, item in enumerate(items):
        text = clean_text(item.get("text", ""))
        if not text:
            continue

        page_num = item.get("page_start") or item.get("page") or 0
        source_file = item.get("source_file") or source_name

        metadata = build_common_metadata(
            file_info=file_info,
            source_name=source_file,
            page_num=page_num,
            chunk_idx=idx,
            section=item.get("section"),
        )

        metadata.update(
            {
                "chunk_id": item.get("chunk_id"),
                "doc_id": item.get("doc_id"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "subsection": item.get("subsection"),
                "row_count": item.get("row_count"),
                "benefit_names": item.get("benefit_names", []),
            }
        )

        docs.append(
            Document(
                page_content=enrich_text_for_multilingual_search(text, file_info),
                metadata=metadata,
            )
        )

    return docs


def build_documents(file_list: List[Dict[str, Any]] | None = None) -> List[Document]:
    all_docs: List[Document] = []
    target_files = file_list or FILES

    for file_info in target_files:
        path: Path = file_info["path"]
        source_name = path.name
        doc_type = file_info["doc_type"]

        print(
            f"[INFO] 처리 중: {source_name} | type={doc_type} | "
            f"region={file_info['region']} | year={file_info['doc_year']} | "
            f"is_latest={file_info.get('is_latest', False)}"
        )

        if doc_type == "tob":
            jsonl_path: Path | None = file_info.get("chunked_path")

            if not jsonl_path:
                print(f"[WARN] TOB chunked_path가 없습니다: {source_name}")
                continue

            docs = chunk_tob_jsonl(jsonl_path, source_name, file_info)
            all_docs.extend(docs)
            continue

        if not path.exists():
            print(f"[WARN] 파일이 없습니다: {path}")
            continue

        pages = read_pdf_pages(path)
        if not pages:
            continue

        if doc_type == "benefit_guide":
            docs = chunk_benefit_guide(pages, source_name, file_info)
        elif doc_type in ["preauth_form", "claim_form"]:
            docs = chunk_form(pages, source_name, file_info)
        else:
            print(f"[WARN] 지원하지 않는 doc_type: {doc_type}")
            continue

        all_docs.extend(docs)

    return all_docs


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBED_BATCH_SIZE,
        },
    )


def reset_vectordbs_if_needed() -> None:
    if RESET_VECTORDB:
        for db_dir in [DB_LATEST_DIR, DB_ALL_DIR]:
            if db_dir.exists():
                print(f"[INFO] 기존 벡터DB 삭제: {db_dir}")
                shutil.rmtree(db_dir, ignore_errors=True)


def _index_to_single_store(
    documents: List[Document],
    persist_directory: Path,
    collection_name: str,
    embeddings: HuggingFaceEmbeddings,
    batch_size: int,
) -> Chroma | None:
    if not documents:
        print(f"[WARN] 인덱싱할 문서가 없습니다: {collection_name}")
        return None

    vectordb = None
    total = len(documents)
    num_batches = ceil(total / batch_size)

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, total)
        batch_docs = documents[start:end]

        print(f"[INFO] [{collection_name}] Embedding batch {i + 1}/{num_batches} ({start}~{end})")

        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                persist_directory=str(persist_directory),
                collection_name=collection_name,
            )
        else:
            vectordb.add_documents(batch_docs)

    if vectordb is not None:
        try:
            vectordb.persist()
        except Exception:
            pass

    return vectordb


def build_vectorstores(
    reload: bool = False,
    batch_size: int = INDEX_BATCH_SIZE,
) -> Tuple[Chroma | None, Chroma | None, List[Document], List[Document]]:
    if reload:
        reset_vectordbs_if_needed()

    documents = build_documents()
    latest_documents = [d for d in documents if d.metadata.get("is_latest", False)]

    print(f"[INFO] Total chunks(all): {len(documents)}")
    print(f"[INFO] Total chunks(latest): {len(latest_documents)}")
    print(
        f"[INFO] EMBED_MODEL_NAME={EMBED_MODEL_NAME}, "
        f"EMBED_DEVICE={EMBED_DEVICE}, "
        f"EMBED_BATCH_SIZE={EMBED_BATCH_SIZE}, "
        f"INDEX_BATCH_SIZE={batch_size}, "
        f"TORCH_NUM_THREADS={TORCH_NUM_THREADS}"
    )

    embeddings = build_embeddings()

    vectorstore_latest = _index_to_single_store(
        documents=latest_documents,
        persist_directory=DB_LATEST_DIR,
        collection_name=COLLECTION_LATEST,
        embeddings=embeddings,
        batch_size=batch_size,
    )

    vectorstore_all = _index_to_single_store(
        documents=documents,
        persist_directory=DB_ALL_DIR,
        collection_name=COLLECTION_ALL,
        embeddings=embeddings,
        batch_size=batch_size,
    )

    print("[DONE] latest/all 벡터스토어 구축 완료")
    return vectorstore_latest, vectorstore_all, documents, latest_documents


def load_vectorstores() -> Tuple[Chroma, Chroma]:
    embeddings = build_embeddings()

    vectorstore_latest = Chroma(
        persist_directory=str(DB_LATEST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_LATEST,
    )
    vectorstore_all = Chroma(
        persist_directory=str(DB_ALL_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_ALL,
    )
    return vectorstore_latest, vectorstore_all


def update_vectorstore_latest(
    vectorstore_latest: Chroma,
    vectorstore_all: Chroma,
    new_file_info: Dict[str, Any],
) -> Tuple[List[Document], List[Document]]:
    """
    신규 문서 버전이 들어왔을 때:
    - 같은 doc_group의 기존 latest 문서는 latest DB에서 제거
    - FILES 내부의 기존 latest는 False로 변경
    - 신규 문서는 is_latest=True로 추가
    - latest DB / all DB 모두에 신규 chunk 추가
    """
    new_file_info = dict(new_file_info)
    doc_group = new_file_info.get(
        "doc_group",
        f'{new_file_info["region"]}_{new_file_info["doc_type"]}'
    )
    new_file_info["doc_group"] = doc_group
    new_file_info["is_latest"] = True

    # 1) 기존 latest 삭제
    existing = vectorstore_latest.get(
        where={
            "$and": [
                {"doc_group": {"$eq": doc_group}},
                {"is_latest": {"$eq": True}},
            ]
        }
    )
    old_ids = existing.get("ids", [])
    if old_ids:
        vectorstore_latest.delete(ids=old_ids)
        print(f"[INFO] 기존 latest 청크 삭제: {len(old_ids)}개 | doc_group={doc_group}")

    # 2) FILES 내부 기존 latest 해제
    for item in FILES:
        item_group = item.get("doc_group", f'{item["region"]}_{item["doc_type"]}')
        if item_group == doc_group and item.get("is_latest", False):
            item["is_latest"] = False

    FILES.append(new_file_info)

    # 3) 신규 문서 청킹
    new_docs = build_documents([new_file_info])
    if not new_docs:
        print("[WARN] 신규 문서에서 생성된 청크가 없습니다.")
        return [], []

    # 4) latest/all에 추가
    vectorstore_latest.add_documents(new_docs)
    vectorstore_all.add_documents(new_docs)

    try:
        vectorstore_latest.persist()
    except Exception:
        pass

    try:
        vectorstore_all.persist()
    except Exception:
        pass

    print(
        f"[DONE] 신규 최신 버전 반영 완료 | "
        f"doc_group={doc_group} | year={new_file_info['doc_year']} | chunks={len(new_docs)}"
    )
    return new_docs, new_docs


def get_documents_from_store(vectorstore: Chroma) -> List[Document]:
    raw = vectorstore.get(include=["documents", "metadatas"])
    docs = []
    for content, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
        docs.append(Document(page_content=content, metadata=meta or {}))
    return docs


def main():
    parser = argparse.ArgumentParser(description="Allianz RAG 벡터스토어 구축")
    parser.add_argument("--reload", action="store_true", help="기존 latest/all DB 삭제 후 재구축")
    args = parser.parse_args()

    vs_latest, vs_all, all_docs, latest_docs = build_vectorstores(
        reload=args.reload,
        batch_size=INDEX_BATCH_SIZE,
    )

    print(f"[INFO] 전체 청크 수(all): {len(all_docs)}")
    print(f"[INFO] 최신 청크 수(latest): {len(latest_docs)}")

    if vs_latest is None or vs_all is None:
        print("[WARN] 벡터스토어 생성이 완전하지 않을 수 있습니다.")


if __name__ == "__main__":
    main()