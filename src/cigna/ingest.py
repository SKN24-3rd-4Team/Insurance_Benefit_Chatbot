"""
ingest.py — Cigna RAG 파이프라인 · 데이터 수집 및 벡터스토어 구축
Section 0 ~ Section 2 (Section 3 리트리버 구성 이전)

실행 방법:
    python ingest.py          # PDF 파싱 + 벡터스토어 신규 구축
    python ingest.py --reload # 기존 Chroma DB 삭제 후 재구축

모듈로 import할 경우:
    from ingest import vectorstore_latest, vectorstore_all, latest_chunks, embed_baai
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List

import pdfplumber
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── 경로 설정 ───────────────────────────────────────────────────
BASE_DIR  = Path('.')
CIGNA_DIR = BASE_DIR / 'Cigna'

CHROMA_LATEST_DIR = './chroma_baai_latest'
CHROMA_ALL_DIR    = './chroma_baai_all'


# ══════════════════════════════════════════════════════════════════
# Section 1 · PDF 파싱 유틸리티
# ══════════════════════════════════════════════════════════════════

CHECKMARK_GLYPHS = {'\uf0fc', '\uf0b7', '✓', '✔'}

_CHECK = {'\uf0fc', '\uf0b7', '\u2713', '\u2714', '\u2611', '\u2705'}
_CROSS = {'\u2297', '\u2717', '\u2718', '\u274c', '\u00d7', '\u2612', '\u29bb', '\uf078'}
_BADGE = re.compile(r'^(Updated|New\b|\d+\s*MONTHS?)', re.IGNORECASE)


def has_monetary_value(row) -> bool:
    """행에 금액·퍼센트 값이 있으면 데이터 행으로 판단 (헤더 행 공백 오판 방지)."""
    for cell in row:
        if cell and any(c in str(cell) for c in ['$', '€', '£', '%', 'Paid in full', 'N/A']):
            return True
    return False


def format_multicurrency(text: str) -> str:
    """다통화 셀: '$25,000\\n€18,500\\n£16,500' → 'USD $25,000 / EUR €18,500 / GBP £16,500'"""
    currency_map = {'$': 'USD', '€': 'EUR', '£': 'GBP'}
    parts = [p.strip() for p in text.split('\n') if p.strip()]
    if len(parts) <= 1:
        return text
    labeled = []
    for part in parts:
        for sym, label in currency_map.items():
            if part.startswith(sym):
                labeled.append(f'{label} {part}')
                break
        else:
            labeled.append(part)
    return ' / '.join(labeled)


def clean_table_row(row: list) -> str:
    """표 행 정제: 체크마크→Covered, 데이터행 빈셀→Not covered, 다통화 포맷."""
    is_data_row = has_monetary_value(row)
    cleaned = []
    for cell in row:
        if cell is None:
            continue
        cell_str = str(cell).strip()
        # 회전된 섹션 라벨 제거 (전부 대문자 + 6자 이상)
        if cell_str.replace('\n', '').isupper() and len(cell_str.replace('\n', '')) > 6:
            continue
        has_ck = any(g in cell_str for g in CHECKMARK_GLYPHS)
        remaining = cell_str
        for g in CHECKMARK_GLYPHS:
            remaining = remaining.replace(g, '').strip()
        if has_ck:
            cleaned.append(f'Covered - {remaining}' if remaining else 'Covered')
        elif cell_str == '':
            if is_data_row:
                cleaned.append('Not covered')
        elif cell_str:
            cleaned.append(format_multicurrency(cell_str))
    return ' | '.join(cleaned)


def _cvt(cell, is_data: bool) -> str:
    if cell is None or not str(cell).strip():
        return 'Not Covered' if is_data else ''
    t = str(cell).strip().replace('\n', ' ')
    chk = any(g in t for g in _CHECK)
    crs = any(g in t for g in _CROSS)
    clean = t
    for g in _CHECK | _CROSS:
        clean = clean.replace(g, '')
    clean = clean.strip()
    if crs:
        return 'Not Covered'
    if chk:
        return f"Covered{' - ' + clean if clean else ''}"
    return clean or ('Not Covered' if is_data else '')


def _is_data(row) -> bool:
    return any(
        c and re.search(r'[$\u20ac\xa3\d]|covered|paid|n/a|refund|no coverage', str(c), re.I)
        for c in row
    )


def _is_rotated(text) -> bool:
    t = (text or '').strip()
    return bool(t) and '\n' in t and t.replace('\n', '').replace(' ', '').isupper() and len(t) > 6


def _clean_benefit(text, extra_badges=()) -> str:
    if not text and not extra_badges:
        return ''
    parts = [p.strip() for p in re.split(r'[\n/]+', text or '') if p.strip()]
    if len(parts) <= 1:
        badges  = list(extra_badges)
        content = [text.strip()] if text and text.strip() else []
    else:
        badges  = [p for p in parts if _BADGE.match(p)] + list(extra_badges)
        content = [p for p in parts if not _BADGE.match(p)]
    result = ' '.join(content)
    if badges:
        result = f"{result} ({' '.join(badges)})" if result else f"({' '.join(badges)})"
    return result.strip()


def _col_map(table):
    """Benefits Summary 표에서 Silver/Gold/Platinum 열 인덱스를 찾는다."""
    s = g = p = hdr = None
    for i, row in enumerate(table[:8]):
        v = [str(c or '').split('\n')[0].strip() for c in row]
        if 'Silver' in v and 'Gold' in v:
            s, g = v.index('Silver'), v.index('Gold')
            p = next((j for j, x in enumerate(v) if x == 'Platinum'), None)
            hdr = i
            break
    if s is None:
        return None
    b = max(0, s - 1)
    for row in table[hdr + 1:]:
        if not _is_data(row):
            continue
        for ci in range(s - 1, -1, -1):
            v = str(row[ci] or '').strip()
            if v and not _is_rotated(v):
                b = ci
                break
        break
    return {'b': b, 's': s, 'g': g, 'p': p}


def _table_to_md(table) -> str:
    """pdfplumber 표 → 마크다운 테이블 문자열 변환."""
    cm = _col_map(table)
    lines = []
    hdr_done = False
    for row in table:
        data = _is_data(row)
        if cm:
            n      = len(row)
            b_raw  = str(row[cm['b']] or '').strip() if cm['b'] < n else ''
            mid_badges = []
            for ci in range(cm['b'] + 1, cm['s']):
                v = str(row[ci] or '').strip()
                if v and _BADGE.match(v):
                    mid_badges.append(v)
                elif v:
                    b_raw = f'{b_raw} {v}'.strip()
            s_raw = row[cm['s']] if cm['s'] < n else None
            g_raw = row[cm['g']] if cm['g'] < n else None
            p_raw = row[cm['p']] if cm['p'] and cm['p'] < n else None
            if not any(str(v or '').strip() for v in [s_raw, g_raw, p_raw]):
                data = False
            b_txt = _clean_benefit(b_raw, mid_badges)
            cells = [b_txt, _cvt(s_raw, data), _cvt(g_raw, data)]
            if cm['p']:
                cells.append(_cvt(p_raw, data))
        else:
            cells = [_cvt(c, data) for c in row]
        if not any(str(c).strip() for c in cells):
            continue
        line = '| ' + ' | '.join(str(c) for c in cells) + ' |'
        lines.append(line)
        if not hdr_done:
            lines.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')
            hdr_done = True
    return '\n'.join(lines)


# ── PDF 메타데이터 정의 ─────────────────────────────────────────
PDF_META = [
    {'path': 'Cigna/Customer_Guide/200008 CGHO Customer Guide EN_05_2019.pdf',
     'source_type': 'customer_guide', 'doc_version': '2019', 'is_latest': False, 'plan_type': 'all'},
    {'path': 'Cigna/Customer_Guide/591048 CGHO Customer Guide EN_05_2022.pdf',
     'source_type': 'customer_guide', 'doc_version': '2022', 'is_latest': False, 'plan_type': 'all'},
    {'path': 'Cigna/Customer_Guide/591048-cgho-customer-guide-en_05_2023.pdf',
     'source_type': 'customer_guide', 'doc_version': '2023', 'is_latest': False, 'plan_type': 'all'},
    {'path': 'Cigna/Customer_Guide/Cigna-Global-Health-Options-Customer-Guide_02_2026.pdf',
     'source_type': 'customer_guide', 'doc_version': '2026', 'is_latest': True, 'plan_type': 'all'},
    {'path': 'Cigna/Policy_Rules/200008 CGHO Customer Guide EN_05_2019.pdf',
     'source_type': 'policy_rules', 'doc_version': '2019', 'is_latest': False, 'plan_type': 'all'},
    {'path': 'Cigna/Policy_Rules/CGHO Policy Rules CGIC NA_EN_05_2023.pdf',
     'source_type': 'policy_rules', 'doc_version': '2023', 'is_latest': False, 'plan_type': 'all'},
    {'path': 'Cigna/Policy_Rules/CGHO Policy Rules CGIC_EN_02_2024.pdf',
     'source_type': 'policy_rules', 'doc_version': '2024', 'is_latest': False, 'plan_type': 'all'},
    {'path': 'Cigna/Policy_Rules/CGHO Policy Rules CGIC_EN_02_2025.pdf',
     'source_type': 'policy_rules', 'doc_version': '2025', 'is_latest': False, 'plan_type': 'all'},
    {'path': 'Cigna/Policy_Rules/CGHP Policy Rules CGIC EN 02_2026.pdf',
     'source_type': 'policy_rules', 'doc_version': '2026', 'is_latest': True, 'plan_type': 'all'},
    {'path': 'Cigna/Benefits_Summary/591116 Cigna_Global_International_Health_Plans_Benefits_Summary_USD_EN_0523.pdf',
     'source_type': 'benefits_summary', 'doc_version': '2023-05', 'is_latest': False, 'plan_type': 'all'},
    {'path': 'Cigna/Benefits_Summary/591116 Cigna Global Benefits Summary USD_EN_0924.pdf',
     'source_type': 'benefits_summary', 'doc_version': '2024-09', 'is_latest': False, 'plan_type': 'all'},
    {'path': 'Cigna/Benefits_Summary/591116-cigna-global-benefits-summary-usd_en_02_2025.pdf',
     'source_type': 'benefits_summary', 'doc_version': '2025-02', 'is_latest': True, 'plan_type': 'all'},
]


# ── PDF 로더 ────────────────────────────────────────────────────
def load_pdf(meta: dict) -> List[Document]:
    """모든 PDF 통합 로더 — pdfplumber로 텍스트+표 동시 추출."""
    docs = []
    try:
        with pdfplumber.open(meta['path']) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text      = page.extract_text() or ''
                tables    = page.extract_tables()
                table_mds = [md for t in tables if (md := _table_to_md(t))]
                combined  = text
                if table_mds:
                    combined += '\n\n[TABLE]\n' + '\n\n'.join(table_mds)
                if combined.strip():
                    docs.append(Document(
                        page_content=combined,
                        metadata={**meta, 'page': page_num, 'file_name': Path(meta['path']).name}
                    ))
    except Exception as e:
        print(f'  ⚠ 오류 {meta["path"]}: {e}')
    return docs


# ── 텍스트 스플리터 ─────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=['\n\n', '\n', '. ', ' ', ''],
)


# ══════════════════════════════════════════════════════════════════
# Section 2 · 임베딩 + 벡터스토어
# ══════════════════════════════════════════════════════════════════

# BAAI/bge-m3: 100개 이상 언어 지원 다국어 임베딩 (dense 768-dim)
embed_baai = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device': 'cpu'},        # GPU 사용 시 'cuda' 로 변경
    encode_kwargs={'normalize_embeddings': True},
)


def build_vectorstores(reload: bool = False):
    """PDF를 로드하고 Chroma 벡터스토어 2개를 구축한다.

    Args:
        reload: True면 기존 Chroma DB 삭제 후 재구축

    Returns:
        (vectorstore_latest, vectorstore_all, all_chunks, latest_chunks)
    """
    import shutil

    if reload:
        for d in [CHROMA_LATEST_DIR, CHROMA_ALL_DIR]:
            if Path(d).exists():
                shutil.rmtree(d)
                print(f'  🗑 기존 DB 삭제: {d}')

    all_raw_docs: List[Document] = []
    for meta in PDF_META:
        path = Path(meta['path'])
        if not path.exists():
            print(f'  ⚠ 파일 없음: {path}')
            continue
        docs = load_pdf(meta)
        all_raw_docs.extend(docs)
        print(f'  ✅ [{meta["source_type"]} {meta["doc_version"]}] {len(docs)} pages')
    print(f'\n총 {len(all_raw_docs)} pages 로드 완료')

    all_chunks_: List[Document]    = text_splitter.split_documents(all_raw_docs)
    latest_chunks_: List[Document] = [c for c in all_chunks_ if c.metadata.get('is_latest', False)]
    print(f'전체 청크: {len(all_chunks_)}  최신 청크: {len(latest_chunks_)}')

    vs_latest = Chroma.from_documents(
        documents=latest_chunks_, embedding=embed_baai,
        collection_name='cigna_latest', persist_directory=CHROMA_LATEST_DIR,
    )
    vs_all = Chroma.from_documents(
        documents=all_chunks_, embedding=embed_baai,
        collection_name='cigna_all', persist_directory=CHROMA_ALL_DIR,
    )
    print('✅ 벡터스토어 구축 완료')
    return vs_latest, vs_all, all_chunks_, latest_chunks_


def load_vectorstores():
    """디스크에 저장된 Chroma DB를 로드한다 (빠른 재시작용)."""
    vs_latest = Chroma(
        collection_name='cigna_latest',
        embedding_function=embed_baai,
        persist_directory=CHROMA_LATEST_DIR,
    )
    vs_all = Chroma(
        collection_name='cigna_all',
        embedding_function=embed_baai,
        persist_directory=CHROMA_ALL_DIR,
    )
    return vs_latest, vs_all


def update_vectorstore_latest(vectorstore_latest_, vectorstore_all_,new_pdf_meta: dict) -> None:
    """신규 PDF가 나왔을 때 vectorstore_latest를 업데이트한다."""
    new_type = new_pdf_meta['source_type']

    existing = vectorstore_latest_.get(
        where={'$and': [{'source_type': {'$eq': new_type}},
                        {'is_latest': {'$eq': True}}]}
    )
    old_ids = existing['ids']
    if old_ids:
        vectorstore_latest_.delete(ids=old_ids)
        print(f'  🗑 기존 최신 청크 {len(old_ids)}개 삭제 ({new_type})')

    for m in PDF_META:
        if m['source_type'] == new_type and m['is_latest']:
            m['is_latest'] = False
    new_pdf_meta['is_latest'] = True
    PDF_META.append(new_pdf_meta)

    new_raw    = load_pdf(new_pdf_meta)
    new_chunks = text_splitter.split_documents(new_raw)
    vectorstore_latest_.add_documents(new_chunks)
    vectorstore_all_.add_documents(new_chunks)  # all에도 신규 버전 추가

    print(f'  ✅ 신규 최신 청크 {len(new_chunks)}개 추가 ({new_type} {new_pdf_meta["doc_version"]})')
    print(f'  ✅ vectorstore_all에도 {len(new_chunks)}개 추가')



# ══════════════════════════════════════════════════════════════════
# Section 2-A · 코사인 유사도 헬퍼
# ══════════════════════════════════════════════════════════════════

def cosine_sim(text1: str, text2: str) -> float:
    """두 텍스트 간 코사인 유사도 (BAAI/bge-m3). 반환값: 0.0 ~ 1.0"""
    if not text1.strip() or not text2.strip():
        return 0.0
    vecs = embed_baai.embed_documents([text1, text2])
    a, b = np.array(vecs[0]), np.array(vecs[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def print_sim(question: str, answer: str, label: str = '') -> float:
    """유사도 계산 후 출력 + 반환."""
    sim     = cosine_sim(question, answer)
    bar_len = int(sim * 20)
    bar     = '█' * bar_len + '░' * (20 - bar_len)
    tag     = label + ' ' if label else ''
    level   = '🟢 우수' if sim >= 0.75 else ('🟡 보통' if sim >= 0.55 else '🔴 낮음')
    print(f'  {tag}질문↔답변 코사인 유사도: {sim:.4f}  [{bar}]  {level}')
    return sim


# ══════════════════════════════════════════════════════════════════
# 모듈 레벨 초기화 (import 시 자동 로드)
# ══════════════════════════════════════════════════════════════════

if Path(CHROMA_LATEST_DIR).exists() and Path(CHROMA_ALL_DIR).exists():
    # 이미 구축된 DB가 있으면 디스크에서 로드
    vectorstore_latest, vectorstore_all = load_vectorstores()
    _raw = vectorstore_latest.get()
    latest_chunks = [
        Document(page_content=pc, metadata=meta)
        for pc, meta in zip(_raw['documents'], _raw['metadatas'])
    ]
    _raw_all = vectorstore_all.get()
    all_chunks = [
        Document(page_content=pc, metadata=meta)
        for pc, meta in zip(_raw_all['documents'], _raw_all['metadatas'])
    ]
    print(f'✅ 기존 벡터스토어 로드 완료 (latest: {len(latest_chunks)}, all: {len(all_chunks)})')
else:
    print('벡터스토어 없음 → PDF 파싱 및 구축 시작...')
    vectorstore_latest, vectorstore_all, all_chunks, latest_chunks = build_vectorstores()


# ── CLI 진입점 ──────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cigna RAG 벡터스토어 구축')
    parser.add_argument('--reload', action='store_true', help='기존 DB 삭제 후 재구축')
    args = parser.parse_args()
    vectorstore_latest, vectorstore_all, all_chunks, latest_chunks = build_vectorstores(
        reload=args.reload
    )
    print(f'\n최신 청크: {len(latest_chunks)} / 전체 청크: {len(all_chunks)}')
