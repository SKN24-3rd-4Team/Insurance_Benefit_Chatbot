"""
tricare_core.py
TRICARE RAG 핵심 함수 모듈

포함 내용:
- 벡터 DB 로드 (chroma_db / chroma_db2)
- Hybrid+Rerank 검색 파이프라인
- format_docs / detect_language 유틸
- make_rag_chain_v3 (단발성 RAG 체인)

사용 방법:
    from tricare_core import load_vector_stores, make_rag_chain_v3
    load_vector_stores()
    answer, docs = make_rag_chain_v3("질문", model)
"""

# 만약 오류 난다면 경로 먼저 확인하기
# tricare의 chroma_db, chroma_db2 파일을 tricare_core.py 파일이랑 같은 위치에 두게 하고 돌려보면 됨

import os
import json as _json
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder

load_dotenv()

# 상수 — tricare_core.py 기준 같은 폴더에서 chroma_db, chroma_db2를 찾음
BASE_DIR      = Path(__file__).resolve().parent
PERSIST_TEXT  = str(BASE_DIR / 'chroma_db')
PERSIST_TABLE = str(BASE_DIR / 'chroma_db2')
COLLECTION_TEXT  = 'tricare_rag'
COLLECTION_TABLE = 'tricare_cost_tables'

# 파일명 → 공식 출처 URL 매핑
SOURCE_URL_MAP = {
    "TOP_Handbook_AUG_2023_FINAL_092223_508_1.pdf":          "https://www.tricare-overseas.com/beneficiaries/top",
    "Overseas_HB해외_프로그램_안내서.pdf":                      "https://www.tricare.mil/Plans/HealthPlans/overseas",
    "TFL_HB평생_트라이케어.pdf":                               "https://www.tricare.mil/Plans/HealthPlans/TFL",
    "TRICARE_ADDP_HB_FINAL_508c현역_군인_치과_프로그램_안내서.pdf": "https://www.tricare.mil/Plans/Dental/ADDP",
    "Pharmacy_HBtricare_약국_프로그램_안내서.pdf":               "https://www.tricare.mil/CoveredServices/Pharmacy",
    "Choices_US_HB미국_내_tricare_선택_가이드.pdf":              "https://www.tricare.mil/Plans",
    "NGR_HB국가방위군_및_예비군을_위한_트라이케어_안내서.pdf":       "https://www.tricare.mil/Plans/Reserve",
    "ADDP_Brochure_FINAL_122624_508c.pdf":                   "https://www.tricare.mil/Plans/Dental/ADDP",
    "Maternity_Br_1.pdf":        "https://www.tricare.mil/CoveredServices/Maternity",
    "Medicare_Turning_65_Br.pdf": "https://www.tricare.mil/Plans/HealthPlans/TFL",
    "Medicare_Under_65_Br_7.pdf": "https://www.tricare.mil/Plans/HealthPlans/TFL",
    "Plans_Overview_FS_1.pdf":   "https://www.tricare.mil/Plans",
    "QLEs_FS_2.pdf":             "https://www.tricare.mil/LifeEvents",
    "Retiring_AD_Br.pdf":        "https://www.tricare.mil/LifeEvents/Retiring",
    "Retiring_NGR_Br.pdf":       "https://www.tricare.mil/LifeEvents/Retiring/RetiredNGR",
    "mental_health_services.csv":  "https://www.tricare.mil/CoveredServices/MentalHealth",
    "tricare_exclusions.csv":      "https://www.tricare.mil/CoveredServices/IsItCovered/Exclusions",
    "TricarePlans.csv":            "https://www.tricare.mil/Plans",
    "Health_Plan_Costs.csv":       "https://www.tricare.mil/Costs",
}

# [수정 1] 다국어 지원 추가 — zh-cn/zh-tw/fr/de/es 추가
LANGUAGE_NAME_MAP = {
    'ko':    'Korean',
    'en':    'English',
    'zh':    'Chinese',
    'zh-cn': 'Chinese',
    'zh-tw': 'Chinese',
    'ja':    'Japanese',
    'fr':    'French',
    'de':    'German',
    'es':    'Spanish',
    'other': "the same language as the user's question",
}

# 전역 객체 (load_vector_stores() 호출 후 초기화됨)
embedding_model    = None
vector_store       = None
table_vector_store = None
bm25_retriever     = None
reranker           = None
model              = None
_norm_model        = None
_all_text_chunks   = None


def load_vector_stores(device: str = 'cpu') -> None:
    """
    저장된 Chroma 벡터 DB를 로드하고 전역 객체를 초기화함.
    Streamlit 앱 시작 시 한 번만 호출하면 됨.
    """
    global embedding_model, vector_store, table_vector_store
    global bm25_retriever, reranker, model, _norm_model, _all_text_chunks

    print('⏳ 모델 및 벡터 DB 로드 중...')

    embedding_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={'device': device}
    )

    vector_store = Chroma(
        collection_name=COLLECTION_TEXT,
        embedding_function=embedding_model,
        persist_directory=PERSIST_TEXT
    )
    table_vector_store = Chroma(
        collection_name=COLLECTION_TABLE,
        embedding_function=embedding_model,
        persist_directory=PERSIST_TABLE
    )

    model       = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
    _norm_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    reranker    = CrossEncoder('BAAI/bge-reranker-v2-m3')

    # BM25용 전체 청크 목록 — 벡터 DB에서 직접 추출
    raw = vector_store._collection.get(include=['documents', 'metadatas'])
    from langchain_core.documents import Document
    _all_text_chunks = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(raw['documents'], raw['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(_all_text_chunks, k=6)

    print(f'✅ 텍스트 벡터 DB: {vector_store._collection.count()}개 벡터')
    print(f'✅ 표 벡터 DB: {table_vector_store._collection.count()}개 벡터')
    print(f'✅ BM25 인덱스: {len(_all_text_chunks)}개 청크')
    print('✅ 로드 완료')


# 유틸 함수

# [수정 2] detect_language — langdetect 폴백 추가 (프랑스어/독일어 등 라틴 계열 감지)
try:
    from langdetect import detect, LangDetectException
    _langdetect_available = True
except ImportError:
    _langdetect_available = False

def detect_language(text: str) -> str:
    """
    유니코드 범위로 한/중/일 판별 후 나머지는 langdetect로 감지.
    langdetect 미설치 시 'en' 반환.
    """
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return 'zh'
    if any('\u3040' <= c <= '\u30ff' for c in text):
        return 'ja'
    if any('\uac00' <= c <= '\ud7a3' for c in text):
        return 'ko'
    # 프랑스어·독일어 등 라틴 계열은 langdetect로 판별
    if _langdetect_available:
        try:
            lang = detect(text)
            return lang if lang in LANGUAGE_NAME_MAP else 'en'
        except LangDetectException:
            return 'en'
    return 'en'


def format_docs(docs: list) -> str:
    """
    출처 정보([파일명, p.페이지])를 청크 앞에 붙임.
    search_tags 블록은 LLM 컨텍스트에서 제거.
    """
    result = []
    for doc in docs:
        source  = doc.metadata.get('source_file', doc.metadata.get('source', 'unknown'))
        page    = doc.metadata.get('page', '')
        label   = f'[{source}{", p." + str(page) if page else ""}]'
        content = doc.page_content
        if '[search_tags]' in content:
            content = content.split('[search_tags]')[0].strip()
        result.append(f'{label}\n{content}')
    return '\n\n'.join(result)


def normalize_question(question: str) -> dict:
    """LLM으로 intent / region / 영어쿼리 추출"""
    prompt = (
        "You are a TRICARE insurance query analyzer.\n"
        "Extract:\n"
        "1. intent: one of [coverage, eligibility, cost, pharmacy, dental, overseas, general]\n"
        "2. region: one of [CONUS, OCONUS, korea, unknown]\n"
        "3. english_query: rewrite in clear English (for vector search)\n\n"
        'Respond ONLY in JSON: {{"intent":"...","region":"...","english_query":"..."}}\n\n'
        f"Question: {question}"
    )
    try:
        resp = _norm_model.invoke([HumanMessage(content=prompt)])
        return _json.loads(resp.content)
    except Exception:
        return {'intent': 'general', 'region': 'unknown', 'english_query': question}


# Hybrid + Rerank 검색

def _hybrid_retrieve(question: str, bm25_ret, vec_ret,
                     bm25_weight: float = 0.4, vec_weight: float = 0.6,
                     k: int = 6) -> list:
    """BM25 + 벡터 검색 결과를 RRF 방식으로 합산"""
    bm25_docs = bm25_ret.invoke(question)
    vec_docs  = vec_ret.invoke(question)

    scores, doc_map = {}, {}
    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:100]
        scores[key]  = scores.get(key, 0) + bm25_weight * (1 / (rank + 1))
        doc_map[key] = doc
    for rank, doc in enumerate(vec_docs):
        key = doc.page_content[:100]
        scores[key]  = scores.get(key, 0) + vec_weight * (1 / (rank + 1))
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:k]]


def hybrid_retrieve_wide(question: str, k: int = 20) -> list:
    """
    Rerank 전 단계: BM25+similarity로 넓게 후보 수집.
    bm25_retriever k를 동적으로 변경해 BM25도 k개 후보 수집.
    """
    bm25_retriever.k = k
    # [수정 3] search_type 'mmr' → 'similarity'
    # 후보 수집 단계는 다양성보다 recall이 목적이므로 similarity가 적합
    vec_wide = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )
    result = _hybrid_retrieve(question, bm25_retriever, vec_wide, k=k)
    bm25_retriever.k = 6  # 기본값 복원
    return result


def rerank_docs(question: str, docs: list, top_k: int = 6) -> list:
    """CrossEncoder로 재점수 후 상위 top_k 반환"""
    if not docs:
        return docs
    pairs = []
    for doc in docs:
        content = doc.page_content
        if '[search_tags]' in content:
            content = content.split('[search_tags]')[0].strip()
        pairs.append((question, content))

    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def search(question: str) -> list:
    """
    Hybrid+Rerank 검색을 한 번에 실행하는 편의 함수.
    Streamlit / LangGraph 노드 어디서든 호출 가능.

    Returns
    -------
    list[Document]  상위 6개 문서
    """
    candidates = hybrid_retrieve_wide(question)
    docs = rerank_docs(question, candidates, top_k=6)
    # [수정 4] 출처 일관성 보장 — 동일 질문이면 항상 같은 순서로 정렬
    docs.sort(key=lambda d: (
        d.metadata.get('source_file', d.metadata.get('source', '')),
        d.metadata.get('page', 0)
    ))
    return docs


# RAG 체인 (단발성 질문용)

def make_rag_chain_v3(question: str, conversation_context: str = '') -> tuple:
    """
    Hybrid+Rerank RAG 체인 (단발성).
    LangGraph 없이 단순 Q&A 할 때 사용.
    """
    language_code   = detect_language(question)
    answer_language = LANGUAGE_NAME_MAP.get(language_code, 'English')

    docs    = search(question)
    context = format_docs(docs)

    conv_section = (
        f'[이전 대화]\n{conversation_context}\n\n'
        if conversation_context else ''
    )

    prompt_text = (
        'You are a TRICARE health benefits specialist.\n'
        'This system is designed for OCONUS beneficiaries, '
        'primarily Korean residents and USFK (주한미군) personnel.\n\n'
        'IMPORTANT OCONUS RULES:\n'
        '- In South Korea, TRICARE is the PRIMARY payer (not Medicare).\n'
        '- Medicare does NOT cover overseas medical expenses.\n'
        '- Overseas claims require pay-up-front then submit within 3 years.\n'
        '- Medicare Part B must be actively enrolled for overseas residents.\n\n'
        + conv_section +
        'Answer ONLY based on the provided TRICARE documents below.\n'
        'If not in documents, say "해당 내용은 제공된 문서에서 확인되지 않습니다."\n'
        'Always mention Group A/B, plan type, beneficiary status when relevant.\n'
        '보험 추천이나 특정 플랜을 권유하는 답변은 하지 마세요.\n\n'
        f'IMPORTANT: Answer in {answer_language}.\n'
        'Term locking: 본인부담금(Copay), 공제액(Deductible), 사전승인(Prior Authorization).\n\n'
        f'[참고 문서]\n{context}\n\n'
        f'[질문]\n{question}\n\n'
        '[답변]\n'
    )

    resp = model.invoke([HumanMessage(content=prompt_text)])
    return resp.content, docs
