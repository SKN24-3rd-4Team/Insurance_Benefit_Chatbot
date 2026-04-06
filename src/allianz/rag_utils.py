# annotations : 참조 문제 방지 위해 future annotations 사용
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

# 변수 타입 설성
from typing import Optional, Tuple, List, Dict, Any, TypedDict

# 랭체인 및 관련 라이브러리
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage

# 랭그래프 및 메모리 세이버
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# pydantic : 데이터 모델링 및 검증
# sentence_transformers : 문서 재점수화를 위한 CrossEncoder 모델ㅌ
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PERSIST_DIR = str(BASE_DIR / "vectordb" / "allianz")
COLLECTION_NAME = "allianz_care"
print('여기여기여기여기여기여기여기여기여기여기여기', BASE_DIR)
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

# 랭그래프에서 사용할 채팅 상태 타입 정의
# messages: 대화 메시지 리스트
# user_question: 사용자의 원본 질문
# normalized: 질문에서 추출된 의도(intent), 지역(region), 영어 검색 쿼리(english_query), 키워드(keywords) 등을 포함하는 정규화된 정보 딕셔너리
# slots: 질문에서 추출된 슬롯 정보 (의도, 지역, 치료 유형 등)
# missing_slots: 현재 대화에서 누락된 슬롯 정보 리스트
# needs_followup: 추가 질문이 필요한지 여부
# followup_question: 추가 질문이 필요한 경우, LLM이 생성한 추가 질문
# retrieved_docs: 검색된 문서 리스트
# search_queries: 질문에서 생성된 검색 쿼리 리스트
# answer: LLM이 생성한 답변
# suggested_next_questions: LLM이 생성한 이어서 물어볼 만한 질문 리스트
# followup_count: 지금까지 물어본 추가 질문의 수
# max_followups: 최대 추가 질문 허용 수 (예: 2)
# is_followup_answer: 현재 답변이 추가 질문에 대한 답변인지 여부
class ChatState(TypedDict, total=False):
    messages: List[BaseMessage]
    user_question: str

    normalized: Dict[str, Any]
    slots: Dict[str, Any]
    missing_slots: List[str]

    needs_followup: bool
    followup_question: str

    retrieved_docs: List[Document]
    search_queries: List[str]
    answer: str
    suggested_next_questions: List[str]

    followup_count: int
    max_followups: int
    is_followup_answer: bool

# 슬롯 추출
# language: 질문 언어
# intent: 질문 의도 (coverage, preauth, claim)
# region: 질문에서 감지된 지역 정보
# country_of_treatment: 치료가 이루어지는 국가 (region과 일치하거나, region이 없는 경우 none)
# plan: 질문에서 감지된 보험 플랜 정보 (Care Base, Care Enhanced, Care Signature 중 하나)
# treatment_type: 질문에서 감지된 치료 유형 (inpatient, outpatient, maternity, dental, surgery, emergency, checkup, unknown 중 하나)
# form_type: 질문에서 감지된 관련 서류 유형 (preauth_form, claim_form, none 중 하나)
# injury_or_condition: 질문에서 감지된 부상이나 질병 정보 (짧은 문구 형태)
# asked_info: 질문에서 사용자가 무엇을 묻고 있는지에 대한 정보 리스트 (예: ["preauth requirement"], ["required documents"], ["form fields"], ["coverage limit"] 등) 
class SlotExtractionResult(BaseModel):
    language: str = Field(default="ko")
    intent: str = Field(default="coverage")
    region: str = Field(default="none")
    country_of_treatment: Optional[str] = None
    plan: Optional[str] = None
    treatment_type: Optional[str] = None
    form_type: Optional[str] = None
    injury_or_condition: Optional[str] = None
    asked_info: List[str] = Field(default_factory=list)

# ConversationState : 대화의 현재 상태를 나타내는 모델
# thread_id: 대화의 고유 ID
# slots: 현재까지 추출된 슬롯 정보
# pending_followup: 추가 질문이 필요한지 여부
# last_followup_question: 마지막으로 물어본 추가 질문
# followup_count: 지금까지 물어본 추가 질문의 수
# max_followups: 최대 추가 질문 허용 수 (예: 2)
class ConversationState(TypedDict, total=False):
    thread_id: str
    slots: Dict[str, Any]
    pending_followup: bool
    last_followup_question: str
    followup_count: int
    max_followups: int

# vectordb 연결 함수
# HuggingFaceEmbeddings를 사용하여 문서 임베딩을 생성하고, Chroma 벡터스토어에 연결하여 검색 기능을 제공
def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

# 언어 및 지역 감지 함수
def fallback_detect_language(text: str) -> str:
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return "zh"
    if any('\u3040' <= c <= '\u30ff' for c in text):
        return "ja"
    if any('\uac00' <= c <= '\ud7a3' for c in text):
        return "ko"
    return "en"

def fallback_detect_region(question: str) -> Optional[str]:
    q = question.lower()

    region_patterns = {
        "singapore": [r"싱가포르", r"\bsingapore\b", r"新加坡", r"シンガポール"],
        "dubai_northern_emirates": [r"두바이", r"북부에미리트", r"\bdubai\b", r"northern emirates", r"\buae\b"],
        "lebanon": [r"레바논", r"\blebanon\b"],
        "indonesia": [r"인도네시아", r"\bindonesia\b"],
        "vietnam": [r"베트남", r"\bvietnam\b"],
        "hong_kong": [r"홍콩", r"hong kong", r"\bhk\b"],
        "china": [r"중국", r"\bchina\b", r"중화권"],
        "switzerland": [r"스위스", r"\bswitzerland\b", r"\bsuisse\b"],
        "uk": [r"영국", r"\buk\b", r"united kingdom", r"\bengland\b", r"britain"],
        "france_benelux_monaco": [r"프랑스", r"\bfrance\b", r"benelux", r"모나코", r"\bmonaco\b"],
        "latin_america": [r"남미", r"라틴아메리카", r"latin america"],
        "global": [r"글로벌", r"전세계", r"worldwide", r"global"],
    }

    for region, patterns in region_patterns.items():
        for pattern in patterns:
            if re.search(pattern, q):
                return region
    return None

# 의도에 따라 어떤 문서 유형이 검색되는지
def get_allowed_doc_types(intent: str) -> List[str]:
    if intent == "coverage":
        return ["benefit_guide", "tob"]
    if intent == "preauth":
        return ["benefit_guide", "preauth_form", "tob"]
    if intent == "claim":
        return ["benefit_guide", "claim_form"]
    return ["benefit_guide", "tob"]

# 문서의 고유 키를 생성하는 함수 (중복 제거 및 랭킹 과정에서 활용)
def doc_unique_key(doc: Document) -> tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("chunk_idx"),
        doc.metadata.get("doc_type"),
        doc.metadata.get("region"),
    )


def strip_search_tags(text: str) -> str:
    if "[search_tags]" in text:
        return text.split("[search_tags]")[0].strip()
    return text.strip()


def build_context(docs: List[Document]) -> str:
    context_parts = []

    for d in docs:
        source = d.metadata.get("source")
        region = d.metadata.get("region")
        page = d.metadata.get("page")
        doc_type = d.metadata.get("doc_type")
        year = d.metadata.get("doc_year")
        content = strip_search_tags(d.page_content)

        context_parts.append(
            f"[Document: {source} | region: {region} | type: {doc_type} | year: {year} | page: {page}]\n"
            f"{content}"
        )

    return "\n\n".join(context_parts)


LANGUAGE_NAME_MAP = {
    "ko": "Korean",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "other": "the same language as the user's question",
}


# LLM이 실패할 경우 대비 간단한 룰 기반 질문 정규화에서 fallback으로 영어 검색 쿼리를 생성하는 함수
def fallback_build_english_query(question: str, intent: str, region: Optional[str]) -> str:
    region_text = "" if not region or region == "none" else region.replace("_", " ")

    if intent == "preauth":
        return f"Is pre-authorisation required before inpatient treatment {f'in {region_text}' if region_text else ''}?".strip()
    if intent == "claim":
        return f"What documents are required to submit a claim {f'in {region_text}' if region_text else ''}?".strip()
    return f"What is covered under the insurance plan {f'in {region_text}' if region_text else ''}?".strip()

# LLM이 실패할 경우 대비 간단한 룰 기반 질문 정규화
# 현재 한국어 질문만 대상으로 함.
def fallback_normalize_question(question: str, language: str) -> Dict[str, Any]:
    q = question.lower()

    preauth_terms = [
        "사전승인", "입원 전 승인", "pre-author", "preauthor", "pre-auth",
        "prior approval", "approval before", "hospital approval", "direct billing"
    ]
    claim_terms = [
        "청구", "환급", "보험금", "영수증", "서류", "claim",
        "reimbursement", "invoice", "receipt", "refund"
    ]

    if any(t in q for t in preauth_terms):
        intent = "preauth"
    elif any(t in q for t in claim_terms):
        intent = "claim"
    else:
        intent = "coverage"
    
    region = fallback_detect_region(question)
    english_query = fallback_build_english_query(question, intent, region)

    return {
        "language": language,
        "intent": intent,
        "region": region if region else "none",
        "english_query": english_query,
        "keywords": [],
    }

# LLM을 활용한 질문 정규화.
# 질문에서 랭그래프 노드에서 활용할 수 있는 슬롯 정보(의도, 지역, 치료 유형 등)도 함께 추출하여 반환
def normalize_question(question: str) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # LLM이 의도(intent), 지역(region), 영어로 변환된 검색 쿼리(english_query), 키워드(keywords) 등을 명확히 추출하도록 유도
    prompt = f"""
You are a multilingual insurance query normalizer.

Return STRICT JSON only:
{{
  "language": "ko|en|zh|ja|es|fr|de|other",
  "intent": "coverage|preauth|claim",
  "region": "singapore|dubai_northern_emirates|lebanon|indonesia|vietnam|hong_kong|china|switzerland|uk|france_benelux_monaco|latin_america|global|none",
  "english_query": "concise English retrieval query",
  "keywords": ["keyword1", "keyword2"]
}}

User question:
{question}
"""
   # LLM이 제대로 된 JSON을 반환하지 않거나, 의도/지역을 잘못 감지하는 경우에 대비하여 fallback 언어 감지 및 룰 기반 정규화도 함께 수행 
    fallback_language = fallback_detect_language(question)
    fallback_region = fallback_detect_region(question)

    try:
        # llm 실행
        raw = llm.invoke(prompt).content.strip()
        # 답변에서 json 추출
        data = json.loads(raw)
        
        # 여기 테스트용
        print(f'!!! Data: {data}')

        # language에 값이 있으면 그래로 사용, 없거나 허용되지 않은 값이면 fallback으로 감지된 언어 사용
        language = data.get("language", fallback_language)
        # intent은 coverage, preauth, claim 중 하나여야 하며, 잘못된 값이거나 없는 경우 coverage로 기본 설정
        intent = data.get("intent", "coverage")
        # region은 허용된 값 중 하나여야 하며, 잘못된 값이거나 없는 경우 none으로 설정
        region = data.get("region", "none")
        # english_query는 필수적으로 제공되어야 하며, 없거나 빈 문자열인 경우 원본 질문 사용
        english_query = data.get("english_query", question)
        # keywords는 리스트 형태여야 하며, 잘못된 형식이거나 없는 경우 빈 리스트로 설정
        keywords = data.get("keywords", [])

        # 허용된 값인지 검증하기 위한 셋업
        allowed_languages = {"ko", "en", "zh", "ja", "es", "fr", "de", "other"}
        allowed_regions = {
            "singapore", "dubai_northern_emirates", "lebanon", "indonesia", "vietnam",
            "hong_kong", "china", "switzerland", "uk", "france_benelux_monaco",
            "latin_america", "global", "none"
        }
        allowed_intents = {"coverage", "preauth", "claim"}

        if language not in allowed_languages:
            language = fallback_language
        if intent not in allowed_intents:
            intent = "coverage"
        if region not in allowed_regions:
            region = "none"
        if not isinstance(keywords, list):
            keywords = []

        return {
            "language": language,
            "intent": intent,
            "region": region,
            "english_query": english_query.strip(),
            "keywords": [str(k).strip() for k in keywords if str(k).strip()],
        }
    except Exception:
        return fallback_normalize_question(question, fallback_language)

# 질문에서 추출된 의도(intent), 지역(region), 키워드(keywords) 등을 활용하여 검색 쿼리를 생성하는 함수
def build_keyword_query(intent: str, region: str, keywords: List[str]) -> str:
    region_text = "" if region in {"none", "global"} else region.replace("_", " ")
    base = " ".join(keywords[:5]).strip()

    if base:
        return f"{region_text} {base}".strip()

    if intent == "preauth":
        return f"{region_text} pre-authorisation inpatient hospitalisation approval".strip()
    if intent == "claim":
        return f"{region_text} claim reimbursement invoice receipt documents".strip()
    return f"{region_text} coverage benefits limits exclusions".strip()

# 키워드 기반 쿼리가 충분히 구체적이지 않거나, 
# LLM이 키워드를 제대로 추출하지 못한 경우에 대비하여, 
# 의도(intent)와 지역(region)에 기반한 추가적인 fallback 검색 쿼리를 생성하는 함수
def fallback_build_queries(intent: str, region: str) -> List[str]:
    region_text = "" if region in {"none", "global"} else region.replace("_", " ")

    # 의도와 지역 정보로
    # 관련된 검색 쿼리 예시 생성
    # 예를 들어, 사전승인(preauth) 의도가 감지되고 지역이 싱가포르로 감지된 경우,
    # "싱가포르에서 입원 치료 전에 사전승인이 필요한가요?"와 같은 검색 쿼리를 생성하여, 벡터DB에서 관련 문서를 더 잘 검색할 수 있도록 도움
    if intent == "preauth":
        return [
            f"{region_text} pre-authorisation required before inpatient treatment".strip(),
            f"{region_text} planned hospitalisation prior approval".strip(),
        ]
    if intent == "claim":
        return [
            f"{region_text} claim reimbursement required documents".strip(),
            f"{region_text} invoice receipt claim form".strip(),
        ]
    return [
        f"{region_text} coverage benefits limits exclusions".strip(),
        f"{region_text} inpatient outpatient benefit limit".strip(),
    ]

# 검색 쿼리 생성 함수
def make_search_queries(normalized: Dict[str, Any], original_question: str) -> List[str]:
    region = normalized["region"]
    intent = normalized["intent"]
    english_query = normalized["english_query"]
    keywords = normalized.get("keywords", [])

    # 원본 질문
    queries = [original_question.strip(), english_query.strip()]
    keyword_query = build_keyword_query(intent, region, keywords)
    if keyword_query:
        queries.append(keyword_query)
    queries.extend(fallback_build_queries(intent, region))

    deduped = []
    seen = set()
    for q in queries:
        nq = q.lower().strip()
        if nq and nq not in seen:
            seen.add(nq)
            deduped.append(q.strip())

    return deduped[:5]

# 검색된 문서에 대해 질문과의 관련성 점수를 계산하는 함수
def score_document(question: str, doc: Document, intent: str, detected_region: Optional[str]) -> int:
    score = 0
    q = question.lower()
    content = doc.page_content.lower()
    metadata = doc.metadata

    if detected_region and metadata.get("region") == detected_region:
        score += 6
    if metadata.get("region") == "global":
        score += 2

    if intent == "preauth" and metadata.get("doc_type") in ["preauth_form", "benefit_guide", "tob"]:
        score += 4
    elif intent == "claim" and metadata.get("doc_type") in ["claim_form", "benefit_guide"]:
        score += 4
    elif intent == "coverage" and metadata.get("doc_type") in ["benefit_guide", "tob"]:
        score += 4

    keyword_groups = [
        ["pre-authorisation", "preauthorization", "prior approval", "preauth"],
        ["direct billing"],
        ["claim", "reimbursement", "refund"],
        ["invoice", "receipt", "documents"],
        ["inpatient", "hospitalisation", "hospitalization", "admission"],
        ["outpatient"],
        ["maternity", "pregnancy"],
        ["benefit limit", "coverage limit", "limit"],
        ["exclusion"],
    ]

    for group in keyword_groups:
        if any(k in q for k in group) and any(k in content for k in group):
            score += 3

    score += min(len(content) // 300, 3)
    print(f"관련성 점수 Scoring doc {metadata.get('source')} (region: {metadata.get('region')}, type: {metadata.get('doc_type')}) => score: {score}")
    return score

# reranking을 통해 상위 8개 문서만 최종 검색 결과로 반환 (기본적으로 벡터 유사도 검색으로 20개 정도 가져온 후 재점수화)
def rerank_documents(question: str, docs: List[Document], top_n: int = 8) -> List[Document]:
    if not docs:
        return docs

    try:
        reranker = get_reranker()
        pairs = []

        for d in docs:
            content = strip_search_tags(d.page_content)
            short_content = content[:1500]
            pairs.append([question, short_content])

        scores = reranker.predict(pairs)

        rescored = list(zip(docs, scores))
        rescored.sort(key=lambda x: float(x[1]), reverse=True)

        return [doc for doc, _ in rescored[:top_n]]

    except Exception:
        return docs[:top_n]

_BM25_INDEX = None

# 간단한 토큰화 함수. BM25 검색을 위해 텍스트를 소문자로 변환하고, 특수문자를 제거한 후 공백으로 분할하여 토큰 리스트를 반환
def simple_tokenize(text: str) -> List[str]:
    text = strip_search_tags(text.lower())
    text = re.sub(r"[^a-z0-9가-힣\s]", " ", text)
    return text.split()

# BM25 인덱스 구축 함수. 벡터DB에서 문서와 메타데이터를 가져와서, BM25 검색을 위한 토큰화된 코퍼스를 생성하고, BM25 모델을 초기화
def build_bm25_index(vectordb):
    data = vectordb.get(include=["documents", "metadatas"])
    raw_docs = data.get("documents", [])
    raw_metas = data.get("metadatas", [])

    docs: List[Document] = []
    for content, meta in zip(raw_docs, raw_metas):
        docs.append(Document(page_content=content, metadata=meta or {}))

    tokenized_corpus = [simple_tokenize(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, docs

def get_bm25_index():
    global _BM25_INDEX
    if _BM25_INDEX is None:
        vectordb = get_vectorstore()
        _BM25_INDEX = build_bm25_index(vectordb)
    return _BM25_INDEX

def bm25_search(bm25, docs: List[Document], query: str, top_k: int = 10):
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        [(docs[i], float(scores[i])) for i in range(len(scores))],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]

# vectorstore에서 검색된 문서 리스트를 질문과 함께 LLM에 전달하여, 답변 생성에 활용할 수 있도록 하는 함수
# 결과값으로 답변(answer), 슬롯 정보(slots), 검색된 문서 리스트(retrieved_docs), 이어서 물어볼 만한 질문 리스트(suggested_next_questions) 등을 반환
def retrieve_documents_from_slots(
    question: str,
    normalized: Dict[str, Any],
    slots: Dict[str, Any]
) -> Tuple[List[Document], List[str]]:
    vectordb = get_vectorstore()
    bm25, bm25_docs = get_bm25_index()

    intent = slots.get("intent") or normalized["intent"]
    region_value = slots.get("country_of_treatment") or slots.get("region") or normalized["region"]
    detected_region = None if region_value in [None, "", "none"] else region_value

    allowed_doc_types = get_allowed_doc_types(intent)

    regions = ["global"]
    if detected_region and detected_region != "global":
        regions.append(detected_region)

    slot_hint_parts = []
    if slots.get("plan"):
        slot_hint_parts.append(f"plan: {slots['plan']}")
    if slots.get("treatment_type"):
        slot_hint_parts.append(f"treatment_type: {slots['treatment_type']}")
    if slots.get("injury_or_condition"):
        slot_hint_parts.append(f"condition: {slots['injury_or_condition']}")
    if slots.get("asked_info"):
        slot_hint_parts.append(f"asked_info: {', '.join(slots['asked_info'])}")

    enriched_question = question
    if slot_hint_parts:
        enriched_question = question + " | " + " | ".join(slot_hint_parts)

    temp_normalized = dict(normalized)
    temp_normalized["region"] = detected_region or "none"
    temp_normalized["intent"] = intent
    queries = make_search_queries(temp_normalized, enriched_question)

    search_filter = {
        "$and": [
            {"doc_type": {"$in": allowed_doc_types}},
            {"region": {"$in": regions}}
        ]
    }

    hybrid_pool: Dict[tuple, Dict[str, Any]] = {}

    def passes_filter(doc: Document) -> bool:
        return (
            doc.metadata.get("doc_type") in allowed_doc_types and
            doc.metadata.get("region") in regions
        )

    for q in queries:
        # 1) dense
        try:
            dense_docs = vectordb.max_marginal_relevance_search(
                q, k=10, fetch_k=30, filter=search_filter
            )
        except Exception:
            dense_docs = vectordb.similarity_search(
                q, k=10, filter=search_filter
            )

        for rank, d in enumerate(dense_docs, start=1):
            key = doc_unique_key(d)
            if key not in hybrid_pool:
                hybrid_pool[key] = {
                    "doc": d,
                    "dense_rank": rank,
                    "bm25_rank": None,
                }
            else:
                hybrid_pool[key]["dense_rank"] = min(
                    hybrid_pool[key]["dense_rank"] or rank, rank
                )

        # 2) bm25
        bm25_ranked = bm25_search(bm25, bm25_docs, q, top_k=10)
        for rank, (d, _) in enumerate(bm25_ranked, start=1):
            if not passes_filter(d):
                continue
            key = doc_unique_key(d)
            if key not in hybrid_pool:
                hybrid_pool[key] = {
                    "doc": d,
                    "dense_rank": None,
                    "bm25_rank": rank,
                }
            else:
                hybrid_pool[key]["bm25_rank"] = min(
                    hybrid_pool[key]["bm25_rank"] or rank, rank
                )

    # Reciprocal Rank Fusion + 기존 rule score
    scored_docs = []
    for item in hybrid_pool.values():
        d = item["doc"]
        dense_rrf = 1 / (60 + item["dense_rank"]) if item["dense_rank"] else 0.0
        bm25_rrf = 1 / (60 + item["bm25_rank"]) if item["bm25_rank"] else 0.0
        print(f"Doc: {d.metadata.get('source')} \
            | dense_rank: {item['dense_rank']} | \
                bm25_rank: {item['bm25_rank']} | \
                    dense_rrf: {dense_rrf:.4f} | \
                        bm25_rrf: {bm25_rrf:.4f}")
        rule_score = score_document(enriched_question, d, intent, detected_region)

        final_score = (0.65 * dense_rrf) + (0.35 * bm25_rrf) + (0.02 * rule_score)
        scored_docs.append((d, final_score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    ranked_docs = [doc for doc, _ in scored_docs[:15]]

    reranked_docs = rerank_documents(enriched_question, ranked_docs, top_n=8)
    return reranked_docs, queries

# KNOWN_PLANS 딕셔너리는 질문에서 플랜 정보를 추출할 때 활용되는 키워드와 해당 키워드가 매핑되는 표준 플랜 이름을 정의
KNOWN_PLANS = {
    "care base": "Care Base",
    "base": "Care Base",
    "care enhanced": "Care Enhanced",
    "enhanced": "Care Enhanced",
    "care signature": "Care Signature",
    "signature": "Care Signature",
}

# TREATMENT_KEYWORDS 딕셔너리는 질문에서 치료 유형을 추출할 때 활용되는 키워드와 해당 키워드가 매핑되는 표준 치료 유형 이름을 정의
TREATMENT_KEYWORDS = {
    "inpatient": ["입원", "inpatient", "hospitalisation", "hospitalization", "admission"],
    "outpatient": ["외래", "outpatient"],
    "maternity": ["출산", "임신", "maternity", "pregnancy"],
    "dental": ["치과", "dental"],
    "surgery": ["수술", "surgery", "operation"],
}

# 간단한 룰 기반 슬롯 추출. LLM이 잘못 추출하거나 놓친 경우를 보완하기 위함
def extract_slots_heuristic(question: str) -> Dict[str, Any]:
    q = question.lower()
    slots: Dict[str, Any] = {}

    region = fallback_detect_region(question)
    if region:
        slots["region"] = region
        slots["country_of_treatment"] = region

    for key, val in KNOWN_PLANS.items():
        if key in q:
            slots["plan"] = val
            break

    for treatment_type, words in TREATMENT_KEYWORDS.items():
        if any(w in q for w in words):
            slots["treatment_type"] = treatment_type
            break

    if any(x in q for x in ["사전승인", "pre-auth", "preauthor", "pre-author", "prior approval", "직접청구"]):
        slots["intent"] = "preauth"
        slots["form_type"] = "preauth_form"
        slots["asked_info"] = ["preauth requirement"]
    elif any(x in q for x in ["청구", "claim", "reimbursement", "환급", "서류", "영수증", "invoice", "receipt"]):
        slots["intent"] = "claim"
        slots["form_type"] = "claim_form"
        if any(x in q for x in ["서류", "documents", "invoice", "receipt", "영수증"]):
            slots["asked_info"] = ["required documents"]
    else:
        slots["intent"] = "coverage"
        if any(x in q for x in ["보장", "cover", "coverage", "한도", "limit"]):
            slots["asked_info"] = ["coverage limit"]

    condition_patterns = [
        r"출산",
        r"임신",
        r"암",
        r"치과",
        r"수술",
        r"입원",
        r"외래",
        r"pregnancy",
        r"cancer",
        r"dental",
        r"surgery",
    ]
    for pat in condition_patterns:
        if re.search(pat, q):
            slots["injury_or_condition"] = pat
            break

    return slots

# LLM이 추출한 슬롯 정보와 룰 기반으로 추출한 슬롯 정보를 병합하는 함수
def merge_slots(existing: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing or {})
    for k, v in (new_data or {}).items():
        if v not in [None, "", [], {}]:
            merged[k] = v
    return merged

# missing_slots 리스트를 기반으로, 어떤 슬롯이 누락되었는지 판단하는 함수
def decide_missing_slots(intent: str, slots: Dict[str, Any], question: str) -> List[str]:
    q = question.lower()
    missing = []

    if intent in ["preauth", "claim"] and not slots.get("country_of_treatment"):
        missing.append("country_of_treatment")

    if intent == "coverage":
        if not slots.get("country_of_treatment") and not slots.get("region"):
            if any(x in q for x in ["보장", "cover", "coverage", "limit", "한도"]):
                missing.append("country_of_treatment")

    if intent == "preauth" and not slots.get("treatment_type"):
        missing.append("treatment_type")

    return missing

# missing_slots 리스트를 기반으로, 후속 질문을 생성하는 함수
def fallback_build_followup_question(language: str, missing_slots: List[str], intent: str) -> str:
    if not missing_slots:
        return ""
    print("fallback_build_followup_question 진입==============================================`")
    print(f"Missing slots: {missing_slots}")
    print("fallback_build_followup_question 진입==============================================")
    first = missing_slots[0]

    if language == "ko":
        if first == "country_of_treatment":
            return "어느 국가에서 치료를 받으실 예정인지 알려주시면 지역 기준으로 더 정확히 확인해드릴게요."
        if first == "treatment_type":
            return "입원 치료인지, 외래 치료인지 알려주시면 사전승인 필요 여부를 더 정확히 확인해드릴게요."
        if first == "plan":
            return "가입하신 플랜이 Care Base, Care Enhanced, Care Signature 중 무엇인지 아시면 알려주세요."
        return "추가 정보를 알려주시면 더 정확히 확인해드릴게요."

    return "Please share a bit more detail so I can check the documents more accurately."

# 후속 질문과 함께 이어서 물어볼 만한 질문 리스트를 생성하는 함수 (LLM 생성 실패시)
def fallback_suggested_next_questions(language: str, intent: str, slots: Dict[str, Any]) -> List[str]:
    if language == "ko":
        if intent == "preauth":
            return [
                "사전승인 폼에 어떤 항목을 입력해야 하는지 정리해드릴까요?",
                "해당 치료가 어떤 플랜에서 얼마나 보장되는지도 확인해드릴까요?",
                "사전승인 전에 준비할 서류도 같이 정리해드릴까요?",
            ]
        if intent == "claim":
            return [
                "청구할 때 필요한 서류를 정리해드릴까요?",
                "영수증이나 인보이스 외에 추가로 필요한 문서가 있는지도 확인해드릴까요?",
                "지역별 청구 절차 차이도 같이 정리해드릴까요?",
            ]
        return [
            "해당 치료가 사전승인 대상인지도 같이 확인해드릴까요?",
            "플랜별 보장 한도까지 같이 찾아드릴까요?",
            "관련 청구 절차나 필요 서류도 이어서 정리해드릴까요?",
        ]

    return [
        "Would you like me to summarize the required form fields as well?",
        "Would you like me to check pre-authorisation requirements too?",
        "Would you like me to also check coverage limits by plan?",
    ]

def looks_like_followup_answer(text: str) -> bool:
    t = text.strip().lower()

    short_patterns = [
        r"^싱가포르(요)?$",
        r"^입원(입니다)?$",
        r"^외래(입니다)?$",
        r"^care base$",
        r"^care enhanced$",
        r"^care signature$",
    ]

    if len(t) <= 20:
        for pat in short_patterns:
            if re.search(pat, t):
                return True

    return False

# 랭그래프 노드 함수
# classify_and_extract 노드는 사용자 질문을 입력으로 받아, 질문 정규화와 슬롯 추출을 동시에 수행하여, 이후 노드에서 활용할 수 있는 형태로 정보를 구조화하는 역할을 함
def classify_and_extract_node(state: ChatState) -> ChatState:
    question = state["user_question"]
    old_slots = state.get("slots", {})
    followup_count = state.get("followup_count", 0)
    max_followups = state.get("max_followups", 2)

    normalized = normalize_question(question)

    # 현재 입력에서 슬롯 추출
    llm_slots = extract_slots_llm(
        question,
        existing_slots=old_slots,
        pending_followup=state.get("is_followup_answer", False),
        last_followup_question=state.get("followup_question", ""),
    )

    # 기존 슬롯 + 새 슬롯 병합
    new_slots = merge_slots(old_slots, llm_slots)

    # normalized 기반 보정
    if normalized["intent"] and not new_slots.get("intent"):
        new_slots["intent"] = normalized["intent"]

    if normalized["region"] != "none" and not new_slots.get("region"):
        new_slots["region"] = normalized["region"]

    if new_slots.get("region") and not new_slots.get("country_of_treatment"):
        new_slots["country_of_treatment"] = new_slots["region"]

    intent = new_slots.get("intent", normalized["intent"])
    missing_slots = decide_missing_slots(intent, new_slots, question)

    # follow-up 한도를 넘으면 더 안 묻고 그냥 검색으로 넘기기 위한 표시
    if followup_count >= max_followups:
        missing_slots = []

    return {
        "normalized": normalized,
        "slots": new_slots,
        "missing_slots": missing_slots,
        "followup_count": followup_count,
        "max_followups": max_followups,
    }

# followup_router는 classify_and_extract 노드에서 후속 질문이 필요한 경우와 그렇지 않은 경우를 분기 처리하여, ask_followup 노드로 갈지 retrieve 노드로 갈지 결정하는 역할을 함  
def followup_router(state: ChatState) -> str:
    return "ask_followup" if state.get("missing_slots") else "retrieve"


# ask_followup 노드는 missing_slots에 기반해 적절한 후속 질문을 생성하여 사용자에게 추가 정보를 요청하는 역할을 함
def ask_followup_node(state: ChatState) -> ChatState:
    normalized = state["normalized"]
    followup_question = build_followup_question_llm(
        language=normalized["language"],
        missing_slots=state.get("missing_slots", []),
        intent=normalized["intent"],
        slots=state.get("slots", {}),
    )

    return {
        "needs_followup": True,
        "followup_question": followup_question,
        "answer": followup_question,
        "suggested_next_questions": [],
        "retrieved_docs": [],
        "search_queries": [],
        "followup_count": state.get("followup_count", 0) + 1,
    }

# retrieve 노드는 질문과 슬롯 정보를 활용하여 벡터DB에서 관련 문서를 검색하는 역할을 함
def retrieve_node(state: ChatState) -> ChatState:
    docs, queries = retrieve_documents_from_slots(
        question=state["user_question"],
        normalized=state["normalized"],
        slots=state.get("slots", {}),
    )
    return {
        "retrieved_docs": docs,
        "search_queries": queries,
    }

# answer 노드는 검색된 문서들을 바탕으로 사용자 질문에 대한 답변을 생성하는 역할을 함
def answer_node(state: ChatState) -> ChatState:
    docs = state.get("retrieved_docs", [])
    normalized = state["normalized"]
    slots = state.get("slots", {})
    context = build_context(docs)

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    language_code = normalized["language"]
    answer_language = LANGUAGE_NAME_MAP.get(language_code, "English")
    intent = normalized["intent"]

    region_text = (
        slots.get("country_of_treatment")
        or slots.get("region")
        or normalized.get("region")
        or "none"
    )

    plan_text = slots.get("plan", "unknown")
    treatment_text = slots.get("treatment_type", "unknown")

    prompt = f"""
You are an Allianz insurance document-based assistant.

You must answer ONLY based on the provided context.
Do not guess unsupported facts.
If something cannot be confirmed from the documents, say so clearly.
Do not present the answer as a legal or medical final judgment.
Present it as document-based insurance guidance.

IMPORTANT:
- Answer in {answer_language}.
- Match the user's language.

Conversation state:
- intent: {intent}
- region/country: {region_text}
- plan: {plan_text}
- treatment_type: {treatment_text}
- extracted slots: {slots}
- search queries used: {state.get("search_queries", [])}

Answer format:
1. Conclusion
2. Region-specific basis
3. General/global rule
4. Procedure or notes
5. Sources

User question:
{state["user_question"]}

Context:
{context}
"""

    result = llm.invoke(prompt).content

    next_questions = build_suggested_next_questions_llm(
        language=language_code,
        intent=intent,
        slots=slots,
        answer=result,
    )

    return {
        "needs_followup": False,
        "answer": result,
        "suggested_next_questions": next_questions,
    }

# 랭그래프
# 노드간의 흐름과 상태 관리를 담당하는 랭그래프를 구축하는 함수
# 조건부 엣지를 활용하여 classify_and_extract 노드에서 후속 질문이 필요한 경우와 그렇지 않은 경우를 분기 처리
def build_chatbot_graph():
    graph = StateGraph(ChatState)

    graph.add_node("classify_and_extract", classify_and_extract_node)
    graph.add_node("ask_followup", ask_followup_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("classify_and_extract")
    graph.add_conditional_edges(
        "classify_and_extract",
        followup_router,
        {
            "ask_followup": "ask_followup",
            "retrieve": "retrieve",
        },
    )

    # 기존: graph.add_edge("ask_followup", END)
    graph.add_edge("ask_followup", "classify_and_extract")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


CHATBOT_GRAPH = build_chatbot_graph()


# 본격 챗봇 인터페이스.
# 질문과 thread_id(대화 세션 식별자)를 받아서 챗봇 그래프를 실행시키고 답변과 참고 문서를 반환하는 함수
def run_chat_turn(
    question: str,
    conversation_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    conversation_state = conversation_state or {}

    thread_id = conversation_state.get("thread_id", "default-thread")
    prior_slots = conversation_state.get("slots", {})
    followup_count = conversation_state.get("followup_count", 0)
    max_followups = conversation_state.get("max_followups", 2)
    pending_followup = conversation_state.get("pending_followup", False)
    is_followup_answer = pending_followup or looks_like_followup_answer(question)
    
    init_state: ChatState = {
        "user_question": question,
        "messages": [HumanMessage(content=question)],
        "slots": prior_slots,
        "followup_count": followup_count,
        "max_followups": max_followups,
        "is_followup_answer": is_followup_answer,
    }

    result = CHATBOT_GRAPH.invoke(
        init_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    updated_conversation_state = {
        "thread_id": thread_id,
        "slots": result.get("slots", prior_slots),
        "pending_followup": result.get("needs_followup", False),
        "last_followup_question": result.get("followup_question", ""),
        "followup_count": result.get("followup_count", followup_count),
        "max_followups": max_followups,
    }

    return {
        "answer": result.get("answer", ""),
        "retrieved_docs": result.get("retrieved_docs", []),
        "suggested_next_questions": result.get("suggested_next_questions", []),
        "needs_followup": result.get("needs_followup", False),
        "followup_question": result.get("followup_question", ""),
        "slots": result.get("slots", prior_slots),
        "conversation_state": updated_conversation_state,
    }

# 랭그래프 사용하지 않은 단발성 함수. (테스트용)
def generate_answer(question: str) -> Tuple[str, List[Document]]:
    """
    하위 호환용.
    기존 main.py에서 단발 호출할 때도 동작하게 유지.
    """
    result = run_chat_turn(question=question, thread_id="single-turn")
    return result.get("answer", ""), result.get("retrieved_docs", [])

_RERANKER = None

# Reranker 모델을 싱글턴으로 로드하여 재사용
def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        model_name = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        _RERANKER = CrossEncoder(model_name)
    return _RERANKER

# LLM이 추출한 슬롯 정보가 기존 슬롯 정보와 일관되지 않거나,
# LLM이 제대로 추출하지 못한 경우를 보완하기 위해 기존 룰 기반 추출 함수도 함께 활용하여, 두 결과를 병합하는 방식으로 슬롯 정보를 최종 결정
# 예를 들어, LLM이 질문에서 치료 유형(treatment_type)을 추출하지 못했지만,
# 기존 룰 기반 함수가 "입원"이라는 키워드를 감지하여 치료 유형을 "inpatient"로 추출한 경우,
# 최종 슬롯 정보에는 치료 유형이 "inpatient"로 포함될 수 있도록 함
def extract_slots_llm(
    question: str,
    existing_slots: Optional[Dict[str, Any]] = None,
    pending_followup: bool = False,
    last_followup_question: str = "",
) -> Dict[str, Any]:
    existing_slots = existing_slots or {}
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    structured_llm = llm.with_structured_output(SlotExtractionResult)

    allowed_regions = [
        "singapore", "dubai_northern_emirates", "lebanon", "indonesia",
        "vietnam", "hong_kong", "china", "switzerland", "uk",
        "france_benelux_monaco", "latin_america", "global", "none",
    ]

    prompt = f"""
You are an insurance dialogue slot extractor.

Extract structured slots from the user's current message.
If the current message looks like a short reply to a previous follow-up question,
infer only the missing slot(s) conservatively.
Do not overwrite valid existing slots unless the user clearly corrects them.

Pending follow-up: {pending_followup}
Last follow-up question: {last_followup_question}

Previously known slots:
{existing_slots}

User message:
{question}
"""

    try:
        result = structured_llm.invoke(prompt)
        data = result.model_dump()

        if data.get("region") not in allowed_regions:
            data["region"] = "none"

        if data.get("form_type") not in ["preauth_form", "claim_form", "none", None]:
            data["form_type"] = None

        if data.get("country_of_treatment") is None and data.get("region") not in [None, "none"]:
            data["country_of_treatment"] = data["region"]

        cleaned = {}
        for k, v in data.items():
            if v not in [None, "", [], {}]:
                cleaned[k] = v
        return cleaned

    except Exception:
        return extract_slots_heuristic(question)

# LLM을 활용하여, 질문에 대한 답변과 현재 슬롯 정보에 기반하여, 이어서 물어볼 만한 자연스러운 후속 질문을 생성하는 함수
# build_suggested_next_questions_llm랑 다른 점 : 
# build_followup_question_llm는 missing_slots 리스트에 기반하여,
# 가장 필요한 한 가지 정보를 물어보는 후속 질문을 생성하는 데 초점이 있는 반면,
# build_suggested_next_questions_llm는 질문에 대한 답변과 현재 슬롯 정보를 모두 고려하여, 
# 사용자가 이어서 물어볼 만한 자연스러운 질문 리스트를 생성하는 데 초점이 있음
def build_followup_question_llm(language: str, missing_slots: List[str], intent: str, slots: Dict[str, Any]) -> str:
    if not missing_slots:
        return ""

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt = f"""
You are an insurance assistant.

Generate exactly one concise follow-up question.
Ask only the single most necessary question needed to improve document-based insurance guidance.
Do not ask multiple questions at once.
Be polite and concise.

Language: {language}
Intent: {intent}
Missing slots: {missing_slots}
Current known slots: {slots}

Rules:
- If language is ko, respond in Korean.
- Do not mention internal slot names.
- Keep it to one sentence.
"""

    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return fallback_build_followup_question(language, missing_slots, intent)

# LLM을 활용하여, 질문에 대한 답변과 현재 슬롯 정보에 기반하여, 이어서 물어볼 만한 자연스러운 질문 리스트를 생성하는 함수
# build_followup_question_llm와 다른 점 :
# build_followup_question_llm는 missing_slots 리스트에 기반하여,
# 가장 필요한 한 가지 정보를 물어보는 후속 질문을 생성하는 데 초점이 있는 반면,
# build_suggested_next_questions_llm는 질문에 대한 답변과 현재 슬롯 정보를
# 모두 고려하여, 사용자가 이어서 물어볼 만한 자연스러운 질문 리스트를 생성하는 데 초점이 있음
def build_suggested_next_questions_llm(
    language: str,
    intent: str,
    slots: Dict[str, Any],
    answer: str
) -> List[str]:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt = f"""
You are an insurance assistant.

Generate exactly 3 natural next questions the user may want to ask next.
They must be highly relevant to the current answer and current known slots.

Language: {language}
Intent: {intent}
Known slots: {slots}

Current answer:
{answer}

Rules:
- If language is ko, output Korean.
- Return JSON array only.
- Each item must be a single user-style question.
- Avoid duplicates.
"""

    try:
        raw = llm.invoke(prompt).content.strip()
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()][:3]
    except Exception:
        pass

    return fallback_suggested_next_questions(language, intent, slots)
