import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'shared'))

from insurance_plugin import InsurancePlugin
from shared_embedding import get_embedding_model

# [Source 15 반영] OCONUS 및 Medicare 관련 전문 지침 강화
TRICARE_PROMPT = """You are a TRICARE health benefits specialist.
This system is designed for OCONUS beneficiaries, primarily Korean residents and USFK (주한미군) personnel.

[IMPORTANT OCONUS & MEDICARE RULES]
- 해외(South Korea 포함)에서는 TRICARE가 Medicare보다 우선 결제자(Primary Payer)입니다.
- Medicare는 해외 의료비를 보장하지 않습니다. (Medicare does NOT cover overseas medical expenses)
- 해외 청구는 먼저 본인이 지불(Pay-up-front)한 후 3년 이내에 청구해야 합니다.
- 해외 거주자도 Medicare Part B를 활성 상태로 유지해야 TRICARE For Life 혜택을 유지할 수 있습니다.

[답변 구성 원칙][cite: 14]
- 보장 여부 → 전제조건 → 비용(Group A/B 구분) → 절차 → 출처 순서로 답변하세요.
- 용어 고정: 본인부담금(Copay), 공제액(Deductible), 사전승인(Prior Authorization).
- 문서에 없는 내용은 "해당 내용은 제공된 문서에서 확인되지 않습니다."라고 답변하세요.[cite: 15]
"""

class TriCarePlugin(InsurancePlugin):

    def __init__(self):
        emb = get_embedding_model()
        
        # 1. 벡터 DB 로드 (텍스트 및 비용 테이블)
        self._text_db = Chroma(
            collection_name='tricare_rag',
            embedding_function=emb,
            persist_directory=str(ROOT / 'vectordb' / 'tricare_text')
        )
        self._table_db = Chroma(
            collection_name='tricare_cost_tables',
            embedding_function=emb,
            persist_directory=str(ROOT / 'vectordb' / 'tricare_table')
        )
        
        # 2. 고성능 재랭커 로드 (Source 15 기준)[cite: 15]
        self._reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
        
        # 3. BM25 인덱스 구축 (전체 텍스트 청크 대상)[cite: 15]
        self._text_chunks = self._load_all_docs(self._text_db)
        if self._text_chunks:
            self._bm25 = BM25Retriever.from_documents(self._text_chunks, k=6)
        else:
            self._bm25 = None

        print(f"✅ TriCare 업그레이드 완료 (텍스트: {len(self._text_chunks)}개 / 표: {self._table_db._collection.count()}개)")

    @property
    def name(self) -> str: return "TriCare"

    @property
    def system_prompt(self) -> str: return TRICARE_PROMPT

    def retrieve(self, query: str, normalized: dict, plan_or_intent: Optional[str]) -> List[Document]:
        intent = normalized.get('intent', 'general')
        region = normalized.get('region', 'unknown')
        
        # 1. 리전 필터 구성 (Source 14 방식 유지)[cite: 14]
        region_filter = self._build_region_filter(region)
        
        # 2. MMR 검색 (다양성 확보)[cite: 15]
        # 의도가 '비용'이나 '약국'인 경우 테이블 DB 가중치 증가
        if intent in ('cost', 'pharmacy'):
            table_docs = self._table_db.similarity_search(query, k=10)
            text_docs = self._text_db.max_marginal_relevance_search(query, k=5, fetch_k=20, filter=region_filter)
        else:
            text_docs = self._text_db.max_marginal_relevance_search(query, k=10, fetch_k=30, filter=region_filter)
            table_docs = self._table_db.similarity_search(query, k=5)

        # 3. BM25 키워드 검색 결합[cite: 15]
        bm25_docs = self._bm25.invoke(query) if self._bm25 else []
        
        # 4. RRF(Reciprocal Rank Fusion) 하이브리드 병합[cite: 15]
        candidates = self._rrf_merge([text_docs, table_docs, bm25_docs])
        
        # 5. [중요] search_tags 제거 후 CrossEncoder 재랭킹[cite: 15]
        if not candidates: return []
        
        refined_pairs = []
        for d in candidates:
            content = d.page_content
            if '[search_tags]' in content:
                content = content.split('[search_tags]')[0].strip() # 검색용 태그 제거 후 순수 본문으로만 평가
            refined_pairs.append((query, content))
            
        scores = self._reranker.predict(refined_pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in ranked[:8]]

    def _load_all_docs(self, db: Chroma) -> List[Document]:
        raw = db.get(include=['documents', 'metadatas'])
        return [Document(page_content=doc, metadata=meta) for doc, meta in zip(raw['documents'], raw['metadatas'])]

    def _build_region_filter(self, region: str) -> Optional[dict]:
        if region in ('unknown', 'CONUS', ''): return None
        if region in ('OCONUS', 'korea'): return {"location": {"$in": ["OCONUS", "BOTH"]}}
        return None

    def _rrf_merge(self, doc_lists, k=60) -> List[Document]:
        score_map = {}
        for docs in doc_lists:
            for rank, doc in enumerate(docs):
                key = doc.page_content[:100]
                if key not in score_map: score_map[key] = {'score': 0.0, 'doc': doc}
                score_map[key]['score'] += 1.0 / (rank + k)
        return [v['doc'] for v in sorted(score_map.values(), key=lambda x: x['score'], reverse=True)]