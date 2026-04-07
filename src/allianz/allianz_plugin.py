"""
Allianz 플러그인.
vectordb/allianz/ 참조.
Vector + BM25 + RRF + CrossEncoder 재랭킹.
"""
import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'shared'))

from insurance_plugin import InsurancePlugin
from shared_embedding import get_embedding_model

ALLIANZ_PROMPT = """You are an Allianz Care insurance document-based assistant.

Answer ONLY based on the provided context.
Do not guess unsupported facts.

[Allianz Care 플랜]
- Care Base / Care Enhanced / Care Signature

[답변 구성 원칙]
1. 결론 (보장 여부 또는 절차 요약)
2. 지역별 근거
3. 일반/글로벌 규정
4. 절차 또는 주의사항
5. 출처

[전문 용어]
- Pre-authorisation: 사전승인
- TOB (Table of Benefits): 보장 항목 표
- Deductible: 자기부담금
- Co-insurance: 공동부담률

[주의]
- 보험 추천이나 법적/의학적 최종 판단은 하지 마세요.
- 문서에 없으면 "문서에 명시되지 않았습니다"라고 하세요.
"""

DOC_TYPE_MAP = {
    'coverage': ['benefit_guide', 'tob'],
    'preauth':  ['benefit_guide', 'preauth_form', 'tob'],
    'claim':    ['benefit_guide', 'claim_form'],
}


class AllianzPlugin(InsurancePlugin):

    def __init__(self):
        db_path = str(ROOT / 'vectordb' / 'allianz')
        self._db = Chroma(
            collection_name='allianz_care',
            embedding_function=get_embedding_model(),
            persist_directory=db_path
        )
        self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # BM25 인덱스
        self._all_docs = self._load_all_docs()
        if self._all_docs:
            tokenized  = [d.page_content.lower().split() for d in self._all_docs]
            self._bm25 = BM25Okapi(tokenized)
        else:
            print("⚠️  Allianz DB가 비어있음 → BM25 비활성화")
            self._bm25 = None

        print(f"✅ Allianz DB 로드: {self._db._collection.count()}개")

    @property
    def name(self) -> str:
        return "Allianz"

    @property
    def system_prompt(self) -> str:
        return ALLIANZ_PROMPT

    def retrieve(self, query, normalized, plan_or_intent) -> List[Document]:
        intent = normalized.get('intent', 'coverage')
        region = normalized.get('region', 'none')
        
        # 1. 확장 쿼리 생성 (Source 11 로직 적용)
        expanded_queries = self._make_expanded_queries(query, normalized)
        
        # 2. 필터 구성 (Global 문서 포함 전략)
        search_filter = self._build_filter(intent, region)
        
        hybrid_pool = {} # 중복 방지 및 점수 합산용
        
        for q in expanded_queries:
            # 3. Dense (Vector) 검색
            dense_docs = self._db.similarity_search(q, k=10, filter=search_filter)
            for rank, d in enumerate(dense_docs, 1):
                self._update_pool(hybrid_pool, d, rank, 'dense')
            
            # 4. Sparse (BM25) 검색
            if self._bm25:
                bm25_res = self._bm25_search(q, k=10) # 필터가 적용된 BM25 필요
                for rank, d in enumerate(bm25_res, 1):
                    self._update_pool(hybrid_pool, d, rank, 'bm25')

        # 5. RRF 점수 계산 + Rule-based 가중치 합산
        scored_docs = []
        for key, item in hybrid_pool.items():
            doc = item['doc']
            # RRF 점수 (60은 상수 k)
            dense_score = 1 / (60 + item['dense_rank']) if item['dense_rank'] else 0
            bm25_score = 1 / (60 + item['bm25_rank']) if item['bm25_rank'] else 0
            
            # 원본 로직의 규칙 점수 반영
            rule_extra = self._calculate_rule_score(doc, intent, region)
            
            final_score = (0.65 * dense_score) + (0.35 * bm25_score) + rule_extra
            scored_docs.append((doc, final_score))

        # 6. 정렬 및 CrossEncoder 재랭킹
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [d for d, s in scored_docs[:15]]
        
        return self._final_rerank(query, top_candidates)

    def _load_all_docs(self) -> List[Document]:
        from langchain_core.documents import Document as LCDoc
        result = self._db.get()
        return [
            LCDoc(page_content=c, metadata=m)
            for c, m in zip(result['documents'], result['metadatas'])
        ]

    def _build_filter(self, intent: str, region: str) -> Optional[dict]:
        allowed_types = DOC_TYPE_MAP.get(intent, ['benefit_guide', 'tob'])
        filters = [{"doc_type": {"$in": allowed_types}}]
        if region and region not in ('none', 'global', ''):
            filters.append({"region": {"$in": [region, 'global']}})
        return {"$and": filters} if len(filters) == 2 else filters[0]

    def _rrf_merge(self, doc_lists, k=60) -> List[Document]:
        score_map: Dict[str, Dict] = {}
        for docs in doc_lists:
            for rank, doc in enumerate(docs):
                key = doc.page_content[:100]
                if key not in score_map:
                    score_map[key] = {'score': 0.0, 'doc': doc}
                score_map[key]['score'] += 1.0 / (rank + k)
        return [
            v['doc']
            for v in sorted(score_map.values(), key=lambda x: x['score'], reverse=True)
        ]
