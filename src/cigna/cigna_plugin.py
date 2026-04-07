import sys
from pathlib import Path
from typing import Optional, List
from langchain_chroma import Chroma
from langchain_core.documents import Document

# shared 모듈 경로 추가
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'shared'))

from insurance_plugin import InsurancePlugin
from shared_embedding import get_embedding_model

# Cigna 전용 용어 고정 및 프롬프트
CIGNA_PROMPT = """당신은 Cigna Global 국제 건강보험 전문 안내 어시스턴트입니다.

=== 보험 용어 번역 고정 테이블 (Term Locking) ===
- Deductible → 공제액(Deductible)
- Co-insurance / Cost Share → 공동부담률(Co-insurance)
- Copay → 정액 본인부담(Copay)
- Out-of-Pocket Maximum → 최대 본인부담금(OOP Max)
- Prior Approval → 사전 승인(Prior Approval)
- In-network → 네트워크 내(In-network)
- Out-of-network → 네트워크 외(Out-of-network)

[답변 규칙]
1. 반드시 [참조 문서]의 내용에만 근거하여 답변하세요.
2. 위 '용어 고정 테이블'에 정의된 용어를 우선적으로 사용하세요.
3. 모든 문장 끝에 (출처: 파일명, p.번호)를 명시하세요.
4. 보험 상품 추천이나 "이게 더 좋다"는 식의 비교는 절대 하지 마세요.
5. 문서에 정보가 없으면 "확인 불가"라고 답변하세요.
"""

class CignaPlugin(InsurancePlugin):
    def __init__(self):
        db_path = str(ROOT / 'vectordb' / 'cigna')
        self._db = Chroma(
            collection_name='cigna_collection',
            embedding_function=get_embedding_model(),
            persist_directory=db_path
        )
        print(f"✅ Cigna DB 로드 완료")

    @property
    def name(self) -> str:
        return "Cigna"

    @property
    def system_prompt(self) -> str:
        return CIGNA_PROMPT

    def retrieve(self, query, normalized, plan_or_intent) -> List[Document]:
        # Cigna 소스의 Hybrid Retrieval 로직을 단순화하여 적용
        # 필요 시 소스의 BM25 로직을 여기에 추가할 수 있습니다.
        docs = self._db.similarity_search(query, k=6)
        return docs