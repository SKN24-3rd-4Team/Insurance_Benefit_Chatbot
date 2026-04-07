"""
Bupa 플러그인.
vectordb/bupa/ 폴더의 ChromaDB를 참조.
"""
import sys
from pathlib import Path
from typing import Optional, List
from langchain_chroma import Chroma
from langchain_core.documents import Document

# shared 모듈 경로 추가
ROOT = Path(__file__).resolve().parent.parent.parent  # Insurance_Benefit_Chatbot/
sys.path.insert(0, str(ROOT / 'src' / 'shared'))

from insurance_plugin import InsurancePlugin
from shared_embedding import get_embedding_model

BUPA_PROMPT = """당신은 Bupa 국제 의료보험 전문 상담사입니다.

[중요: 플랜별 판단 로직]
1. Premier, Elite, Select, Major Medical 플랜:
   - 이 플랜들은 모듈형이 아닙니다.
   - [Documents]에 'Paid in full', 'Full refund', 'will be paid'라고 적혀 있으면 무조건 "100% 보장"입니다.
   - 절대로 "모듈이 필요하다"거나 "보장되지 않는다"고 하지 마세요.

2. IHHP (International Health Insurance Plan) 플랜:
   - 이 플랜만 유일하게 모듈형입니다.
   - 문서에 '※ 모듈형 표' 또는 '[일부 모듈 가입 시 보장]' 태그가 있으면 "특정 모듈 가입 시 보장"으로 안내하세요.
   - Not Covered만 보고 전체 미보장으로 판단하지 마세요.

[답변 구성 원칙]
- 보장 여부 → 필요한 조건 → 안 되는 케이스 → 금액 한도 → 출처 순서로 자연스러운 대화체로 작성하세요.
- 문서를 참조한 경우, 어느 문서 몇 페이지를 참고했는지 명시해주세요.

[전문 용어]
- Deductible/Excess: 자기부담금
- Co-insurance: 공동부담률
- Pre-authorization: 사전승인
- Paid in full: 100% 보장
- Waiting period: 보장 대기 기간
- Pre-existing condition: 기존 질환

[불명확한 질문 처리]
- 지시어만 있는 경우("이거"): "어떤 치료/항목에 대해 문의하시는 건가요?" 라고 되물으세요.
- 보험 추천이나 법적/의학적 최종 판단은 하지 마세요.
- 플랜이 없는 경우 : "어떤 플랜에 가입하셨나요?" 라고 확인하세요.
- 문서에 없는 내용은 추측하지 말고 "문서에 명시되지 않았습니다"라고 하세요.
"""


class BupaPlugin(InsurancePlugin):

    def __init__(self):
        db_path = str(ROOT / 'vectordb' / 'bupa')
        self._db = Chroma(
            collection_name='bupa_preprocessed',
            embedding_function=get_embedding_model(),
            persist_directory=db_path
        )
        print(f"✅ Bupa DB 로드: {self._db._collection.count()}개")

    @property
    def name(self) -> str:
        return "Bupa"

    @property
    def system_prompt(self) -> str:
        return BUPA_PROMPT

    def retrieve(self, query, normalized, plan_or_intent) -> List[Document]:
        plan    = plan_or_intent
        section = normalized.get('section_type')

        filters = []
        if plan and plan != "None":
            filters.append({"plan_tier": plan})
        if section:
            filters.append({"section_type": section})

        search_filter = (
            {"$and": filters} if len(filters) == 2
            else filters[0] if filters
            else None
        )

        docs = self._db.similarity_search(query, k=8, filter=search_filter)
        if not docs:
            docs = self._db.similarity_search(query, k=8)
        return docs
