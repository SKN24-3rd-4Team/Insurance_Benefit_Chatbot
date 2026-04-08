"""
bupa_plugin.py
Bupa 전용 플러그인 — plan_tier 메타필터 + BM25/MMR 하이브리드 검색
"""
import json
import re
import sys
from pathlib import Path
from typing import Optional, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'shared'))
from insurance_plugin import InsurancePlugin
from shared_embedding import get_embedding_model

RECOMMENDATION_KEYWORDS = [
    "추천", "어떤게 좋아", "뭐가 나아", "골라줘", "비교해줘",
    "recommend", "which is better", "어떤 플랜이 좋", "뭐가 더 좋",
    "더 나은", "뭐가 더", "어떤게 더",  # ← 추가
]

BUPA_SYSTEM_PROMPT = """You are a Bupa insurance query analyzer.

[CLARIFICATION MESSAGE STYLE - CRITICAL]
- If user says they don't know or can't remember:
  * Acknowledge it first with empathy
  * Explain WHY the info is needed
  * Offer an alternative path
- NEVER use the exact same phrasing as the previous clarification message
- Each clarification must feel like a natural conversation, not a form
- clarification_message MUST be in the SAME language as the user message


[플랜 목록]
- Elite / Premier / Select / MajorMedical / IHHP
"""

PLAN_ALIASES = {
    "elite":        "Elite",
    "premier":      "Premier",
    "프리미어":      "Premier",
    "select":       "Select",
    "셀렉트":        "Select",
    "major":        "MajorMedical",
    "majormedical": "MajorMedical",
    "메이저":        "MajorMedical",
    "ihhp":         "IHHP",
    "아이에이치":     "IHHP",
}

SECTION_TYPE_MAP = {
    "exclusion":     ["안 되는", "제외", "미보장", "exclusion", "not covered"],
    "claim_process": ["청구", "claim", "서류", "reimbursement", "환급"],
    "pre_auth":      ["사전승인", "pre-auth", "prior auth", "prior approval"],
    "benefit_table": ["보장", "한도", "얼마", "cover", "limit", "benefit"],
}


class BupaPlugin(InsurancePlugin):

    def __init__(self):
        from langchain_chroma import Chroma
        db_path = str(ROOT / 'vectordb' / 'bupa' / 'bupa_latest')
        assert Path(db_path).exists(), f"❌ Bupa DB 없음: {db_path} — 전처리 후 임베딩을 먼저 실행하세요."
        self._db = Chroma(
            collection_name='bupa_latest',
            embedding_function=get_embedding_model(),
            persist_directory=db_path,
        )
        self._analyzer_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        print("✅ Bupa DB 로드 완료")

    @property
    def name(self) -> str:
        return "Bupa"

    @property
    def plans(self) -> List[str]:
        return ["Elite", "Premier", "Select", "MajorMedical", "IHHP"]

    @property
    def system_prompt(self) -> str:
        return BUPA_SYSTEM_PROMPT

    def analyze(self, question: str, context_str: str, state: dict = None) -> dict:
        state = state or {}
        followup_count = state.get("followup_count", 0)
        max_followups  = state.get("max_followups", 2)

        # [PRIORITY 0] 추천 방어
        if any(kw in question.lower() for kw in RECOMMENDATION_KEYWORDS):
            return {
                "language": "ko",
                "plan_or_intent": state.get("plan_or_intent"),
                "known_treatment": state.get("known_treatment"),
                "english_query": "",
                "needs_clarification": True,
                "clarification_message": (
                    "보험 추천은 법적으로 제공이 불가하며, "
                    "가입하신 플랜의 보장 내용만 안내 가능합니다."
                ),
                "extra": {},
            }

        prompt = f"""You are a Bupa insurance query analyzer.

        [ALREADY KNOWN FROM STATE - DO NOT RE-ASK]
        - known_plan: {state.get("plan_or_intent", "unknown")}
        - known_treatment: {state.get("known_treatment", "unknown")}

        [CONVERSATION CONTEXT]
        {context_str}

        [CURRENT USER MESSAGE]
        {question}

        [PLAN LIST] Elite / Premier / Select / MajorMedical / IHHP
        Extract plan even from casual Korean: "셀렉트야" → Select, "프리미어인데" → Premier

        [TREATMENT EXTRACTION]
        Use medical knowledge to interpret any natural language description.
        "다리 부러진 것 같아" → leg fracture, "숨쉬기 힘들어" → respiratory distress

        [SECTION TYPE] Pick one: benefit_table / exclusion / claim_process / pre_auth / general

        [RULES]
        - If plan is still unknown → needs_clarification=true, ask ONLY for plan
        - If plan is known → needs_clarification=false, generate english_query
        - Never re-ask something already known

        Return STRICT JSON only:
        {{
          "language": "ko|en|ja|zh",
          "plan_or_intent": "plan name or null",
          "known_treatment": "english medical term or null",
          "section_type": "benefit_table|exclusion|claim_process|pre_auth|general",
          "english_query": "search query or empty string",
          "needs_clarification": true | false,
          "clarification_message": "question or empty string"
        }}"""

        raw = self._analyzer_llm.invoke([HumanMessage(content=prompt)]).content
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        analysis = json.loads(match.group(0)) if match else {}

        # plan 누적
        new_plan = analysis.get("plan_or_intent")
        plan = new_plan if new_plan and new_plan != "None" else state.get("plan_or_intent")
        if plan:
            plan = PLAN_ALIASES.get(plan.lower(), plan)

        # treatment 누적
        new_treatment = analysis.get("known_treatment")
        treatment = new_treatment if new_treatment and new_treatment != "None" else state.get("known_treatment")

        if followup_count >= max_followups:
            analysis["needs_clarification"] = False
            analysis["clarification_message"] = ""


        return {
            "language":             analysis.get("language", "ko"),
            "plan_or_intent":       plan,
            "known_treatment":      treatment,
            "english_query":        analysis.get("english_query", ""),
            "needs_clarification":  analysis.get("needs_clarification", False),
            "clarification_message": analysis.get("clarification_message", ""),
            "extra": {"section_type": analysis.get("section_type", "benefit_table"),
                      "followup_count": followup_count + (1 if analysis.get("needs_clarification") else 0),
                    },
        }

    def retrieve(self, query: str, normalized: dict, plan_or_intent: Optional[str], **kwargs) -> List[Document]:
        from langchain_community.retrievers import BM25Retriever
        section_type = kwargs.get("extra", {}).get("section_type", "benefit_table")

        # 메타필터: plan_tier + section_type
        search_filter = {}
        conditions = []
        if plan_or_intent and plan_or_intent != "None":
            conditions.append({"plan_tier": {"$eq": plan_or_intent}})
        if section_type and section_type != "general":
            conditions.append({"section_type": {"$eq": section_type}})

        if len(conditions) == 2:
            search_filter = {"$and": conditions}
        elif len(conditions) == 1:
            search_filter = conditions[0]

        # Dense 검색
        try:
            dense_docs = self._db.similarity_search(query, k=8, filter=search_filter or None)
        except Exception:
            dense_docs = self._db.similarity_search(query, k=8)

        # BM25
        try:
            raw = self._db.get()
            all_docs = [
                Document(page_content=c, metadata=m)
                for c, m in zip(raw["documents"], raw["metadatas"])
            ]
            bm25 = BM25Retriever.from_documents(all_docs, k=8)
            bm25_docs = bm25.invoke(query)
        except Exception:
            bm25_docs = []

        # RRF 병합
        scores, doc_map = {}, {}
        for rank, doc in enumerate(bm25_docs, 1):
            key = doc.page_content[:80]
            scores[key] = scores.get(key, 0) + 0.4 * (1 / rank)
            doc_map[key] = doc
        for rank, doc in enumerate(dense_docs, 1):
            key = doc.page_content[:80]
            scores[key] = scores.get(key, 0) + 0.6 * (1 / rank)
            doc_map[key] = doc

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[k] for k in ranked[:6]]
