"""
allianz_plugin.py
Allianz 전용 플러그인 — slot 기반 분석 + Hybrid/RRF/Rerank 검색
rag_utils.py의 함수들을 그대로 활용
"""
import sys
from pathlib import Path
from typing import Optional, List

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'shared'))
sys.path.insert(0, str(ROOT / 'src' / 'allianz'))
from src.shared.insurance_plugin import InsurancePlugin

from rag_utils import (
    normalize_question,
    extract_slots_llm,
    merge_slots,
    decide_missing_slots,
    build_followup_question_llm,
    retrieve_documents_from_slots,
    looks_like_followup_answer,
    KNOWN_PLANS,
)

RECOMMENDATION_KEYWORDS = [
    "추천", "어떤게 좋아", "뭐가 나아", "골라줘", "비교해줘",
    "recommend", "which is better", "어떤 플랜이 좋", "뭐가 더 좋",
    "더 나은", "뭐가 더", "어떤게 더",  # ← 추가
]

ALLIANZ_SYSTEM_PROMPT = """You are an Allianz Care insurance document-based assistant.

Answer ONLY based on the provided context.

[CLARIFICATION MESSAGE STYLE - CRITICAL]
- If user says they don't know or can't remember:
  * Acknowledge it first with empathy
  * Explain WHY the info is needed
  * Offer an alternative path
- NEVER use the exact same phrasing as the previous clarification message
- Each clarification must feel like a natural conversation, not a form
- clarification_message MUST be in the SAME language as the user message

[Allianz Care 플랜]
- Care Base / Care Enhanced / Care Signature

[답변 구성 원칙]
1. 결론 (보장 여부 또는 절차 요약)
2. 지역별 근거
3. 일반/글로벌 규정

[전문 용어]
- Pre-authorisation: 사전승인
- TOB (Table of Benefits): 보장 항목 표
- Deductible: 자기부담금
- Co-insurance: 공동부담률
"""


class AllianzPlugin(InsurancePlugin):

    @property
    def name(self) -> str:
        return "Allianz"

    @property
    def plans(self) -> List[str]:
        return ["Care Base", "Care Enhanced", "Care Signature"]

    @property
    def system_prompt(self) -> str:
        return ALLIANZ_SYSTEM_PROMPT

    def analyze(self, question: str, context_str: str, state: dict = None) -> dict:
        state = state or {}

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

        # [PRIORITY 1] 슬롯 추출 및 누적
        old_slots = state.get("slots", {})
        is_followup = state.get("needs_clarification", False) or looks_like_followup_answer(question)

        # 질문 정규화
        normalized = normalize_question(question)

        # 슬롯 추출 + 병합
        new_slots = extract_slots_llm(
            question,
            existing_slots=old_slots,
            pending_followup=is_followup,
            last_followup_question=state.get("clarification_message", ""),
        )
        print(f"추출된 슬롯: {new_slots}")
        merged_slots = merge_slots(old_slots, new_slots)

        # plan 정규화 및 누적
        plan = merged_slots.get("plan") or state.get("plan_or_intent")
        if plan:
            plan = KNOWN_PLANS.get(plan.lower(), plan)
        merged_slots["plan"] = plan

        # treatment 누적
        known_treatment = (
            merged_slots.get("injury_or_condition")
            or state.get("known_treatment")
        )

        # missing slots 판단
        intent = merged_slots.get("intent", normalized.get("intent", ""))
        missing_slots = decide_missing_slots(intent, merged_slots, question)

        # followup 한도 초과 시 강제 검색
        followup_count = state.get("followup_count", 0)
        if followup_count >= state.get("max_followups", 2):
            missing_slots = []

        print(f'missing_slots: {missing_slots}, followup_count: {followup_count}')
        needs_clarification = bool(missing_slots)
        clarification_message = ""
        if needs_clarification:
            clarification_message = build_followup_question_llm(
                language=normalized.get("language", "ko"),
                missing_slots=missing_slots,
                intent=intent,
                slots=merged_slots,
            )
            print(f"추가 질문: {clarification_message}")
        english_query = normalized.get("english_query", "") if not needs_clarification else ""

        return {
            "language":             normalized.get("language", "ko"),
            "plan_or_intent":       plan,
            "known_treatment":      known_treatment,
            "english_query":        english_query,
            "needs_clarification":  needs_clarification,
            "clarification_message": clarification_message,
            "extra": {
                "slots":      merged_slots,
                "normalized": normalized,
                "followup_count": followup_count + (1 if needs_clarification else 0),
            },
        }

    def retrieve(self, query: str, normalized: dict, plan_or_intent: Optional[str], **kwargs) -> List[Document]:
        slots = kwargs.get("extra", {}).get("slots", {})
        _normalized = kwargs.get("extra", {}).get("normalized", normalized)

        if plan_or_intent:
            slots["plan"] = plan_or_intent

        docs, _ = retrieve_documents_from_slots(
            question=query,
            normalized=_normalized,
            slots=slots,
            use_latest_only=False,
        )
        return docs
