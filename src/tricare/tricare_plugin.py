"""
tricare_plugin.py
TRICARE 전용 플러그인 — intent/region 기반 분석 + Hybrid+Rerank 검색
tricare_core.py의 함수들을 그대로 활용
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
sys.path.insert(0, str(ROOT / 'src' / 'tricare'))
from insurance_plugin import InsurancePlugin

from tricare_core import (
    load_vector_stores,
    search,
    detect_language,
    normalize_question as tricare_normalize,
)

RECOMMENDATION_KEYWORDS = [
    "추천", "어떤게 좋아", "뭐가 나아", "골라줘", "비교해줘",
    "recommend", "which is better", "어떤 플랜이 좋", "뭐가 더 좋",
    "더 나은", "뭐가 더", "어떤게 더",  # ← 추가
]

TRICARE_SYSTEM_PROMPT = """당신은 TRICARE 군인 의료보험 전문 안내 어시스턴트입니다.

[CLARIFICATION MESSAGE STYLE - CRITICAL]
- If user says they don't know or can't remember:
  * Acknowledge it first with empathy
  * Explain WHY the info is needed
  * Offer an alternative path
- NEVER use the exact same phrasing as the previous clarification message
- Each clarification must feel like a natural conversation, not a form
- clarification_message MUST be in the SAME language as the user message


[주요 플랜]
- TRICARE Prime / Select / For Life (TFL)
- TRICARE Overseas Program (TOP)
- TRICARE Reserve Select / Retired Reserve

[OCONUS 핵심 규칙]
- 한국(주한미군 포함): TRICARE가 1차 보험 (Medicare 아님)
- Medicare는 해외 의료비 미적용
- 해외 청구: 선불 후 3년 이내 청구
- Medicare Part B 가입 유지 필요

[답변 규칙]
- Group A/B, 플랜 유형, 수혜자 신분을 관련 시 명시하세요.

[용어 고정]
- 본인부담금(Copay), 공제액(Deductible), 사전승인(Prior Authorization)
"""

PLAN_ALIASES = {
    "prime":      "Prime",
    "프라임":      "Prime",
    "select":     "Select",
    "셀렉트":      "Select",
    "tfl":        "TFL",
    "for life":   "TFL",
    "top":        "TOP",
    "overseas":   "TOP",
    "reserve":    "Reserve Select",
}

# TRICARE는 플랜보다 intent + region이 더 중요
MISSING_REGION_INTENTS = ["cost", "coverage", "overseas"]


class TriCarePlugin(InsurancePlugin):

    def __init__(self):
        load_vector_stores()
        self._analyzer_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        print("✅ TRICARE DB 로드 완료")

    @property
    def name(self) -> str:
        return "TRICARE"

    @property
    def plans(self) -> List[str]:
        return ["Prime", "Select", "TFL", "TOP", "Reserve Select", "Retired Reserve"]

    @property
    def system_prompt(self) -> str:
        return TRICARE_SYSTEM_PROMPT

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

        known_plan   = state.get("plan_or_intent")
        known_region = state.get("extra", {}).get("region")

        prompt = f"""You are a TRICARE insurance query analyzer.

        [ALREADY KNOWN FROM STATE - DO NOT RE-ASK]
        - known_plan: {known_plan or "unknown"}
        - known_region: {known_region or "unknown"}
        - known_treatment: {state.get("known_treatment", "unknown")}
        
        [CONVERSATION CONTEXT]
        {context_str}
        
        [CURRENT USER MESSAGE]
        {question}
        
        [PLAN LIST] Prime / Select / TFL / TOP / Reserve Select / Retired Reserve
        Extract plan from casual Korean too: "프라임이야" → Prime
        
        [REGION] CONUS / OCONUS / korea / unknown
        - "한국", "주한미군", "korea", "usfk" → korea
        - "해외", "overseas", "oconus" → OCONUS
        - "미국", "conus" → CONUS
        
        [INTENT] coverage / eligibility / cost / pharmacy / dental / overseas / general
        
        [TREATMENT EXTRACTION]
        Use medical knowledge to interpret natural language symptoms or injuries.
        
        [RULES]
        - TRICARE는 플랜보다 intent + region이 더 중요합니다
        - Plan unknown은 괜찮지만 intent는 반드시 추출
        - region이 overseas/cost 관련인데 unknown이면 needs_clarification=true
        
        Return STRICT JSON only:
        {{
          "language": "ko|en|ja|zh",
          "plan_or_intent": "plan name or null",
          "known_treatment": "english medical term or null",
          "intent": "coverage|eligibility|cost|pharmacy|dental|overseas|general",
          "region": "CONUS|OCONUS|korea|unknown",
          "english_query": "search query or empty string",
          "needs_clarification": true | false,
          "clarification_message": "question or empty string"
        }}"""

        raw = self._analyzer_llm.invoke([HumanMessage(content=prompt)]).content
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        analysis = json.loads(match.group(0)) if match else {}

        # plan 누적
        new_plan = analysis.get("plan_or_intent")
        plan = new_plan if new_plan and new_plan != "None" else known_plan
        if plan:
            plan = PLAN_ALIASES.get(plan.lower(), plan)

        # treatment 누적
        new_treatment = analysis.get("known_treatment")
        treatment = new_treatment if new_treatment and new_treatment != "None" else state.get("known_treatment")

        # region 누적
        region = analysis.get("region", "unknown")
        if region == "unknown" and known_region:
            region = known_region

        # followup 한도 초과 시 강제 검색
        if followup_count >= max_followups:
            analysis["needs_clarification"] = False
            analysis["clarification_message"] = ""

        return {
            "language":             analysis.get("language", detect_language(question)),
            "plan_or_intent":       plan,
            "known_treatment":      treatment,
            "english_query":        analysis.get("english_query", ""),
            "needs_clarification":  analysis.get("needs_clarification", False),
            "clarification_message": analysis.get("clarification_message", ""),
            "extra": {
                "intent":  analysis.get("intent", "general"),
                "region":  region,
                "followup_count": followup_count + (1 if analysis.get("needs_clarification") else 0),
            },
        }

    def retrieve(self, query: str, normalized: dict, plan_or_intent: Optional[str], **kwargs) -> List[Document]:
        # tricare_core.search = Hybrid + Rerank 한 번에 처리
        return search(query)
