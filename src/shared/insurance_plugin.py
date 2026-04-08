# insurance_plugin.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from langchain_core.documents import Document
import re

class InsurancePlugin(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    def plans(self) -> List[str]:
        return []

    @abstractmethod
    def retrieve(
        self,
        query: str,
        normalized: dict,
        plan_or_intent: Optional[str],
        **kwargs
    ) -> List[Document]:
        pass

    @abstractmethod
    def analyze(
        self,
        question: str,
        context_str: str,
        state: Dict[str, Any],
    ) -> dict:
        """보험사별 분석. 표준 dict 반환."""
        pass
    
    @property
    def common_rules(self) -> str:
        return f"""
    [CRITICAL: ANTI-MIXING & CITATION RULE]
    1. **No plan/product mixing**: If retrieved documents refer to different insurance products, do NOT merge them into a single procedure. Clearly distinguish differences per product.
    2. **Mandatory citation & disclaimer**: At the end of every sentence, include the source as [Source]: [Insurance] [File name] [Year] [Page] along with a [Disclaimer].
    3. **No insurance recommendations**: Never recommend any insurance product under any circumstances.
    4. **No personal information**: Never collect or process any personally identifiable information that can be specified to a single person.
    5. **No speculation**: If information is absent from the documents, explicitly state "This information could not be confirmed in the provided documents." Do not guess.
    """

    @property
    def analyze_rules(self) -> str:
        return """
    [PLAN EXTRACTION RULES]
    - If two plans are compared ("A vs B", "A와 B 차이") → needs_clarification=true
      Ask which single plan they want to know about
    - Single plan mentioned → extract that plan
    
    [CLARIFICATION MESSAGE STYLE - CRITICAL]
    - If user says they don't know or can't remember:
      * Acknowledge it first with empathy
      * Explain WHY the info is needed
      * Offer an alternative path
    - NEVER use the exact same phrasing as the previous clarification message
    - Each clarification must feel like a natural conversation, not a form
    - clarification_message MUST be in the SAME language as the user message
    """

    PII_PATTERNS = [
        r'\b\d{6}[-]\d{7}\b',          # 주민번호
        r'\b[A-Z]{1,2}\d{7,9}\b',      # 여권번호
        r'\b\d{13}\b',                  # 외국인등록번호
        r'\b\d{4}[-]\d{2}[-]\d{2}\b',  # 생년월일
    ]
    
    PII_KEYWORDS = [
        "주민번호", "주민등록번호", "여권번호", "외국인등록번호",
        "social security", "ssn", "passport number", "date of birth",
        "insurance id", "member id", "policy number",
    ]
    
    RECOMMENDATION_KEYWORDS = [
        "추천", "어떤게 좋아", "뭐가 나아", "골라줘", "비교해줘",
        "recommend", "which is better", "어떤 플랜이 좋", "뭐가 더 좋",
        "더 나은", "뭐가 더", "어떤게 더",
    ]
    
    @staticmethod
    def check_blocked(question: str) -> Optional[str]:
        """추천/PII 감지. 차단 이유 반환, 없으면 None"""
        q_lower = question.lower()
        if any(kw in q_lower for kw in InsurancePlugin.RECOMMENDATION_KEYWORDS):
            return "recommendation"
        if any(kw in q_lower for kw in InsurancePlugin.PII_KEYWORDS):
            return "pii"
        if any(re.search(p, question) for p in InsurancePlugin.PII_PATTERNS):
            return "pii"
        return None