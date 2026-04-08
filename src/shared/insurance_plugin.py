# insurance_plugin.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from langchain_core.documents import Document


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

    [DISCLAIMER RULE]
    If your answer includes any disclaimer, limitation, or caveat about coverage accuracy,
    always append the following sentence at the very end of your response,
    translated into the same language as your answer:
    "⚠️ For final plan selection, please consult directly with your insurance provider based on your personal circumstances and coverage needs."
    This must appear as the last line, after all other content."
    """

    @property
    def clarification_style(self) -> str:
        return """
    [CLARIFICATION MESSAGE STYLE - CRITICAL]
    - If user says they don't know or can't remember:
      * Acknowledge it first with empathy
      * Explain WHY the info is needed
      * Offer an alternative path
    - NEVER use the exact same phrasing as the previous clarification message
    - Each clarification must feel like a natural conversation, not a form
    - clarification_message MUST be in the SAME language as the user message
    """