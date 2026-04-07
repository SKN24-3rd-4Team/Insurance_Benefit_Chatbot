from abc import ABC, abstractmethod
from typing import Optional, List
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

    @abstractmethod
    def retrieve(
        self,
        query: str,
        normalized: dict,
        plan_or_intent: Optional[str]
    ) -> List[Document]:
        pass

    def build_answer_prompt(self, language: str) -> str:
        lang_map = {
            'ko': '한국어', 'en': 'English', 'ja': '日本語',
            'zh': '中文', 'fr': 'Français', 'de': 'Deutsch', 'es': 'Español',
        }
        return f"\n\n[언어 규칙] 반드시 {lang_map.get(language, '한국어')}로 답변하세요."
