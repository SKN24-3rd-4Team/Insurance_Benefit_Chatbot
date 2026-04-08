"""
cigna_plugin.py
Cigna 전용 플러그인 — 난이도 기반 검색 (Dense/Hybrid/Multi-hop) + HyDE
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

CIGNA_SYSTEM_PROMPT = """당신은 Cigna Global 국제 건강보험 전문 안내 어시스턴트입니다.

=== 보험 용어 번역 고정 테이블 (Term Locking) ===
- Deductible → 공제액(Deductible)
- Co-insurance / Cost Share → 공동부담률(Co-insurance)
- Copay → 정액 본인부담(Copay)
- Out-of-Pocket Maximum → 최대 본인부담금(OOP Max)
- Prior Approval → 사전 승인(Prior Approval)
- In-network → 네트워크 내(In-network)
- Out-of-network → 네트워크 외(Out-of-network)
"""

PLAN_ALIASES = {
    "silver":    "Silver",
    "실버":       "Silver",
    "gold":      "Gold",
    "골드":       "Gold",
    "platinum":  "Platinum",
    "플래티넘":   "Platinum",
    "global silver":   "Global Silver",
    "global gold":     "Global Gold",
    "global platinum": "Global Platinum",
}

# 계산 질문 키워드 — 수치 정보가 있어야 답변 가능한 경우
CALCULATION_KEYWORDS = ["얼마", "계산", "환급", "본인부담", "how much", "calculate", "reimburse"]


class CignaPlugin(InsurancePlugin):

    def __init__(self):
        from langchain_chroma import Chroma
        from rank_bm25 import BM25Okapi

# ✅ 최종
        db_path = str(ROOT / 'vectordb' / 'cigna' / 'chroma_baai_latest')
        assert Path(db_path).exists(), f"❌ Cigna DB 없음: {db_path} — 전처리 후 임베딩을 먼저 실행하세요."
        self._db = Chroma(
            collection_name='cigna_latest',
            embedding_function=get_embedding_model(),
            persist_directory=db_path,
        )
        
        self._analyzer_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        self._llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

        # BM25 인덱스 초기화
        raw = self._db.get()
        self._all_docs = [
            Document(page_content=c, metadata=m)
            for c, m in zip(raw["documents"], raw["metadatas"])
        ]
        corpus = [d.page_content.lower().split() for d in self._all_docs]
        self._bm25 = BM25Okapi(corpus)
        print("✅ Cigna DB 로드 완료")

    @property
    def name(self) -> str:
        return "Cigna"

    @property
    def plans(self) -> List[str]:
        return ["Silver", "Gold", "Platinum", "Global Silver", "Global Gold", "Global Platinum"]

    @property
    def system_prompt(self) -> str:
        return CIGNA_SYSTEM_PROMPT

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

        # 계산 질문 + 플랜/수치 미제공 → missing_info 처리
        is_calc = any(kw in question for kw in CALCULATION_KEYWORDS)
        known_plan = state.get("plan_or_intent")

        prompt = f"""You are a Cigna Global insurance query analyzer.

[ALREADY KNOWN FROM STATE - DO NOT RE-ASK]
- known_plan: {known_plan or "unknown"}
- known_treatment: {state.get("known_treatment", "unknown")}

[CONVERSATION CONTEXT]
{context_str}

[CURRENT USER MESSAGE]
{question}

[PLAN LIST] Silver / Gold / Platinum / Global Silver / Global Gold / Global Platinum
Extract plan even from casual Korean: "실버야" → Silver, "골드인데" → Gold

[TREATMENT EXTRACTION]
Use medical knowledge to interpret any natural language description of symptoms or injuries.

[DIFFICULTY]
- low: 수치/정의 단순 조회
- medium: 절차/비교
- high: 계산/다중조건 (계산 질문인데 deductible/cost_share 수치가 없으면 needs_clarification=true)

[RULES]
- If plan unknown → needs_clarification=true, ask ONLY for plan
- If plan known → needs_clarification=false, generate english_query
- Never re-ask something already known

Return STRICT JSON only:
{{
  "language": "ko|en|ja|zh",
  "plan_or_intent": "plan name or null",
  "known_treatment": "english medical term or null",
  "difficulty": "low|medium|high",
  "missing_info": ["list of missing fields for calculation, or empty"],
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
            "extra": {
                "difficulty":    analysis.get("difficulty", "medium"),
                "missing_info":  analysis.get("missing_info", []),
                "followup_count": followup_count + (1 if analysis.get("needs_clarification") else 0),
            },
        }

    def _bm25_search(self, query: str, k: int = 5) -> List[Document]:
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._all_docs[i] for i in top_idx]

    def _hybrid(self, query: str, k: int = 5) -> List[Document]:
        bm25_res  = self._bm25_search(query, k=k * 2)
        dense_res = self._db.as_retriever(
            search_type='mmr', search_kwargs={'k': k, 'fetch_k': 20}
        ).invoke(query)

        scores, doc_map = {}, {}
        for rank, doc in enumerate(bm25_res, 1):
            key = doc.page_content[:80]
            scores[key] = scores.get(key, 0) + 0.4 / rank
            doc_map[key] = doc
        for rank, doc in enumerate(dense_res, 1):
            key = doc.page_content[:80]
            scores[key] = scores.get(key, 0) + 0.6 / rank
            doc_map[key] = doc

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[k_] for k_ in ranked[:k]]

    def _multihop(self, query: str, max_hop: int = 3) -> List[Document]:
        accumulated, current_query = [], query
        for hop in range(max_hop):
            new_docs = self._hybrid(current_query, k=3)
            accumulated.extend(new_docs)
            if hop < max_hop - 1:
                ctx = "\n".join(d.page_content[:200] for d in accumulated[:4])
                nq = self._llm.invoke(
                    f"원래질문: {query}\n현재컨텍스트:\n{ctx}\n"
                    "추가 검색이 필요하면 검색어를, 충분하면 DONE:"
                ).content.strip()
                if nq.upper() == "DONE":
                    break
                current_query = nq
        seen, unique = set(), []
        for d in accumulated:
            k = d.page_content[:80]
            if k not in seen:
                seen.add(k)
                unique.append(d)
        return unique

    def retrieve(self, query: str, normalized: dict, plan_or_intent: Optional[str], **kwargs) -> List[Document]:
        difficulty = kwargs.get("extra", {}).get("difficulty", "medium")

        if difficulty == "low":
            return self._db.similarity_search(query, k=5)
        elif difficulty == "high":
            return self._multihop(query)
        else:
            return self._hybrid(query, k=5)
    
