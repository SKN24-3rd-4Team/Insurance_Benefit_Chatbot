import json
import re
import sys
from pathlib import Path
from typing import Annotated, Optional, Dict, Any, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 같은 shared 패키지의 plugin import
sys.path.insert(0, str(Path(__file__).parent))
from insurance_plugin import InsurancePlugin


class InsuranceState(TypedDict):
    messages:         Annotated[list[BaseMessage], add_messages]
    plan_or_intent:   Optional[str]
    normalized_query: Dict[str, Any]
    retrieved_docs:   List[str]
    current_question: str
    clarification_msg: str 
    known_treatment: Optional[str]
    slots:             Dict[str, Any]   # Allianz slot 시스템용
    followup_count:    int              # followup 횟수 추적
    max_followups:     int              # 최대 followup 허용 수
    extra:             Dict[str, Any]   # 보험사별 추가 데이터


class InsuranceRAGGraph:

    def __init__(self, plugin: InsurancePlugin, model_name: str = "gpt-4o-mini"):
        self.plugin       = plugin
        self.chat_llm     = ChatOpenAI(model=model_name, temperature=0.1)

    def analyze_node(self, state: InsuranceState) -> dict:
        messages = state['messages']
        question = messages[-1].content

        history = []
        for m in messages[:-1]:
            role = "User" if isinstance(m, HumanMessage) else "AI"
            history.append(f"{role}: {m.content}")
        context_str = "\n".join(history[-5:])
        context_str += "\n\n" + self.plugin.clarification_style  # 공통 규칙 추가


        # ── 각 플러그인의 analyze() 호출 ──────────────────────────
        analysis = self.plugin.analyze(
            question=question,
            context_str=context_str,
            state={
                "plan_or_intent":    state.get("plan_or_intent"),
                "known_treatment":   state.get("known_treatment"),
                "slots":             state.get("slots", {}),
                "needs_clarification": False,
                "clarification_message": state.get("clarification_msg", ""),
                "followup_count":    state.get("followup_count", 0),
                "max_followups":     state.get("max_followups", 2),
                "extra":             state.get("extra", {}),
            }
        )

        # plan / treatment 누적 (플러그인이 이미 처리했지만 state 반영)
        final_plan      = analysis.get("plan_or_intent")
        final_treatment = analysis.get("known_treatment")

        return {
            "normalized_query":  analysis,
            "plan_or_intent":    final_plan,
            "known_treatment":   final_treatment,
            "current_question":  question,
            "clarification_msg": analysis.get("clarification_message", ""),
            "slots":             analysis.get("extra", {}).get("slots", state.get("slots", {})),
            "followup_count":    analysis.get("extra", {}).get("followup_count", state.get("followup_count", 0)),
            "extra":             analysis.get("extra", {}),
        }

    
    def retrieve_node(self, state: InsuranceState) -> dict:
        query = state['normalized_query'].get('english_query') or state['current_question']
        plan  = state.get('plan_or_intent')
    
        docs = self.plugin.retrieve(
            query=query,
            normalized=state['normalized_query'],
            plan_or_intent=plan,
            extra=state.get('extra', {}),
        )
    
        formatted = []
        for d in docs:
            m = d.metadata
            # 보험사별 파일명 키 통합
            file_name = (
                m.get('source_file')      # tricare
                or m.get('file_name')     # cigna
                or m.get('source', 'unknown.pdf')  # bupa, allianz
            )
            file_name = Path(file_name).name
            page_num  = m.get('page', '?')
            formatted.append(f"[Source: {file_name} / Page: {page_num}] {d.page_content}")
    
        return {"retrieved_docs": formatted}

    def generate_node(self, state: InsuranceState) -> dict:
        context  = "\n\n".join(state['retrieved_docs'])
        language = state.get('normalized_query', {}).get('language', 'ko')  # 🚨 변수명 수정

        lang_map = {
            'ko': '한국어', 'en': 'English', 'ja': '日本語',
            'zh': '中文', 'fr': 'Français', 'de': 'Deutsch', 'es': 'Español',
        }

        answer_language = lang_map.get(language, '한국어')  # 🚨 language_code 에러 수정

        common_rules = f"""
        [CRITICAL: LANGUAGE RULE]
        - You MUST respond entirely in {answer_language}. 
        - 답변은 반드시 {answer_language}로 작성하세요.
        """
        common_rules +=  "\n\n" + self.plugin.common_rules

        messages = [
            SystemMessage(content=common_rules + self.plugin.system_prompt),
            *state['messages'][:-1],
            HumanMessage(
                content=f"[참조 문서 리스트]\n{context}\n\n"
                        f"[사용자 질문]\n{state['current_question']}\n\n"
                        f"명령: 위 문서들을 바탕으로 질문에 답하되, 모든 문장에 정확한 (출처: 파일명, p.번호)를 포함하세요."
            ),
        ]
        response = self.chat_llm.invoke(messages)
        return {"messages": [response]}
    
    def clarify_node(self, state: InsuranceState) -> dict:
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=state['clarification_msg'])]}
    
    def route_after_analyze(self, state: InsuranceState) -> str:
            analysis = state.get('normalized_query', {})
            
            # 플랜이 없어서 LLM이 되물어야 한다고 판단했다면, 검색하지 않고 질문만 던집니다.
            if analysis.get('needs_clarification', False):
                return "clarify"
                
            return "retrieve"
    
    def build(self):
        b = StateGraph(InsuranceState)
        b.add_node("analyze",  self.analyze_node)
        b.add_node("clarify",  self.clarify_node)
        b.add_node("retrieve", self.retrieve_node)
        b.add_node("generate", self.generate_node)

        b.add_edge(START, "analyze")
        b.add_conditional_edges(
            "analyze",
            self.route_after_analyze,
            {"clarify": "clarify", "retrieve": "retrieve"}
        )
        b.add_edge("clarify",  END)
        b.add_edge("retrieve", "generate")
        b.add_edge("generate", END)
        return b.compile(checkpointer=MemorySaver())