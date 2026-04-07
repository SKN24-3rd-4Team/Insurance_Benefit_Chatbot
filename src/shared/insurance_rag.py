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


class InsuranceRAGGraph:

    def __init__(self, plugin: InsurancePlugin, model_name: str = "gpt-4o-mini"):
        self.plugin       = plugin
        self.analyzer_llm = ChatOpenAI(model=model_name, temperature=0)
        self.chat_llm     = ChatOpenAI(model=model_name, temperature=0.1)

    def analyze_node(self, state: InsuranceState) -> dict:
        messages = state['messages']
        question = messages[-1].content  # 🚨 변수명 수정 및 통일

        history = []
        for m in messages[:-1]:
            role = "User" if isinstance(m, HumanMessage) else "AI"
            history.append(f"{role}: {m.content}")
        context_str = "\n".join(history[-5:])

        # insurance_rag.py 내 analyze_node 프롬프트 전면 수정

        prompt = f"""You are an insurance query analyzer for {self.plugin.name}.

        [ABSOLUTE CRITICAL RULES - DO NOT IGNORE]
        1. STRICT BAN ON RECOMMENDATIONS (보험 추천 절대 금지): 
           - If the user asks for a plan recommendation or subjective comparison (e.g., "어떤 플랜이 좋나요?", "추천해주세요", "A랑 B중 뭐가 나아요?"), 
           - You MUST set "needs_clarification" to true.
           - Write a polite refusal in "clarification_message" (e.g., "저는 특정 보험 상품의 가입을 추천하거나 주관적인 비교를 해드릴 수 없습니다. 원하시는 플랜의 객관적인 보장 내용이나 약관 정보만 확인해 드릴 수 있습니다. 어떤 플랜에 대해 알아보고 싶으신가요?").
           - Set "english_query" to "". (Do not search).
        
        2. PLAN TIER IS MANDATORY (정보 일관성을 위한 플랜 확인 필수): 
           - To ensure information consistency, you MUST know the user's exact 'plan_tier' (e.g., Premier, Select, IHHP, MajorMedical) before searching.
           - IF PLAN IS MISSING (and it's not a recommendation request): 
             * Set "needs_clarification" to true.
             * Set "english_query" to "". (Do not search yet).
             * In "clarification_message", politely ask for the plan tier. 
             * CRUCIAL EMPATHY: If the user mentioned a symptom (e.g., "다리가 부러짐", "골절"), acknowledge it first! (e.g., "다리가 부러지셨다니 많이 놀라셨겠어요. 정확한 보장 한도와 절차를 안내해 드리기 위해, 가입하신 플랜명(Premier, IHHP 등)을 먼저 알려주시겠어요?")
           - IF PLAN IS KNOWN (from current message or past context):
             * Set "needs_clarification" to false.
             * Generate the "english_query" combining the symptom/question and the plan.

        [CONVERSATION CONTEXT]
        {context_str}

        [CURRENT USER MESSAGE]
        {question}

        [STRICT JSON FORMAT]
        {{
          "language": "ko|en|ja|zh",
          "plan_or_intent": "extracted plan or null",
          "region": "extracted region or null",
          "english_query": "search query or empty string if clarification needed",
          "needs_clarification": true | false,
          "clarification_message": "polite refusal/question or empty string"
        }}
        """

        raw      = self.analyzer_llm.invoke(prompt).content
        match    = re.search(r'\{.*\}', raw, re.DOTALL)
        analysis = json.loads(match.group(0)) if match else {}
        
        new_val  = analysis.get('plan_or_intent')
        prev_val = state.get('plan_or_intent')
        final    = new_val if new_val and new_val != "None" else prev_val
        
        return {
            "normalized_query": analysis,
            "plan_or_intent":   final,
            "current_question": question,
            "clarification_msg": analysis.get('clarification_message', ''),
        }

    def retrieve_node(self, state: InsuranceState) -> dict:
        query = state['normalized_query'].get('english_query', state['current_question'])[cite: 1]
        plan = state.get('plan_or_intent') # 파싱된 플랜명

        # 해당 플랜의 문서만 가져오도록 필터 락(Lock)을 겁니다.
        search_filter = {"plan_tier": plan} if plan and plan != "None" else None[cite: 1]

        docs = self.plugin.retrieve(
            query=query,
            normalized=state['normalized_query'],
            plan_or_intent=plan,
            filter=search_filter  # 플러그인 내부 구현에 따라 필터 전달
        )

        formatted = []
        for d in docs:
            source_path = d.metadata.get('source', 'document_unknown.pdf')
            file_name = Path(source_path).name
            page_num = d.metadata.get('page', '?')
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

        [CRITICAL: ANTI-MIXING & CITATION RULE]
        1. **플랜/제품 혼용 금지**: 검색된 문서들이 서로 다른 보험 상품에 대한 것이라면, 이를 하나의 절차로 합쳐서 답변하지 마세요. 상품별 차이점을 구분하세요.
        2. **출처 고정**: 모든 문장 끝에 반드시 해당 정보의 근거가 된 (출처: 파일명, p.번호)를 남기세요.
        3. **정보 부재 시**: 문서에 없는 내용을 추측하지 말고 "해당 내용은 제공된 문서에서 확인되지 않습니다"라고 명시하세요.
        """

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