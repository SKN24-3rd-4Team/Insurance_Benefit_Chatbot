"""
tricare_graph.py

TRICARE LangGraph 멀티턴 챗봇 모듈

포함 내용:
- TriCareState (TypedDict)
- clarify_node / retrieve_node / generate_node
- tricare_graph (컴파일된 LangGraph 그래프)
- TricareChat 클래스 (Streamlit에서 세션별로 인스턴스 생성)

사용 방법:
    from tricare_core import load_vector_stores
    from tricare_graph import TricareChat

    load_vector_stores()          # 앱 시작 시 한 번만

    chat = TricareChat()          # 사용자 세션마다 생성
    response = chat.send("질문")
    chat.reset()                  # 새 대화 시작
"""

import json
from typing import TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# tricare_core에서 필요한 함수/객체를 가져옴
import tricare_core
from tricare_core import (
    detect_language,
    LANGUAGE_NAME_MAP,
    format_docs,
    hybrid_retrieve_wide,
    rerank_docs,
)


# State 정의 

class TriCareState(TypedDict):
    messages:             Annotated[list, add_messages]  # 대화 히스토리 누적
    plan_tier:            Optional[str]   # 'TRICARE Prime' 등 기억
    region:               Optional[str]   # 'korea', 'OCONUS' 등 기억
    turns:                int
    retrieved_docs:       list
    needs_clarification:  bool


#  컨텍스트 추출 유틸 

PLAN_KEYWORDS = {
    'prime':'TRICARE Prime',
    'select':  'TRICARE Select',
    'for life': 'TRICARE For Life',
    'tfl':     'TRICARE For Life',
    'reserve': 'TRICARE Reserve Select',
    'trs':  'TRICARE Reserve Select',
    'young adult': 'TRICARE Young Adult',
    'tya':    'TRICARE Young Adult',
    'addp': 'TRICARE ADDP',
}

REGION_KEYWORDS = {
    'korea':    'korea', '한국': 'korea', '주한': 'korea', 'usfk': 'korea',
    'overseas': 'OCONUS', '해외': 'OCONUS', 'oconus': 'OCONUS',
    'conus':    'CONUS',  '미국': 'CONUS',
}


def _extract_context(text: str) -> dict:
    """사용자 메시지에서 플랜명 / 지역 정보 추출"""
    text_lower = text.lower()
    plan_tier, region = None, None

    for kw, val in PLAN_KEYWORDS.items():
        if kw in text_lower:
            plan_tier = val
            break
    for kw, val in REGION_KEYWORDS.items():
        if kw in text_lower:
            region = val
            break

    return {'plan_tier': plan_tier, 'region': region}


def _build_conv_history(messages: list, max_turns: int = 6) -> str:
    """최근 N턴 대화를 문자열로 변환"""
    recent = messages[-(max_turns * 2):]
    lines  = []
    for msg in recent:
        role = '사용자' if isinstance(msg, HumanMessage) else '챗봇'
        lines.append(f'{role}: {msg.content}')
    return '\n'.join(lines) if lines else '(이전 대화 없음)'


def _get_last_user_msg(messages: list) -> str:
    """메시지 리스트에서 마지막 사용자 메시지 추출"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ''


#  노드 함수 

def clarify_node(state: TriCareState) -> dict:
    """
    [clarify 노드]
    질문이 충분한지 LLM으로 판단.
    부족하면 한국어 후속 질문 생성 → needs_clarification=True 반환.
    충분하면 needs_clarification=False → retrieve 노드로 진행.
    """
    last_user_msg = _get_last_user_msg(state['messages'])
    extracted     = _extract_context(last_user_msg)

    # 이전 State 값 유지 + 새로 추출된 값으로 업데이트
    plan_tier = extracted['plan_tier'] or state.get('plan_tier')
    region    = extracted['region']    or state.get('region')
    conv_hist = _build_conv_history(state['messages'])

    clarify_prompt = (
        "You are a TRICARE insurance chatbot assistant.\n"
        "Decide if the user's question has enough info to search TRICARE documents.\n\n"
        f"Known context — Plan: {plan_tier or 'unknown'}, Region: {region or 'unknown'}\n"
        f"Conversation so far:\n{conv_hist}\n\n"
        f'Latest question: "{last_user_msg}"\n\n'
        "Rules:\n"
        "- Too vague (e.g. '보험 되나요?') AND plan/region both unknown → needs clarification\n"
        "- plan OR region known from context → likely sufficient\n"
        "- References unclear previous topic → ask clarification\n"
        "- Clear enough to search → sufficient\n\n"
        'Respond ONLY in JSON:\n'
        '{"needs_clarification": true/false, '
        '"follow_up_question": "한국어 질문 (needs_clarification=true일 때만)"}'
    )

    try:
        resp = tricare_core._norm_model.invoke([HumanMessage(content=clarify_prompt)])
        result = json.loads(resp.content)
        needs  = result.get('needs_clarification', False)
        fup    = result.get('follow_up_question', '')
    except Exception:
        needs, fup = False, ''

    base = {
        'plan_tier':  plan_tier,
        'region':     region,
        'turns':      state.get('turns', 0) + 1,
        'retrieved_docs': [],
    }

    if needs and fup:
        return {**base,
                'messages':            [AIMessage(content=fup)],
                'needs_clarification': True}
    else:
        return {**base, 'needs_clarification': False}


def retrieve_node(state: TriCareState) -> dict:
    """
    [retrieve 노드]
    State의 plan_tier / region을 질문에 보강 후 Hybrid+Rerank 검색.
    """
    last_user_msg = _get_last_user_msg(state['messages'])

    # plan/region을 질문 앞에 붙여 검색 정확도 향상
    enriched = last_user_msg
    if state.get('plan_tier'):
        enriched = f"{state['plan_tier']} {enriched}"
    if state.get('region'):
        enriched = f"{state['region']} {enriched}"

    candidates = hybrid_retrieve_wide(enriched)
    docs       = rerank_docs(enriched, candidates, top_k=6)

    return {'retrieved_docs': docs}

# 팀원과 합의하에 통일
def generate_node(state: TriCareState) -> dict:
    """
    [generate 노드]
    검색 문서 + 대화 히스토리 기반 최종 답변 생성.
    """
    last_user_msg   = _get_last_user_msg(state['messages'])
    language_code   = detect_language(last_user_msg)
    answer_language = LANGUAGE_NAME_MAP.get(language_code, 'English')
    context         = format_docs(state.get('retrieved_docs', []))
    conv_history    = _build_conv_history(state['messages'])

    prompt_text = (
        'You are a TRICARE health benefits specialist.\n'
        'This system is designed for OCONUS beneficiaries, '
        'primarily Korean residents and USFK (주한미군) personnel.\n\n'
        'IMPORTANT OCONUS RULES:\n'
        '- In South Korea, TRICARE is the PRIMARY payer (not Medicare).\n'
        '- Medicare does NOT cover overseas medical expenses.\n'
        '- Overseas claims require pay-up-front then submit within 3 years.\n'
        '- Medicare Part B must be actively enrolled for overseas residents.\n\n'
        f'Known context: Plan={state.get("plan_tier") or "unknown"}, '
        f'Region={state.get("region") or "unknown"}\n\n'
        f'[이전 대화]\n{conv_history}\n\n'
        'Answer ONLY based on the provided TRICARE documents below.\n'
        'If not in documents, say "해당 내용은 제공된 문서에서 확인되지 않습니다."\n'
        'Always cite source file. Mention Group A/B, plan type when relevant.\n'
        '보험 추천이나 특정 플랜을 권유하는 답변은 하지 마세요.\n\n'
        f'IMPORTANT: Answer in {answer_language}.\n'
        'Term locking: 본인부담금(Copay), 공제액(Deductible), 사전승인(Prior Authorization).\n\n'
        f'[참고 문서]\n{context}\n\n'
        f'[질문]\n{last_user_msg}\n\n'
        '[답변]\n'
    )

    resp = tricare_core.model.invoke([HumanMessage(content=prompt_text)])
    return {'messages': [AIMessage(content=resp.content)]}

#  그래프 컴파일 

def _should_clarify(state: TriCareState) -> str:
    return END if state.get('needs_clarification', False) else 'retrieve'


def _build_graph() -> StateGraph:
    builder = StateGraph(TriCareState)
    builder.add_node('clarify',  clarify_node)
    builder.add_node('retrieve', retrieve_node)
    builder.add_node('generate', generate_node)

    builder.set_entry_point('clarify')
    builder.add_conditional_edges(
        'clarify',
        _should_clarify,
        {END: END, 'retrieve': 'retrieve'}
    )
    builder.add_edge('retrieve', 'generate')
    builder.add_edge('generate', END)
    return builder.compile()


# 모듈 임포트 시 자동으로 그래프 컴파일
tricare_graph = _build_graph()


#  TricareChat 클래스 

class TricareChat:
    """
    Streamlit 세션별로 인스턴스를 생성해서 사용.
    각 인스턴스가 독립적인 State를 유지함.

    사용 예시 (Streamlit):
        if 'tricare_chat' not in st.session_state:
            st.session_state.tricare_chat = TricareChat()

        result = st.session_state.tricare_chat.send(user_input)
        st.write(result['answer'])
    """

    def __init__(self):
        self._state: TriCareState = self._init_state()

    @staticmethod
    def _init_state() -> TriCareState:
        return {
            'messages':            [],
            'plan_tier':           None,
            'region':              None,
            'turns':               0,
            'retrieved_docs':      [],
            'needs_clarification': False,
        }

    def send(self, user_input: str) -> dict:
        """
        사용자 입력을 받아 챗봇 응답을 반환.

        Returns
        -------
        dict:
            answer      : str          챗봇 답변
            plan_tier   : str | None   현재 State의 플랜명
            region      : str | None   현재 State의 지역
            turns       : int          대화 턴 수
            retrieved_docs : list      검색된 문서 목록
            needs_clarification : bool 후속 질문 여부
        """
        input_state = dict(self._state)
        input_state['messages'] = (
            self._state['messages'] + [HumanMessage(content=user_input)]
        )

        result       = tricare_graph.invoke(input_state)
        self._state  = result

        # 마지막 AI 메시지 추출
        answer = ''
        for msg in reversed(result['messages']):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break

        return {
            'answer':               answer,
            'plan_tier':            result.get('plan_tier'),
            'region':               result.get('region'),
            'turns':                result.get('turns', 0),
            'retrieved_docs':       result.get('retrieved_docs', []),
            'needs_clarification':  result.get('needs_clarification', False),
        }

    def reset(self) -> None:
        """새 대화 시작 — State 전체 초기화"""
        self._state = self._init_state()

    @property
    def history(self) -> list:
        """현재 대화 히스토리 반환 (HumanMessage / AIMessage 리스트)"""
        return self._state['messages']

    @property
    def context(self) -> dict:
        """현재 State의 컨텍스트 정보 반환"""
        return {
            'plan_tier': self._state.get('plan_tier'),
            'region':    self._state.get('region'),
            'turns':     self._state.get('turns', 0),
        }
