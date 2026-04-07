"""
Insurance_Benefit_Chatbot/main.py
실행: streamlit run combined_main.py
"""
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

from bot_manager import BotManager, INSURER_OPTIONS

# ── 페이지 설정 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="보험 AI 챗봇",
    page_icon="🏥",
    layout="wide"
)

# ── 봇 로드 (최초 1회 캐시) ─────────────────────────────────────
@st.cache_resource
def load_manager() -> BotManager:
    return BotManager()

manager = load_manager()

# ── 사이드바: 보험사 선택 ───────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")
    st.markdown("---")

    selected_label = st.selectbox(
        "보험사 선택",
        options=list(INSURER_OPTIONS.keys()),
        index=0,
    )
    selected_key = INSURER_OPTIONS[selected_label]

    st.markdown("---")
    st.markdown(f"**현재 선택:** {selected_label}")

    if st.button("🗑️ 대화 초기화"):
        st.session_state.pop(f"messages_{selected_key}", None)
        st.session_state.pop(f"thread_{selected_key}", None)
        st.rerun()

    st.markdown("---")
    st.caption("© 2025 Insurance AI Chatbot")

# ── 메인 영역 ───────────────────────────────────────────────────
st.title(f"💬 {selected_label}")
st.caption("문서 기반 보험 상담 챗봇입니다.")

# 보험사별 독립 세션
msg_key    = f"messages_{selected_key}"
thread_key = f"thread_{selected_key}"

if msg_key    not in st.session_state:
    st.session_state[msg_key]    = []
if thread_key not in st.session_state:
    st.session_state[thread_key] = str(uuid.uuid4())

# 대화 기록 출력
for msg in st.session_state[msg_key]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# ── 입력창 ──────────────────────────────────────────────────────
if prompt := st.chat_input("질문을 입력하세요..."):

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            bot    = manager.get(selected_key)
            config = {"configurable": {"thread_id": st.session_state[thread_key]}}

            snapshot  = bot.get_state(config)
            prev_plan = snapshot.values.get("plan_or_intent") if snapshot.values else None

            result = bot.invoke(
                {
                    "messages":       [HumanMessage(content=prompt)],
                    "plan_or_intent": prev_plan,
                },
                config=config
            )
            answer = result["messages"][-1].content

        st.write(answer)

    st.session_state[msg_key].append(HumanMessage(content=prompt))
    st.session_state[msg_key].append(AIMessage(content=answer))
