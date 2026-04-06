import streamlit as st
from langchain_core.documents import Document

from rag_utils import run_chat_turn

st.set_page_config(page_title="Allianz Insurance Assistant", page_icon="💬", layout="wide")

st.title("Allianz Insurance Assistant")
st.caption("문서 기반 보험 안내 챗봇")


with st.sidebar:
    st.header("예시 질문")
    st.markdown(
        """
- 싱가포르에서 입원 전에 사전승인이 필요한가요?
- 사전승인 폼에 어떤 정보를 입력해야 하나요?
- 영국에서 청구하려면 어떤 서류가 필요한가요?
- 홍콩에서 출산 관련 보장은 어떻게 되나요?
- 두바이에서 외래 진료는 보장되나요?
"""
    )
    st.markdown("---")

# 1. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = {
        "thread_id": "chat-session-001",
        "slots": {},
        "pending_followup": False,
        "last_followup_question": "",
        "followup_count": 0,
        "max_followups": 2,
    }

if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []


# 2. 문서 표시용 헬퍼
def format_doc_title(doc: Document, idx: int) -> str:
    source = doc.metadata.get("source", f"document_{idx}")
    page = doc.metadata.get("page", "?")
    region = doc.metadata.get("region", "unknown")
    doc_type = doc.metadata.get("doc_type", "unknown")
    return f"{idx}. {source} | page={page} | region={region} | type={doc_type}"


def render_references(docs: list[Document]):
    if not docs:
        return

    with st.expander("참고 문서 보기", expanded=False):
        for i, doc in enumerate(docs, start=1):
            st.markdown(f"**{format_doc_title(doc, i)}**")
            preview = doc.page_content[:700].strip()
            if len(doc.page_content) > 700:
                preview += " ..."
            st.write(preview)
            st.divider()

# 3. 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("docs"):
            render_references(msg["docs"])


# 4. 사용자 입력 처리
user_input = st.chat_input("질문을 입력하세요")

if user_input:
    # 사용자 메시지 저장/출력
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # rag_utils.run_chat_turn 호출
    result = run_chat_turn(
        question=user_input,
        conversation_state=st.session_state.conversation_state,
    )

    answer = result.get("answer", "")
    retrieved_docs = result.get("retrieved_docs", [])
    conversation_state = result.get("conversation_state", st.session_state.conversation_state)
    needs_followup = result.get("needs_followup", False)
    suggested_next_questions = result.get("suggested_next_questions", [])

    # 세션 상태 갱신
    st.session_state.conversation_state = conversation_state
    st.session_state.last_retrieved_docs = retrieved_docs

    # assistant 응답 조립
    assistant_text = answer

    if suggested_next_questions and not needs_followup:
        assistant_text += "\n\n---\n**이어서 물어볼 만한 질문**\n"
        for q in suggested_next_questions:
            assistant_text += f"- {q}\n"

    # assistant 메시지 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_text,
        "docs": retrieved_docs,
    })

    # assistant 출력
    with st.chat_message("assistant"):
        st.markdown(assistant_text)

        if needs_followup:
            st.info("추가 정보를 알려주시면 더 정확하게 안내할 수 있습니다.")

        render_references(retrieved_docs)


# 5. 사이드바 상태 확인
with st.sidebar:
    st.subheader("대화 상태")

    conv = st.session_state.conversation_state

    st.write("**thread_id**")
    st.code(conv.get("thread_id", ""))

    st.write("**pending_followup**")
    st.write(conv.get("pending_followup", False))

    st.write("**followup_count / max_followups**")
    st.write(f"{conv.get('followup_count', 0)} / {conv.get('max_followups', 2)}")

    st.write("**last_followup_question**")
    st.write(conv.get("last_followup_question", ""))

    st.write("**누적 slots**")
    st.json(conv.get("slots", {}))

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.conversation_state = {
            "thread_id": "chat-session-001",
            "slots": {},
            "pending_followup": False,
            "last_followup_question": "",
            "followup_count": 0,
            "max_followups": 2,
        }
        st.session_state.last_retrieved_docs = []
        st.rerun()