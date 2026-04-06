"""
app.py
──────
Streamlit 팀 통합 앱 (TRICARE 탭 기준 작성)
Cigna / Allianz 탭은 팀원 모듈 import 후 동일 방식으로 추가 가능

실행 방법:
    streamlit run tricare_app.py
"""

import re
import streamlit as st
from tricare_core import load_vector_stores, format_docs, SOURCE_URL_MAP
from tricare_graph import TricareChat

#  페이지 설정 
st.set_page_config(
    page_title='Care-Insure-Bot',
    page_icon='🏥',
    layout='wide'
)


def clean_filename(filename: str) -> str:
    """
    파일명을 사람이 읽기 좋은 레이블로 변환.
    - 한국어 괄호 및 괄호 내용 제거: '파일명(한국어설명).pdf' → '파일명'
    - 확장자 제거 (.pdf, .csv)
    - 언더스코어를 공백으로 변환
    예) 'Overseas_HB해외_프로그램_안내서.pdf' → 'Overseas HB해외 프로그램 안내서'
    예) 'TRICARE_ADDP_HB_FINAL_508c(현역_군인_치과_프로그램_안내서).pdf' → 'TRICARE ADDP HB FINAL 508c'
    """
    name = filename
    name = re.sub(r'\(.*?\)', '', name)   # 괄호 및 괄호 안 내용 제거
    name = re.sub(r'\.(pdf|csv)$', '', name, flags=re.IGNORECASE)  # 확장자 제거
    name = name.replace('_', ' ').strip()
    return name


def render_doc_sources(docs: list):
    """참고 문서 출처를 URL 링크와 함께 표시"""
    with st.expander('📄 참고 문서 보기'):
        for doc in docs:
            src  = doc.metadata.get('source_file', doc.metadata.get('source', '?'))
            page = doc.metadata.get('page', '')
            url  = SOURCE_URL_MAP.get(src, '')
            label = clean_filename(src)

            if page:
                label += f', p.{page}'

            if url:
                # 클릭 가능한 링크로 표시
                st.caption(f'[{label}]({url})')
            else:
                st.caption(f'[{label}]')


#  앱 시작 시 한 번만 로드 
# @st.cache_resource: 서버 재시작 전까지 모델/DB를 메모리에 유지
# 여러 사용자가 접속해도 로드는 한 번만 실행됨
@st.cache_resource
def init_tricare():
    """벡터 DB + 모델 초기화 (최초 1회)"""
    load_vector_stores()
    return True

init_tricare()


#  세션별 챗 인스턴스 
# st.session_state: 사용자(탭) 별로 독립적인 State 유지
if 'tricare_chat' not in st.session_state:
    st.session_state.tricare_chat = TricareChat()


#  UI 
st.title('🏥 Care-Insure-Bot')
st.caption('TRICARE · Cigna · Allianz 보험 정보 Q&A')
st.divider()

# 탭 구성 — 팀원 모듈 준비되면 Cigna / Allianz 탭 활성화
tab_tricare, tab_cigna, tab_allianz = st.tabs(['TRICARE', 'Cigna', 'Allianz'])


#  TRICARE 탭 
with tab_tricare:

    col_chat, col_info = st.columns([3, 1])

    with col_info:
        st.subheader('현재 컨텍스트')
        ctx = st.session_state.tricare_chat.context
        st.info(
            f"**플랜**: {ctx['plan_tier'] or '미확인'}\n\n"
            f"**지역**: {ctx['region'] or '미확인'}\n\n"
            f"**대화 수**: {ctx['turns']}턴"
        )

        if st.button('🔄 새 대화 시작', key='tricare_reset'):
            st.session_state.tricare_chat.reset()
            st.session_state.tricare_messages = []
            st.rerun()

        st.caption(
            '⚠️ 본 서비스는 공개 문서 기반 정보 제공이며, '
            '보험 가입 권유·중개·추천을 하지 않습니다.'
        )

    with col_chat:
        st.subheader('TRICARE Q&A')

        # 대화 히스토리 초기화
        if 'tricare_messages' not in st.session_state:
            st.session_state.tricare_messages = []

        # 이전 대화 출력
        for msg in st.session_state.tricare_messages:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
                if msg.get('docs'):
                    render_doc_sources(msg['docs'])

        # 사용자 입력
        user_input = st.chat_input(
            'TRICARE 관련 질문을 입력하세요 (예: Prime에서 한국 정신건강 상담 보장되나요?)'
        )

        if user_input:
            # 사용자 메시지 표시
            with st.chat_message('user'):
                st.write(user_input)
            st.session_state.tricare_messages.append({
                'role': 'user', 'content': user_input
            })

            # 챗봇 응답
            with st.chat_message('assistant'):
                with st.spinner('문서 검색 중...'):
                    result = st.session_state.tricare_chat.send(user_input)

                st.write(result['answer'])

                if result['retrieved_docs']:
                    render_doc_sources(result['retrieved_docs'])

            st.session_state.tricare_messages.append({
                'role':    'assistant',
                'content': result['answer'],
                'docs':    result['retrieved_docs'],
            })

            # 컨텍스트 패널 갱신
            st.rerun()


#  Cigna 탭 (팀원 모듈 연결 시 활성화) 
with tab_cigna:
    st.info('Cigna 탭 — 팀원 모듈 연결 후 활성화 예정')
    # from cigna_core import load_cigna_stores, CignaChat
    # 동일한 패턴으로 구현


#  Allianz 탭 (팀원 모듈 연결 시 활성화) 
with tab_allianz:
    st.info('Allianz 탭 — 팀원 모듈 연결 후 활성화 예정')
    # from allianz_core import load_allianz_stores, AllianzChat
    # 동일한 패턴으로 구현
