import streamlit as st
import requests

st.set_page_config(page_title="한국어 챗봇", page_icon="🤖", layout="wide")

API_URL = "http://localhost:8000"

# --- 사이드바 ---
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("API Key", type="password", value="")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 10, 300, 100, 10)

    if st.button("대화 초기화"):
        st.session_state["chat_messages"] = []
        st.rerun()

    st.divider()
    st.subheader("서버 상태")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        data = resp.json()
        if data.get("model_loaded"):
            st.success("모델 로드됨")
        else:
            st.warning("모델 로드 중...")
    except Exception:
        st.error("서버 연결 실패")

# --- 대화 기록 초기화 ---
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

st.title("🤖 한국어 GPT 챗봇")

# --- 기존 대화 렌더링 ---
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


def call_chat_api(messages, api_key, temperature, max_tokens):
    headers = {"X-API-Key": api_key}
    body = {
        "messages": messages,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    }
    return requests.post(f"{API_URL}/chat", json=body, headers=headers, timeout=60)


# --- 사용자 입력 ---
if prompt := st.chat_input("메시지를 입력하세요..."):
    # user 메시지 추가
    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # API 호출
    if not api_key:
        with st.chat_message("assistant"):
            st.error("🔑 사이드바에서 API Key를 입력하세요")
    else:
        try:
            resp = call_chat_api(
                st.session_state["chat_messages"], api_key, temperature, max_tokens
            )
            if resp.status_code == 401:
                with st.chat_message("assistant"):
                    st.error("🔑 인증 실패: API Key를 확인하세요")
            elif resp.status_code == 200:
                reply = resp.json()["reply"]
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": reply}
                )
                with st.chat_message("assistant"):
                    st.write(reply)
            else:
                with st.chat_message("assistant"):
                    st.error(f"서버 오류: {resp.status_code} - {resp.text}")
        except requests.exceptions.ConnectionError:
            with st.chat_message("assistant"):
                st.error("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"오류 발생: {e}")
