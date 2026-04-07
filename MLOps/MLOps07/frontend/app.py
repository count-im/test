import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="한국어 감정 분석", page_icon="🧠", layout="wide")
st.title("🧠 한국어 감정 분석 서비스")

# --- 사이드바 ---
with st.sidebar:
    st.header("설정")
    api_key = st.text_input("API Key", type="password", value="test-api-key-1234")

    st.divider()
    st.subheader("서버 상태")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            st.success(f"모델: {resp.json().get('model', 'unknown')}")
        else:
            st.warning("서버 응답 이상")
    except Exception:
        st.error("서버 연결 실패")

# --- 메인 ---
text = st.text_area(
    "분석할 텍스트를 입력하세요",
    placeholder="예: 이 제품 정말 좋아요! 배송도 빠르고 품질도 훌륭합니다.",
    height=150,
)

if st.button("분석하기", type="primary"):
    if not text.strip():
        st.warning("텍스트를 입력해주세요.")
    elif not api_key:
        st.warning("🔑 사이드바에서 API Key를 입력하세요.")
    else:
        with st.spinner("분석 중..."):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text},
                    headers={"X-API-Key": api_key},
                    timeout=30,
                )
                if resp.status_code == 401:
                    st.error("🔑 API Key가 올바르지 않습니다")
                elif resp.status_code == 200:
                    data = resp.json()
                    result = data["result"]
                    st.subheader("분석 결과")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("감정", result["label"])
                    with col2:
                        st.metric("신뢰도", f"{result['score']:.2%}")
                    st.progress(result["score"])
                else:
                    st.error(f"서버 오류: {resp.status_code} - {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            except Exception as e:
                st.error(f"오류 발생: {e}")
