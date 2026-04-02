import streamlit as st
import requests

st.set_page_config(page_title="주택 가격 예측", page_icon="🏠", layout="wide")
API = "http://localhost:8000"

def call_predict(data):
    try:
        r = requests.post(f"{API}/predict", json=data, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

st.title("🏠 캘리포니아 주택 가격 예측")
health = requests.get(f"{API}/health", timeout=5).json()
st.info(f"서버 상태: {health['status']} | 모델: {'로드됨' if health['model_loaded'] else '준비중'}")

col1, col2 = st.columns(2)
with col1:
    MedInc     = st.slider("중위 소득 (만달러)", 0.5, 15.0, 3.5)
    HouseAge   = st.slider("주택 연식 (년)", 1, 52, 25)
    AveRooms   = st.slider("평균 방 수", 1.0, 10.0, 5.0)
    AveBedrms  = st.slider("평균 침실 수", 0.5, 5.0, 1.0)
with col2:
    Population = st.number_input("인구", 100, 10000, 1500)
    AveOccup   = st.slider("평균 거주자 수", 1.0, 10.0, 3.0)
    Latitude   = st.slider("위도", 32.5, 42.0, 37.5)
    Longitude  = st.slider("경도", -124.5, -114.0, -122.0)

if st.button("🔍 예측하기", type="primary"):
    data = dict(MedInc=MedInc, HouseAge=HouseAge, AveRooms=AveRooms,
                AveBedrms=AveBedrms, Population=Population,
                AveOccup=AveOccup, Latitude=Latitude, Longitude=Longitude)
    result, err = call_predict(data)
    if result:
        st.success(f"## 예측 가격: ${result['predicted_price']:,.0f}")
        st.caption(result['confidence_note'])
    else:
        st.error(f"오류: {err}")
