# 서버 실행: uvicorn app.main:app --reload --port 8000
# UI 실행:   streamlit run frontend/app.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends
from app.schemas import SentimentRequest, SentimentResponse, SentimentResult
from app.model_service import load_model, predict
from app.auth import verify_api_key

executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="한국어 감정 분석 API",
              description="KR-FinBert-SC 기반 한국어 감정 분석 서비스", version="1.0.0")


@app.on_event("startup")
def startup():
    try:
        app.state.model = load_model()
        print("✅ 모델 로드 완료: KR-FinBert-SC")
    except Exception as e:
        app.state.model = None
        print(f"❌ 모델 로드 실패: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model": "KR-FinBert-SC"}


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(req: SentimentRequest,
                            api_key: str = Depends(verify_api_key)):
    if app.state.model is None:
        raise HTTPException(503, "모델 준비 중입니다")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, predict, app.state.model, req.text
        )
        return SentimentResponse(
            success=True,
            result=SentimentResult(**result),
            message="분석 완료"
        )
    except Exception as e:
        raise HTTPException(500, f"추론 실패: {str(e)}")
