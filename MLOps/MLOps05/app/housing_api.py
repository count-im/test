import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from app.housing_schemas import HousingRequest, HousingResponse
from app.housing_model import HousingPredictor
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware

logger = setup_logger("housing_api")
executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="California Housing Price API",
              description="캘리포니아 주택 가격 예측 API", version="1.0.0")
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

predictor: HousingPredictor | None = None

@app.on_event("startup")
def startup():
    global predictor
    try:
        predictor = HousingPredictor()
        logger.info("✅ 모델 로드 완료")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict", response_model=HousingResponse)
async def predict(req: HousingRequest):
    if predictor is None:
        raise HTTPException(503, "모델 준비 중입니다")
    features = [req.MedInc, req.HouseAge, req.AveRooms, req.AveBedrms,
                req.Population, req.AveOccup, req.Latitude, req.Longitude]
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, predictor.predict, features)
    return result
