# 서버 실행: uvicorn app.chatbot_api:app --reload --port 8000
# UI 실행:   streamlit run frontend/app_chatbot.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends
from app.chatbot_schemas import ChatRequest, ChatResponse
from app.chatbot_model import ChatbotModel
from app.auth import verify_api_key
from app.logger_config import setup_logger
from app.error_handlers import register_error_handlers
from app.middleware import RequestLoggingMiddleware

logger = setup_logger("chatbot_api")
executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="Korean GPT Chatbot API",
              description="한국어 GPT 멀티턴 챗봇 API", version="1.0.0")
app.add_middleware(RequestLoggingMiddleware)
register_error_handlers(app)

chatbot: ChatbotModel | None = None

@app.on_event("startup")
def startup():
    global chatbot
    try:
        chatbot = ChatbotModel()
        logger.info("✅ 챗봇 모델 로드 완료")
    except Exception as e:
        logger.error(f"챗봇 모델 로드 실패: {e}")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": chatbot is not None}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    if chatbot is None:
        raise HTTPException(503, "모델 준비 중입니다")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        executor, chatbot.generate, messages, req.temperature, req.max_new_tokens
    )
    return ChatResponse(reply=reply, turn_count=len(req.messages))
