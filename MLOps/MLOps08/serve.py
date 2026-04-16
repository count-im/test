# serve.py
# FSDL Day3 - FastAPI 서빙 스크립트
#
# 실행 방법:
#   cd /workspace/mycoding/test/MLOps08
#   uvicorn serve:app --reload --port 8000
#
# API 테스트:
#   curl -X POST http://localhost:8000/classify \
#        -H "Content-Type: application/json" \
#        -d '{"title": "삼성전자 반도체 공장 착공"}'
#
# 예상 응답:
#   {"category":"IT과학","confidence":0.95,"all_scores":{...}}

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.model import NewsClassifier

# ── FastAPI 앱 초기화 ──
app = FastAPI(
    title="한국어 뉴스 분류 API",
    description="FSDL Day3 종합 프로젝트 — KLUE/BERT 기반 7카테고리 분류",
    version="1.0.0",
)

# ── 전역 모델/토크나이저 (서버 시작 시 1회 로드) ──
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best.ckpt")
LABELS = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']

model = None
tokenizer = None
device = None


@app.on_event("startup")
def load_model():
    """
    서버 시작 시 모델을 한 번 로드.
    on_event("startup"): FastAPI 라이프사이클 훅.
    요청마다 로드하면 너무 느림 → 전역으로 한 번만.
    """
    global model, tokenizer, device

    # 디바이스 명시적 설정 (GPU 있으면 cuda, 없으면 cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️  체크포인트 없음: {CHECKPOINT_PATH}")
        print("   먼저 train.py를 실행하여 모델을 학습하세요.")
        print("   체크포인트 없이도 서버는 실행되지만, /classify 요청은 실패합니다.")
        return

    print(f"모델 로드 중: {CHECKPOINT_PATH}")
    model = NewsClassifier.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location=device,  # 체크포인트를 현재 디바이스로 로드
    )
    model = model.to(device)  # 명시적으로 디바이스 이동
    model.eval()
    model.freeze()  # 서빙 시에는 파라미터 고정 (메모리·속도 최적화)

    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    print("✅ 모델 로드 완료")


# ── 요청/응답 스키마 ──
class NewsRequest(BaseModel):
    title: str  # 분류할 뉴스 제목


class NewsResponse(BaseModel):
    category: str           # 예측 카테고리
    confidence: float       # 최고 확률값
    all_scores: dict        # 7개 카테고리별 확률


# ── API 엔드포인트 ──
@app.post("/classify", response_model=NewsResponse)
def classify_news(request: NewsRequest):
    """
    뉴스 제목을 7개 카테고리로 분류합니다.

    Request body:
        title (str): 분류할 뉴스 제목

    Returns:
        category: 예측된 카테고리 이름
        confidence: 해당 카테고리의 확률 (0~1)
        all_scores: 7개 카테고리별 확률 딕셔너리
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 로드되지 않았습니다. 먼저 train.py를 실행하세요."
        )

    if not request.title.strip():
        raise HTTPException(status_code=400, detail="제목이 비어있습니다.")

    # 토크나이징
    inputs = tokenizer(
        request.title,
        return_tensors="pt",
        max_length=128,
        padding=True,
        truncation=True,
    )

    # 입력을 모델 디바이스로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 추론 (no_grad: 그래디언트 계산 불필요 → 메모리·속도 절약)
    with torch.no_grad():
        logits = model.model(**inputs).logits  # (1, 7)

    # 소프트맥스로 확률 변환
    probs = F.softmax(logits[0], dim=-1)  # (7,)

    pred_id = probs.argmax().item()
    all_scores = {LABELS[i]: round(probs[i].item(), 4) for i in range(7)}

    return NewsResponse(
        category=LABELS[pred_id],
        confidence=round(probs[pred_id].item(), 4),
        all_scores=all_scores,
    )


@app.get("/health")
def health_check():
    """서버 상태 확인 엔드포인트"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "checkpoint_exists": os.path.exists(CHECKPOINT_PATH),
    }


@app.get("/")
def root():
    return {
        "message": "한국어 뉴스 분류 API",
        "docs": "/docs",
        "health": "/health",
        "classify": "POST /classify",
    }
