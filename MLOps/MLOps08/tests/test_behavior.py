# tests/test_behavior.py
# FSDL Day3 - 행동 테스트 (Behavioral Testing, Day3 Ch.13.6)
#
# 핵심 원리:
#   정답(Ground Truth)이 없어도 모델의 "행동 패턴"이 올바른지 검증 가능.
#   NLP CheckList 논문 (NeurIPS 2020)에서 제안된 방법.
#   모델 업데이트 후 기존 행동이 깨지지 않는지 회귀 테스트로 재사용.
#
# 사전 조건: 학습된 체크포인트가 checkpoints/best.ckpt에 있어야 합니다.
#   학습 전에는 이 테스트를 건너뜁니다 (pytest.skip 처리).
#
# 실행 방법:
#   python -m pytest tests/test_behavior.py -v -s

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from transformers import AutoTokenizer

LABELS = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "best.ckpt"
)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """
    학습된 모델과 토크나이저를 로드.
    체크포인트가 없으면 전체 모듈의 테스트를 건너뜀.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        pytest.skip(
            f"체크포인트 없음: {CHECKPOINT_PATH}\n"
            "먼저 train.py를 실행하여 모델을 학습하세요."
        )

    from src.model import NewsClassifier
    model = NewsClassifier.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    return model, tokenizer


def _predict(model, tokenizer, text: str) -> str:
    """
    단일 텍스트를 분류하여 레이블 이름 반환.
    Helper 함수 — 테스트에서 반복 사용.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    # 모델과 같은 디바이스로 입력 이동
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model.model(**inputs).logits  # (1, 7)

    pred_id = torch.argmax(logits, dim=-1).item()
    return LABELS[pred_id]


def test_sports_keywords(model_and_tokenizer):
    """
    [불변성 테스트] 스포츠 키워드가 있으면 스포츠로 분류해야 함.
    "손흥민", "올림픽", "리그" 등 명확한 스포츠 기사는 반드시 스포츠 예측.
    """
    model, tokenizer = model_and_tokenizer
    sports_titles = [
        "손흥민, 프리미어리그 득점왕 등극",
        "파리올림픽 한국 선수단 금메달 획득",
        "KBO 리그 개막전 입장권 매진",
    ]
    for title in sports_titles:
        pred = _predict(model, tokenizer, title)
        assert pred == "스포츠", \
            f"스포츠 기사를 '{pred}'로 분류: '{title}'"
    print(f"\n✅ 스포츠 키워드 불변성 테스트 통과 ({len(sports_titles)}건)")


def test_it_keywords(model_and_tokenizer):
    """
    [불변성 테스트] IT/반도체 키워드가 있으면 IT과학으로 분류해야 함.
    """
    model, tokenizer = model_and_tokenizer
    it_titles = [
        "삼성전자, 차세대 반도체 공정 개발 성공",
        "애플 아이폰 신모델 발표… AI 기능 대폭 강화",
        "오픈AI GPT-5 출시 발표",
    ]
    for title in it_titles:
        pred = _predict(model, tokenizer, title)
        assert pred == "IT과학", \
            f"IT 기사를 '{pred}'로 분류: '{title}'"
    print(f"\n✅ IT 키워드 불변성 테스트 통과 ({len(it_titles)}건)")


def test_category_consistency(model_and_tokenizer):
    """
    [일관성 테스트] 동일한 입력은 항상 동일한 결과를 반환해야 함.
    모델이 eval() 모드이면 Dropout이 비활성화되어 결정론적.
    이것이 깨지면 eval() 모드 설정 오류.
    """
    model, tokenizer = model_and_tokenizer
    title = "코스피 3000 돌파, 외국인 순매수 확대"

    results = [_predict(model, tokenizer, title) for _ in range(3)]
    assert len(set(results)) == 1, \
        f"동일 입력에 다른 결과: {results} — model.eval() 확인 필요"
    print(f"\n✅ 일관성 테스트 통과: '{title}' → '{results[0]}' (3회 동일)")
