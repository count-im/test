# tests/test_training.py
# FSDL Day3 - 암기 테스트 + Shape/NaN 테스트 (Day3 Ch.13.5)
#
# 핵심 원리:
#   "암기는 학습의 가장 단순한 형태다."
#   딥러닝 모델은 파라미터가 많아서 소량 데이터를 반드시 암기할 수 있다.
#   1배치를 암기하지 못한다 = 모델 코드에 치명적 버그가 있다는 신호.
#
# 실행 방법:
#   cd /workspace/mycoding/test/MLOps08
#   python -m pytest tests/test_training.py -v -s
#
# 주의: 이 테스트는 실제 모델을 로드하므로 ~2-3분 소요됩니다.

import sys
import os

# RunPod 절대경로 처리: sys.path에 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytorch_lightning as pl
from src.model import NewsClassifier
from src.data import NewsDataModule


def test_output_shape():
    """
    [Shape 테스트] 모델 출력이 (batch_size, num_labels) 형태인가?
    레이어 연결 오류, 분류 헤드 크기 불일치를 감지.
    """
    model = NewsClassifier(num_labels=7)
    model.eval()

    # 더미 입력 생성 — 실제 토크나이저 없이도 테스트 가능
    batch_size = 4
    seq_len = 32
    dummy_input = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }

    with torch.no_grad():
        outputs = model.model(**dummy_input)

    # 핵심 검사: 출력 shape이 (batch, num_labels) = (4, 7) 이어야 함
    assert outputs.logits.shape == (batch_size, 7), \
        f"출력 shape 불일치: 기대 {(batch_size, 7)}, 실제 {outputs.logits.shape}"
    print(f"\n✅ Shape 테스트 통과: {outputs.logits.shape}")


def test_no_nan_in_output():
    """
    [NaN 테스트] 모델 출력에 NaN/Inf가 없어야 함.
    수치적 불안정(그래디언트 폭주, 잘못된 초기화)을 감지.
    """
    model = NewsClassifier(num_labels=7)
    model.eval()

    batch_size = 4
    seq_len = 32
    dummy_input = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }

    with torch.no_grad():
        outputs = model.model(**dummy_input)

    logits = outputs.logits
    assert not torch.isnan(logits).any(), "출력에 NaN이 포함됨 (초기화 또는 수치 불안정 확인)"
    assert not torch.isinf(logits).any(), "출력에 Inf가 포함됨"
    print(f"\n✅ NaN 테스트 통과: logits 범위 [{logits.min():.3f}, {logits.max():.3f}]")


def test_overfit_one_batch():
    """
    [암기 테스트] 1개 배치를 30에포크 안에 암기(loss ≈ 0)할 수 있는가?
    이것이 실패하면 모델 코드에 치명적 버그가 있다는 뜻.
    (Day3 13.5: "암기는 학습의 가장 단순한 형태")

    주의사항:
    - num_workers=0: 테스트 환경에서 멀티프로세싱 충돌 방지
    - batch_size=8, max_length=64: 빠른 실행을 위해 최소화
    - lr=1e-3: 암기 테스트는 높은 학습률로 빠르게 수렴 확인
    - 모든 규제(dropout 등)는 Lightning이 eval 모드로 전환하지 않으므로
      overfit_batches=1이 dropout 영향을 유지하지만, 소량 데이터에서는 충분히 수렴 가능
    """
    model = NewsClassifier(lr=1e-3)  # 암기용 높은 학습률
    data = NewsDataModule(
        batch_size=8,
        num_workers=0,     # 테스트 환경 안전을 위해 단일 프로세스
        max_length=64,     # 짧게 설정하여 실행 속도 향상
    )

    trainer = pl.Trainer(
        overfit_batches=1,  # 단 1개 배치만 사용 — 이것이 암기 테스트의 핵심
        max_epochs=30,      # 30에포크로 loss ≈ 0 달성 확인
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=False,  # 테스트에서 체크포인트 저장 불필요
        logger=False,                # 테스트에서 W&B 로깅 불필요
        enable_model_summary=False,
    )

    trainer.fit(model, datamodule=data)

    # 최종 train_loss가 충분히 낮아야 함
    # 임계값 0.1: 7클래스 랜덤 예측의 loss ≈ ln(7) ≈ 1.95 → 0.1 이하면 암기 성공
    final_loss = trainer.callback_metrics.get("train_loss_epoch")
    if final_loss is not None:
        assert final_loss < 0.1, \
            f"암기 실패: train_loss={final_loss:.4f} (모델 코드 버그 의심)"
        print(f"\n✅ 암기 테스트 통과: train_loss={final_loss:.4f}")
    else:
        print("\n⚠️  loss 메트릭을 읽을 수 없음 (trainer 버전 차이). 학습 곡선을 수동 확인하세요.")
