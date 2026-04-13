# train.py
# FSDL Day3 - 뉴스 분류 모델 학습 스크립트
#
# 실행 방법 (RunPod 터미널):
#   cd /workspace/mycoding/test/MLOps08
#   python train.py
#
# 또는 W&B 없이 실행하려면 환경 변수 설정:
#   WANDB_MODE=disabled python train.py
#
# RunPod 주의사항:
#   - pod는 Stop (Terminate 금지)
#   - 체크포인트가 /workspace/ 아래에 저장되어야 pod 재시작 후에도 유지됨

import os
import sys

# RunPod 절대경로 처리
# __file__ 대신 os.path.abspath를 활용
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

import pytorch_lightning as pl
from src.model import NewsClassifier
from src.data import NewsDataModule


def main():
    # ── 재현성 보장 ──
    pl.seed_everything(42, workers=True)

    # ── 모델 초기화 ──
    model = NewsClassifier(
        model_name="klue/bert-base",
        num_labels=7,
        lr=2e-5,
        warmup_ratio=0.1,
    )

    # ── 데이터 초기화 ──
    data = NewsDataModule(
        model_name="klue/bert-base",
        max_length=128,
        batch_size=32,
        # num_workers=None: OS별 자동 설정 (Linux → 4, Windows → 0)
    )

    # ── 콜백 설정 ──
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(PROJECT_ROOT, "checkpoints"),
        filename="best",            # best.ckpt로 저장
        monitor="val_f1",           # 검증 F1이 가장 높을 때 저장
        mode="max",
        save_top_k=1,               # 최고 성능 1개만 유지 (디스크 절약)
        verbose=True,
    )

    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor="val_f1",
        patience=2,                 # 2에포크 연속 개선 없으면 조기 종료
        mode="max",
        verbose=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # ── Trainer 설정 ──
    trainer = pl.Trainer(
        accelerator="auto",            # GPU/CPU 자동 감지
        devices=1,
        precision="16-mixed",          # 혼합 정밀도: 메모리 절약 + 속도 향상
        max_epochs=5,                  # BERT Fine-tuning은 3~5 에포크면 충분
        gradient_clip_val=1.0,         # 그래디언트 클리핑: NaN/폭주 방지

        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],

        # 로깅: W&B (WANDB_MODE=disabled로 끌 수 있음)
        # logger=pl.loggers.WandbLogger(project="news-classifier"),

        # 디버깅용 옵션 (주석 해제하여 사용):
        # fast_dev_run=True,         # 배치 1개로 전체 파이프라인 점검
        # limit_train_batches=0.1,   # 학습 데이터 10%만 사용 (빠른 확인)
        # profiler="simple",         # 병목 구간 측정
    )

    print("\n" + "="*60)
    print("  한국어 뉴스 분류기 학습 시작")
    print(f"  프로젝트 루트: {PROJECT_ROOT}")
    print(f"  체크포인트 저장: {os.path.join(PROJECT_ROOT, 'checkpoints')}")
    print("="*60 + "\n")

    trainer.fit(model, datamodule=data)

    print("\n" + "="*60)
    print(f"  ✅ 학습 완료!")
    print(f"  최고 val_F1: {checkpoint_cb.best_model_score:.4f}")
    print(f"  체크포인트: {checkpoint_cb.best_model_path}")
    print("="*60 + "\n")

    # ── 최종 검증 ──
    trainer.validate(model, datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    main()
