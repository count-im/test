# src/model.py
# FSDL Day3 - 한국어 뉴스 분류기 (LightningModule)
#
# 핵심 원리:
#   - LightningModule = PyTorch 모델 + 학습 로직의 캡슐화
#   - __init__: 모델 구조 정의
#   - training_step / validation_step: 한 배치의 처리 로직만 작성
#   - Trainer가 루프·GPU 이동·로깅을 자동으로 처리
#
# RunPod 환경 주의:
#   - os.chdir('/workspace/...') 방식으로 경로 설정
#   - __file__ 사용 금지

import torch
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torchmetrics import F1Score

# 7개 카테고리 레이블 (KLUE-YNAT 데이터셋 기준)
LABELS = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']


class NewsClassifier(pl.LightningModule):
    """
    KLUE/BERT-base 기반 한국어 뉴스 분류기.

    LightningModule 구조:
      __init__ → 모델/메트릭 정의
      _shared_step → 공통 forward + loss 계산 (중복 제거)
      training_step → train loss 기록
      validation_step → val loss + F1 기록
      configure_optimizers → AdamW + 선형 스케줄러
    """

    def __init__(
        self,
        model_name: str = "klue/bert-base",  # 한국어 사전학습 BERT
        num_labels: int = 7,                  # 카테고리 수
        lr: float = 2e-5,                     # BERT Fine-tuning 표준 학습률
        warmup_ratio: float = 0.1,            # 전체 스텝의 10%를 워밍업으로
    ):
        super().__init__()
        # save_hyperparameters: __init__ 인수를 self.hparams에 자동 저장
        # → 체크포인트에도 포함되어 load_from_checkpoint() 가능
        self.save_hyperparameters()

        # Hugging Face에서 사전학습 모델 로드
        # AutoModelForSequenceClassification: 마지막 레이어가 분류 헤드로 교체됨
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,  # 분류 헤드 크기 불일치 허용
        )

        # Macro F1: 클래스 불균형 시 소수 클래스를 공평하게 평가
        # task="multiclass" + average="macro"
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")

    def _shared_step(self, batch, stage: str):
        """
        학습/검증 공통 로직 — 코드 중복 제거.
        batch: tokenizer가 만든 딕셔너리 {"input_ids", "attention_mask", "labels"}
        stage: "train" 또는 "val" (로그 prefix로 사용)
        """
        # 모델 forward pass — 내부적으로 CrossEntropyLoss 계산
        outputs = self.model(
            input_ids=batch["input_ids"],           # (batch, seq_len)
            attention_mask=batch["attention_mask"],  # (batch, seq_len)
            labels=batch["labels"],                  # (batch,)
        )
        loss = outputs.loss   # 스칼라 텐서
        logits = outputs.logits  # (batch, num_labels)

        # on_step=True: 배치마다 로그 / on_epoch=True: 에포크마다 집계
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, "train")
        return loss  # Lightning이 이 loss로 backward() 자동 실행

    def validation_step(self, batch, batch_idx):
        loss, logits = self._shared_step(batch, "val")
        preds = torch.argmax(logits, dim=-1)  # (batch,) — 가장 높은 확률 클래스
        self.val_f1.update(preds, batch["labels"])  # 에포크 내 누적

    def on_validation_epoch_end(self):
        """에포크가 끝날 때 F1 집계 후 리셋"""
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        """
        AdamW + 선형 감쇠 스케줄러.
        BERT Fine-tuning 표준 설정:
          - AdamW: 가중치 감쇠(L2 규제)가 내장된 Adam
          - 선형 스케줄러: warmup_ratio 구간은 학습률을 선형 증가,
            이후 0으로 선형 감소 → 초기 학습 불안정 방지
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
        )

        # 전체 학습 스텝 수 계산 (Lightning이 제공)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 배치마다 스케줄러 업데이트
            },
        }
