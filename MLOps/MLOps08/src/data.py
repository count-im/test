# src/data.py
# FSDL Day3 - 뉴스 분류 데이터 파이프라인 (LightningDataModule)
#
# 핵심 원리:
#   LightningDataModule = 데이터의 캡슐화
#   prepare_data() → 다운로드 (멀티GPU 환경에서 한 번만 실행)
#   setup()        → 토크나이징·분할 (각 프로세스에서 실행)
#   *_dataloader() → DataLoader 반환
#
# RunPod 환경 주의:
#   - 절대경로 사용, __file__ 금지

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class NewsDataModule(pl.LightningDataModule):
    """
    KLUE-YNAT 데이터셋 파이프라인.
    한국어 뉴스 제목(title) → 토크나이징 → DataLoader.

    데이터셋 정보:
      - 출처: KLUE (Korean Language Understanding Evaluation) 벤치마크
      - 태스크: YNAT (Yonhap News Agency Topic Classification)
      - 카테고리: IT과학, 경제, 사회, 생활문화, 세계, 스포츠, 정치 (7개)
      - 학습: ~45,678건 / 검증: ~9,107건
    """

    def __init__(
        self,
        model_name: str = "klue/bert-base",
        max_length: int = 128,   # 뉴스 제목은 대부분 64자 이내 → 128이면 충분
        batch_size: int = 32,
        num_workers: int = None,  # None → OS 감지하여 자동 설정
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = None
        self.tokenizer = None

        # ── num_workers OS별 자동 설정 ──
        # Windows: fork()가 없어 num_workers>0이 주피터에서 오류를 일으킴 → 0으로 설정
        # Linux/macOS: CPU 코어 수 기반으로 설정
        if num_workers is None:
            if os.name == 'nt':  # Windows
                self.hparams.num_workers = 0
            else:
                self.hparams.num_workers = min(4, os.cpu_count() or 1)

    def prepare_data(self):
        """
        데이터 다운로드 단계.
        멀티GPU 환경에서 rank=0인 프로세스만 이 메서드를 실행하므로
        중복 다운로드가 발생하지 않는다.
        """
        load_dataset("klue", "ynat")                      # 허깅페이스 캐시에 저장
        AutoTokenizer.from_pretrained(self.hparams.model_name)  # 토크나이저 캐시

    def setup(self, stage=None):
        """
        데이터 전처리 단계. 각 GPU 프로세스에서 독립 실행.
        stage: "fit" (학습), "test" (테스트), None (모두)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        raw = load_dataset("klue", "ynat")

        # 토크나이징 함수 — 배치 단위로 적용 (map)
        def tokenize(batch):
            encoding = self.tokenizer(
                batch["title"],          # 분류 대상: 뉴스 제목
                max_length=self.hparams.max_length,
                padding="max_length",    # 모든 배치를 동일 길이로 패딩
                truncation=True,         # max_length 초과 시 자름
            )
            encoding["labels"] = batch["label"]  # 정수 레이블 (0~6)
            return encoding

        # map()은 전체 데이터셋에 함수를 적용하며 캐시를 지원
        self.dataset = raw.map(
            tokenize,
            batched=True,           # 배치 단위로 처리하면 훨씬 빠름
            remove_columns=["guid", "title", "url", "date"],  # 불필요한 컬럼 제거
        )
        # PyTorch 텐서로 변환
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,                            # 학습 시에는 반드시 셔플
            num_workers=self.hparams.num_workers,
            persistent_workers=(self.hparams.num_workers > 0),  # 워커를 에포크 간 유지
            pin_memory=True,                         # CPU→GPU 전송 속도 향상
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.hparams.batch_size,
            shuffle=False,                           # 검증 시에는 셔플 불필요
            num_workers=self.hparams.num_workers,
            persistent_workers=(self.hparams.num_workers > 0),
            pin_memory=True,
        )
