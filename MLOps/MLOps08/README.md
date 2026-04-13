# MLOps08 — FSDL Day3 프로젝트: 한국어 뉴스 분류 시스템

> FSDL(Full Stack Deep Learning) Day3 교재 Chapter 16.3의 예시 프로젝트를 직접 구현한 코드입니다.
> KLUE/BERT-base를 Fine-tuning하여 한국어 뉴스 제목을 7개 카테고리로 분류합니다.

## 프로젝트 구조

```
MLOps08/
├── src/
│   ├── model.py          # LightningModule — BERT 분류기
│   └── data.py           # LightningDataModule — KLUE-YNAT 파이프라인
├── tests/
│   ├── test_data.py      # 데이터 기댓값 테스트 (Day3 Ch.13.4)
│   ├── test_training.py  # Shape/NaN/암기 테스트 (Day3 Ch.13.5)
│   └── test_behavior.py  # 행동 테스트 (Day3 Ch.13.6) ← 학습 후 실행
├── notebooks/
│   └── exploration.ipynb # 데이터 탐색 + 베이스라인
├── checkpoints/          # 학습된 체크포인트 저장 위치
├── baseline.py           # TF-IDF 베이스라인 스크립트
├── train.py              # 메인 학습 스크립트
├── serve.py              # FastAPI 서빙 스크립트
└── requirements.in       # 의존성 목록
```

## 실행 순서 (RunPod 환경)

### 0. 환경 설정

```bash
cd /workspace/mycoding/test/MLOps08

pip install torch pytorch-lightning transformers datasets torchmetrics \
            scikit-learn fastapi uvicorn pydantic pytest --break-system-packages
```

### 1. 데이터 탐색 + 베이스라인

```bash
python baseline.py
```
→ TF-IDF + Logistic Regression의 Macro F1을 확인하고 기록해두세요.

### 2. 데이터 테스트 (모델 학습 전 필수)

```bash
python -m pytest tests/test_data.py -v
```
→ 5개 테스트가 모두 PASSED여야 다음 단계로 진행합니다.

### 3. 모델 구조 테스트 (암기 테스트 포함)

```bash
python -m pytest tests/test_training.py -v -s
```
→ Shape / NaN / 암기 테스트 통과 확인 (약 2~3분 소요).

### 4. 본 학습

```bash
python train.py
```
→ W&B 없이 실행하려면: `WANDB_MODE=disabled python train.py`
→ 약 3에포크, GPU 기준 20~40분 소요.

### 5. 행동 테스트 (학습 후)

```bash
python -m pytest tests/test_behavior.py -v -s
```

### 6. API 서빙

```bash
uvicorn serve:app --reload --port 8000
```

API 테스트:
```bash
curl -X POST http://localhost:8000/classify \
     -H "Content-Type: application/json" \
     -d '{"title": "삼성전자 반도체 새 공정 개발"}'
```

## FSDL 원칙 적용 체크리스트

| 원칙 | 적용 위치 |
|---|---|
| Make it Run | test_training.py — Shape/NaN/암기 테스트 |
| Make it Fast | train.py — precision="16-mixed", num_workers 최적화 |
| Make it Right | test_behavior.py — 행동 테스트, val_f1 모니터링 |
| 베이스라인 먼저 | baseline.py — TF-IDF로 기준선 확보 |
| 데이터 테스트 먼저 | test_data.py — 학습 전 파이프라인 검증 |
| 결정적 빌드 | requirements.in + pl.seed_everything(42) |

## 카테고리

`IT과학` `경제` `사회` `생활문화` `세계` `스포츠` `정치`

## 참고

- 데이터셋: [KLUE-YNAT](https://huggingface.co/datasets/klue/ynat)
- 베이스 모델: [klue/bert-base](https://huggingface.co/klue/bert-base)
- 교재: FSDL Day3 (박광석, 모두의연구소, 2026.04)
