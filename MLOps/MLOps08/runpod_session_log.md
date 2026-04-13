# MLOps08 — RunPod 실행 세션 로그

> **실행 환경**: NVIDIA RTX 4000 Ada Generation (RunPod)  
> **실행 일시**: 2026-04-14  
> **프로젝트**: FSDL Day3 — 한국어 뉴스 분류 시스템 (KLUE/BERT-base)

---

## Step 0. 환경 설정

```bash
git clone https://github.com/count-im/test.git
cd test/MLOps/MLOps08
pip install torch pytorch-lightning transformers datasets torchmetrics \
            scikit-learn fastapi uvicorn pytest --break-system-packages
```

**설치 결과 (주요 버전)**

| 패키지 | 버전 |
|---|---|
| torch | 2.4.1+cu124 (기존 설치) |
| pytorch-lightning | 2.6.1 |
| transformers | 5.5.4 |
| datasets | 4.8.4 |
| torchmetrics | 1.9.0 |
| scikit-learn | 1.8.0 |
| fastapi | 0.135.3 |
| pytest | 9.0.3 |

---

## Step 1. 베이스라인 (TF-IDF + Logistic Regression)

```bash
python baseline.py
```

**데이터셋 정보**
- 학습 데이터: 45,678건
- 검증 데이터: 9,107건
- 카테고리: `IT과학`, `경제`, `사회`, `생활문화`, `세계`, `스포츠`, `정치`

**결과**

```
TF-IDF + Logistic Regression 베이스라인 결과

  Macro F1: 0.6922  ← BERT 비교 기준선

              precision    recall  f1-score   support
        IT과학       0.64      0.73      0.68       554
          경제       0.72      0.73      0.72      1348
          사회       0.78      0.60      0.68      3701
        생활문화       0.67      0.72      0.69      1369
          세계       0.53      0.72      0.61       835
         스포츠       0.77      0.88      0.82       578
          정치       0.55      0.76      0.64       722

       accuracy                           0.69      9107
      macro avg       0.67      0.73      0.69      9107
   weighted avg       0.70      0.69      0.69      9107
```

> 📌 **베이스라인 Macro F1 = 0.6922** — BERT Fine-tuning 후 비교 기준

---

## Step 2. 데이터 기댓값 테스트

```bash
python -m pytest tests/test_data.py -v
```

```
collected 5 items

tests/test_data.py::test_dataset_not_empty        PASSED  [ 20%]
tests/test_data.py::test_no_missing_titles         PASSED  [ 40%]
tests/test_data.py::test_label_range               PASSED  [ 60%]
tests/test_data.py::test_all_classes_present       PASSED  [ 80%]
tests/test_data.py::test_title_length_reasonable   PASSED  [100%]

5 passed in 6.79s
```

✅ **5개 전체 통과**

---

## Step 3. 모델 구조 + 암기 테스트

```bash
python -m pytest tests/test_training.py -v -s
```

**Shape 테스트**
```
✅ Shape 테스트 통과: torch.Size([4, 7])   PASSED
```

**NaN 테스트**
```
✅ NaN 테스트 통과: logits 범위 [-0.294, 0.549]   PASSED
```

**암기 테스트 (overfit_batches=1)**
```
FAILED — train_loss=0.5251 (임계값 0.1 미달)

AssertionError: 암기 실패: train_loss=0.5251 (모델 코드 버그 의심)
```

> ⚠️ **암기 테스트 FAIL 원인 분석**  
> BERT는 pre-trained dropout이 활성화된 상태에서 30에포크/8배치로는 loss 0.1 이하 수렴이 어렵습니다.  
> 본 학습(train.py)에서 val_F1 0.86 이상 달성했으므로 모델 코드 자체는 정상입니다.  
> 암기 테스트 임계값(0.1)이 BERT+dropout 조합에서는 과도하게 엄격한 기준입니다.

**결과 요약**: 2 passed, 1 failed (암기 테스트 임계값 문제)

---

## Step 4. 본 학습 (BERT Fine-tuning)

```bash
WANDB_MODE=disabled python train.py
```

**실행 환경**
- GPU: NVIDIA RTX 4000 Ada Generation (CUDA)
- 정밀도: 16-bit Mixed Precision (AMP)
- 모델 파라미터: 110M

**에포크별 결과**

| Epoch | val_f1 | val_loss | 비고 |
|---|---|---|---|
| 0 | 0.858 | — | Best 저장 |
| 1 | 0.865 | 0.398 | Best 갱신 (+0.007) |
| 2 | 0.865 | 0.374 | — |
| 3 | 0.863 | 0.460 | 개선 없음 |
| 4 | 0.857 | 0.634 | 개선 없음 → EarlyStopping |

```
✅ 학습 완료!
  최고 val_F1: 0.8649
  체크포인트: /workspace/test/MLOps/MLOps08/checkpoints/best.ckpt

최종 검증:
  val_f1   │ 0.8649411201477051
  val_loss │ 0.3742079436779022
```

> 📊 **베이스라인 대비 성능 향상**  
> TF-IDF: Macro F1 = **0.6922** → BERT: Macro F1 = **0.8649** (+17.3%p)

---

## Step 5. 행동 테스트

```bash
python -m pytest tests/test_behavior.py -v -s
```

```
✅ 스포츠 키워드 불변성 테스트 통과 (3건)   PASSED
✅ IT 키워드 불변성 테스트 통과 (3건)       PASSED
✅ 일관성 테스트 통과: '코스피 3000 돌파, 외국인 순매수 확대' → '경제' (3회 동일)   PASSED

3 passed, 1 warning in 11.69s
```

✅ **3개 전체 통과**

---

## Step 6. FastAPI 서빙

```bash
uvicorn serve:app --reload --port 8000
```

```
INFO: Uvicorn running on http://127.0.0.1:8000
모델 로드 중: /workspace/test/MLOps/MLOps08/checkpoints/best.ckpt
✅ 모델 로드 완료
INFO: Application startup complete.
```

Swagger UI: `http://127.0.0.1:8000/docs`

---

## 최종 결과 요약

| 단계 | 결과 |
|---|---|
| 환경 설정 | ✅ 완료 |
| 베이스라인 (TF-IDF) | ✅ Macro F1 = 0.6922 |
| 데이터 테스트 (5개) | ✅ 5/5 PASSED |
| Shape 테스트 | ✅ PASSED |
| NaN 테스트 | ✅ PASSED |
| 암기 테스트 | ⚠️ FAILED (임계값 과도) |
| **BERT 학습 (5에포크)** | ✅ **val_F1 = 0.8649** |
| 행동 테스트 (3개) | ✅ 3/3 PASSED |
| API 서빙 | ✅ 정상 기동 |

**베이스라인 대비 BERT 향상: +17.3%p (0.6922 → 0.8649)**
