# NLP04 - mini BERT Pretrain

vocab_size=8000, 전체 파라미터 ~1M의 mini BERT를 PyTorch로 구현하고 MLM + NSP 태스크로 10 Epoch 사전학습합니다.

## 모델 사양

| 항목 | 값 |
|------|-----|
| vocab_size | 8,000 |
| d_model | 128 |
| num_heads | 4 |
| num_layers | 3 |
| d_ff | 256 |
| max_seq_len | 128 |
| 총 파라미터 | ~1M |

## 폴더 구조

```
NLP04/
├── data/
│   ├── raw/            # kowiki dump, corpus.txt
│   └── processed/      # spm.model, spm.vocab, memmap 데이터
├── models/             # 체크포인트, 학습 로그, 그래프
├── scripts/
│   ├── 01_build_tokenizer.py   # SentencePiece BPE 학습
│   ├── 02_preprocess_data.py   # MLM + NSP 전처리 → memmap 저장
│   └── 03_pretrain.py          # mini BERT 구현 및 학습
├── config.json
└── README.md
```

## 실행 순서

```bash
# 1. 토크나이저 학습 (kowiki dump 다운로드 포함)
python scripts/01_build_tokenizer.py

# 2. 전처리 (MLM + NSP)
python scripts/02_preprocess_data.py
# 샘플 수 제한 시: python scripts/02_preprocess_data.py --max_samples 100000

# 3. 사전학습
python scripts/03_pretrain.py
```

## 핵심 구현

### MLM (Masked Language Modeling)
- 전체 토큰의 15% 선택
- 선택된 토큰 중 80% → [MASK], 10% → 랜덤 토큰, 10% → 원본 유지
- `mlm_labels`: 마스킹 위치만 실제 ID, 나머지는 -100 (CrossEntropyLoss ignore)

### NSP (Next Sentence Prediction)
- 50% IsNext(1): 실제 연속 문장 쌍
- 50% NotNext(0): 랜덤 두 번째 문장
- 포맷: `[CLS] 문장A [SEP] 문장B [SEP]`
- segment_ids: 문장A=0, 문장B=1

### 학습
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: WarmupLinearSchedule (warmup=전체 step의 10%)
- Gradient clipping: max_norm=1.0
- total_loss = mlm_loss + nsp_loss

## 출력 결과

- `models/bert_best.pt`: 최적 체크포인트
- `models/bert_final.pt`: 최종 체크포인트
- `models/epoch_XX_log.json`: epoch별 loss/accuracy
- `models/train_history.json`: 전체 학습 히스토리
- `models/pretrain_loss.png`: loss 시각화 그래프

## 의존 패키지

```bash
pip install torch sentencepiece numpy tqdm matplotlib --break-system-packages
```

---

# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 신기주
- 리뷰어 : 박항아
# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
        -  
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        <img width="1223" height="673" alt="image" src="https://github.com/user-attachments/assets/83a79ec7-0ca2-402a-ab68-78ee4b6cd0b4" />

- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        
- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부


# 리뷰
```
# 전체코드가 아니라 전체 코드에 대한 리포드라서 정확한 판단은 어렵지만, 노드 학습 내용의 발전하기 보단, 바이브 코딩으로 작성되었다는 인상이 있습니다.
# 확인해보니 공유받은 https://github.com/count-im/test/blob/main/NLP/NLP04/miniBERT/NLP04_miniBERT_submit.ipynb 는 완성된 miniBEAR 코드가 아닌 ai 를 통해 별도 제출용으로 만든 것이 라 합니다.
# '주어진 문제를 해결하는 완성된 코드가 제출'되었는지 리뷰에 체크하는 부분이 있기에 다음에는 완성된 코드도 같이 올려주시면 좋을 것 같습니다.
# 전체적인 코드는 확인할 수 없지만, 정리가 깔끔하게 잘되어 있고, 파라미터 수를 변화시키면서 학습하며 학습률을 끌어올린 부분, 테스트 결과를 json으로 저장하여 체계적으로 정리한 부분이 인상 깊었습니다.
```

# 회고(참고 링크 및 코드 개선)
```
# 리뷰어의 회고를 작성합니다.
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
