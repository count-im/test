# baseline.py
# FSDL Day3 - TF-IDF + Logistic Regression 베이스라인 (Day3 Ch.16 Step 3)
#
# 핵심 원리:
#   "첫 번째 버전은 단순한 통계로 시작하라." (FSDL Day1)
#   베이스라인 없이 BERT F1=0.87은 좋은지 나쁜지 알 수 없다.
#   TF-IDF 베이스라인이 0.82라면 → BERT가 5%p 향상, 도입 가치 있음.
#   베이스라인이 0.90이라면 → BERT 도입의 복잡성 대비 개선 폭이 작음.
#
# 실행:
#   cd /workspace/mycoding/test/MLOps08
#   python baseline.py

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


def main():
    print("데이터 로드 중...")
    dataset = load_dataset("klue", "ynat")

    label_names = dataset['train'].features['label'].names
    print(f"\n카테고리: {label_names}")
    print(f"학습 데이터: {len(dataset['train']):,}건")
    print(f"검증 데이터: {len(dataset['validation']):,}건")

    # ── TF-IDF 벡터화 ──
    # Term Frequency × Inverse Document Frequency:
    # 해당 문서에서 자주 등장하면서 전체 문서에서는 드문 단어에 높은 가중치
    print("\nTF-IDF 벡터화 중...")
    vectorizer = TfidfVectorizer(max_features=10000)

    X_train = vectorizer.fit_transform(dataset['train']['title'])
    X_val = vectorizer.transform(dataset['validation']['title'])
    y_train = dataset['train']['label']
    y_val = dataset['validation']['label']

    # ── Logistic Regression 학습 ──
    print("Logistic Regression 학습 중...")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    # ── 평가 ──
    y_pred = clf.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average='macro')

    print("\n" + "="*60)
    print(f"  TF-IDF + Logistic Regression 베이스라인 결과")
    print("="*60)
    print(f"\n  Macro F1: {macro_f1:.4f}  ← 이 수치를 BERT와 비교하세요\n")
    print(classification_report(y_val, y_pred, target_names=label_names))
    print("="*60)
    print(f"\n📌 실행 계획서 '베이스라인' 칸에 Macro F1 = {macro_f1:.4f} 를 기록하세요.")
    print("   BERT Fine-tuning 후 이 수치보다 얼마나 향상되었는지 비교합니다.\n")


if __name__ == "__main__":
    main()
