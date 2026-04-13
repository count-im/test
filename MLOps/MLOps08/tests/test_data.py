# tests/test_data.py
# FSDL Day3 - 데이터 기댓값 테스트 (Expectation Testing)
#
# 핵심 원리 (Day3 Ch.13):
#   "데이터는 새로운 코드다."
#   데이터 오류는 에러 없이 조용히 모델을 망친다.
#   모델 학습 전에 파이프라인에 '불이 났는지' 먼저 확인한다.
#
# 실행 방법 (RunPod 터미널):
#   cd /workspace/mycoding/test/MLOps08
#   python -m pytest tests/test_data.py -v

import pytest
from datasets import load_dataset
from collections import Counter


@pytest.fixture(scope="module")
def ynat_data():
    """
    모듈 전체에서 데이터를 한 번만 로드.
    scope="module": 파일 내 모든 테스트가 같은 객체를 공유 → 시간 절약.
    """
    return load_dataset("klue", "ynat")


def test_dataset_not_empty(ynat_data):
    """
    [스모크 테스트] 데이터가 존재하는가?
    가장 기본적인 검사. 이것이 실패하면 다운로드 혹은 캐시 문제.
    """
    assert len(ynat_data['train']) > 1000, "학습 데이터가 너무 적음 (다운로드 확인)"
    assert len(ynat_data['validation']) > 100, "검증 데이터가 너무 적음"


def test_no_missing_titles(ynat_data):
    """
    [결측치 테스트] 제목이 비어있는 샘플이 없어야 함.
    느슨한 경계 원칙: 완전히 빈 제목만 차단 (공백 1개는 허용).
    """
    for split in ['train', 'validation']:
        missing = [i for i, t in enumerate(ynat_data[split]['title'])
                   if t is None or t.strip() == ""]
        assert len(missing) == 0, \
            f"[{split}] 빈 제목 발견: 인덱스 {missing[:5]}"


def test_label_range(ynat_data):
    """
    [레이블 범위 테스트] 레이블이 0~6 범위에 있어야 함.
    범위 밖의 레이블은 분류 헤드에서 IndexError를 일으킴.
    """
    num_classes = 7
    for split in ['train', 'validation']:
        labels = ynat_data[split]['label']
        invalid = [l for l in labels if not (0 <= l < num_classes)]
        assert len(invalid) == 0, \
            f"[{split}] 범위 밖 레이블 발견: {set(invalid)}"


def test_all_classes_present(ynat_data):
    """
    [클래스 균형 테스트] 학습 데이터에 모든 7개 카테고리가 존재해야 함.
    특정 클래스가 없으면 모델이 해당 클래스를 학습하지 못함.
    """
    unique_labels = set(ynat_data['train']['label'])
    expected = set(range(7))
    missing = expected - unique_labels
    assert len(missing) == 0, f"누락된 클래스: {missing}"


def test_title_length_reasonable(ynat_data):
    """
    [길이 테스트] 제목 길이가 합리적 범위에 있어야 함.
    느슨한 경계: 1자 미만 또는 1000자 초과만 비정상으로 처리.
    (실제 뉴스 제목은 보통 10~80자)
    """
    MAX_REASONABLE = 1000
    for split in ['train', 'validation']:
        too_long = [i for i, t in enumerate(ynat_data[split]['title'])
                    if len(t) > MAX_REASONABLE]
        assert len(too_long) == 0, \
            f"[{split}] {MAX_REASONABLE}자 초과 제목: {len(too_long)}건"
