# Level 0 — Atomic Seeds

Level 0 (Atomic) 시드는 **8개의 기본 인지 기능**을 제공하는 원자 단위 모듈입니다. 각 시드는 독립적으로 실행 가능하며, 상위 레벨 시드의 구성 요소로 사용됩니다.

## 아키텍처 특징

모든 Atomic 시드는 다음 공통 아키텍처를 기반으로 구현되었습니다.

### Multi-Geometry Projection (MGP)

입력을 세 가지 기하학적 공간으로 투영하여 다양한 데이터 구조를 효과적으로 처리합니다.

- **Euclidean (E)**: 일반적인 연속 특징 처리
- **Hyperbolic (H)**: 계층적 관계 표현 (Poincaré ball model)
- **Spherical (S)**: 주기적/방향성 특징 표현 (unit sphere)

게이트 네트워크가 각 기하학 공간의 가중치를 동적으로 결정하여 최적의 표현을 학습합니다.

### Continuous Scale-Equivariant (CSE)

연속 스케일 매개변수를 받아 조건부 정규화(FiLM)를 수행합니다. 이를 통해 입력 스케일 변화에 강건한 처리가 가능합니다.

## 시드 카탈로그

### SEED-A01 — Edge Detector

**Category**: Pattern  
**Params**: ~100K  
**Purpose**: 급격한 변화/경계 검출

경계 검출을 위한 컨볼루션 기반 시드입니다. 시퀀스 데이터에서 급격한 변화 지점을 탐지하고, 경계 강도를 추정합니다.

**주요 기능**:
- `forward(x)`: 경계 정보가 인코딩된 텐서 반환
- `detect_edges(x, threshold)`: 이진 경계 마스크 반환

**사용 예시**:
```python
from seeds.atomic import EdgeDetector

seed = EdgeDetector(input_dim=128, hidden_dim=64)
output = seed(x)  # [B, L, D]
edge_mask = seed.detect_edges(x, threshold=0.5)  # [B, L]
```

### SEED-A02 — Symmetry Detector

**Category**: Spatial  
**Params**: ~109K  
**Purpose**: 반사/회전/병진 대칭성 추정

대칭성을 탐지하고 분류하는 시드입니다. 반사, 회전, 병진 대칭을 구분하고 대칭 축을 추정합니다.

**주요 기능**:
- `forward(x)`: 대칭 정보가 인코딩된 텐서 반환
- `detect_symmetry_type(x)`: 대칭 유형 확률과 축 반환
- `compute_symmetry_score(x)`: 대칭성 점수 계산

**사용 예시**:
```python
from seeds.atomic import SymmetryDetector

seed = SymmetryDetector(input_dim=128, hidden_dim=128)
symmetry_types, symmetry_axis = seed.detect_symmetry_type(x)
score = seed.compute_symmetry_score(x)
```

### SEED-A03 — Recurrence Spotter

**Category**: Temporal  
**Params**: ~267K  
**Purpose**: 반복/주기/모티프 검출

시간적 반복 패턴을 탐지하는 LSTM 기반 시드입니다. 주기성을 추정하고 반복 패턴의 유사도를 계산합니다.

**주요 기능**:
- `forward(x)`: 반복 패턴 정보가 인코딩된 텐서 반환
- `detect_period(x)`: 추정 주기 반환
- `compute_recurrence_score(x, window_size)`: 반복성 점수 계산

**사용 예시**:
```python
from seeds.atomic import RecurrenceSpotter

seed = RecurrenceSpotter(input_dim=128, hidden_dim=96)
period = seed.detect_period(x)
recurrence_score = seed.compute_recurrence_score(x, window_size=5)
```

### SEED-A04 — Contrast Amplifier

**Category**: Pattern  
**Params**: ~50K  
**Purpose**: 신호 대비 증폭·노이즈 억제

신호 대 노이즈 비율(SNR)을 개선하는 경량 시드입니다. 대비를 증폭하고 작은 노이즈를 억제합니다.

**주요 기능**:
- `forward(x)`: 대비가 증폭된 텐서 반환
- `compute_snr(x, output)`: SNR 개선 비율 계산
- `denoise(x, threshold)`: 노이즈 제거

**사용 예시**:
```python
from seeds.atomic import ContrastAmplifier

seed = ContrastAmplifier(input_dim=128, hidden_dim=32)
output = seed(x)
snr_improvement = seed.compute_snr(x, output)
denoised = seed.denoise(x, threshold=0.1)
```

### SEED-A05 — Grouping Nucleus

**Category**: Relation  
**Params**: ~127K  
**Purpose**: 유사도 기반 클러스터 시드

유사한 요소를 그룹화하는 클러스터링 시드입니다. 학습 가능한 클러스터 중심을 사용하여 소프트/하드 할당을 수행합니다.

**주요 기능**:
- `forward(x)`: 그룹 정보가 인코딩된 텐서 반환
- `get_cluster_assignments(x)`: 소프트 클러스터 할당 확률 반환
- `get_hard_assignments(x)`: 하드 클러스터 인덱스 반환
- `compute_cluster_distances(x)`: 클러스터 중심까지의 거리 계산

**사용 예시**:
```python
from seeds.atomic import GroupingNucleus

seed = GroupingNucleus(input_dim=128, hidden_dim=128, num_clusters=8)
assignments = seed.get_cluster_assignments(x)
hard_assignments = seed.get_hard_assignments(x)
```

### SEED-A06 — Sequence Tracker

**Category**: Temporal  
**Params**: ~382K  
**Purpose**: 순서 추적·다음 상태 예측

시퀀스의 순서를 추적하고 미래 상태를 예측하는 GRU 기반 시드입니다. 위치 인코딩을 통해 순서 정보를 명시적으로 처리합니다.

**주요 기능**:
- `forward(x)`: 시퀀스 정보가 인코딩된 텐서 반환
- `predict_next(x, num_steps)`: 다음 상태 예측
- `compute_tracking_accuracy(x)`: 추적 정확도 계산

**사용 예시**:
```python
from seeds.atomic import SequenceTracker

seed = SequenceTracker(input_dim=128, hidden_dim=160)
predictions = seed.predict_next(x, num_steps=5)
accuracy = seed.compute_tracking_accuracy(x)
```

### SEED-A07 — Scale Normalizer

**Category**: Abstraction  
**Params**: ~59K  
**Purpose**: 스케일/단위 정규화

입력의 스케일을 추정하고 목표 스케일로 정규화하는 시드입니다. 분산 안정화와 오버플로우/언더플로우 방지 기능을 제공합니다.

**주요 기능**:
- `forward(x)`: 정규화된 텐서 반환
- `estimate_scale(x)`: 입력 스케일 추정
- `normalize_to_scale(x, target_scale)`: 특정 스케일로 정규화
- `compute_variance_stability(x)`: 분산 안정성 계산
- `check_overflow_underflow(x)`: 오버플로우/언더플로우 위험 체크

**사용 예시**:
```python
from seeds.atomic import ScaleNormalizer

seed = ScaleNormalizer(input_dim=128, hidden_dim=64)
normalized = seed.normalize_to_scale(x, target_scale=1.0)
stability = seed.compute_variance_stability(x)
```

### SEED-A08 — Binary Comparator

**Category**: Logic  
**Params**: ~47K  
**Purpose**: 대소/동등 비교 원자 연산

두 요소를 비교하여 대소 관계를 판정하는 논리 연산 시드입니다. 소프트 비교 확률과 하드 판정을 모두 제공합니다.

**주요 기능**:
- `forward(x)`: 비교 정보가 인코딩된 텐서 반환
- `compare(a, b)`: 비교 결과 확률 반환 (<, =, >)
- `is_less_than(a, b)`: a < b 판정
- `is_equal(a, b)`: a == b 판정
- `is_greater_than(a, b)`: a > b 판정
- `get_comparison_type(a, b)`: 비교 결과 유형 반환

**사용 예시**:
```python
from seeds.atomic import BinaryComparator

seed = BinaryComparator(input_dim=128, hidden_dim=48)
comparison = seed.compare(a, b)  # [B, 3]
is_less = seed.is_less_than(a, b)  # [B]
```

## 입출력 형식

모든 Atomic 시드는 표준화된 입출력 형식을 따릅니다.

### 입력
- **x**: `[B, L, D]` 형태의 텐서
  - `B`: 배치 크기
  - `L`: 시퀀스 길이
  - `D`: 특징 차원 (기본값: 128)
- **scale** (선택): `[B, 1]` 형태의 스케일 매개변수
- **context** (선택): 추가 맥락 정보 딕셔너리

### 출력
- **output**: `[B, L, D]` 형태의 텐서

## 평가 기준

Level 0 시드는 다음 기준을 만족해야 합니다.

- **Exactness**: F1 ≥ 0.90 (태스크별 표준 벤치마크)
- **Latency**: < 1ms/32샘플 (CPU 기준)
- **Robustness**: 스케일/노이즈 변동에 성능 편차 < 10%
- **Bit Depth**: INT8 양자화 지원 (추론 시)

## 테스트

모든 시드는 단위 테스트를 통과했습니다.

```bash
# 테스트 실행
PYTHONPATH=/home/ubuntu/cognitive-seed-framework:$PYTHONPATH python tests/test_atomic_seeds.py

# 사용 예제 실행
PYTHONPATH=/home/ubuntu/cognitive-seed-framework:$PYTHONPATH python examples/level0_usage.py
```

## 다음 단계

Level 0 시드를 조합하여 Level 1 (Molecular) 시드를 구성할 수 있습니다. 예를 들어:

- **M01 (Hierarchy Builder)** = A05 + A08 + A07
- **M02 (Causality Detector)** = A06 + A03 + A08
- **M03 (Pattern Completer)** = A03 + A06 + A01

자세한 내용은 [표준 인지 시드 설계 가이드 v1.1](../../docs/표준_인지_시드_설계_가이드_v_1.md)을 참조하세요.

## 라이선스

Apache License 2.0

