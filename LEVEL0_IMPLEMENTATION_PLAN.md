# Level 0 (Atomic) Seeds 구현 계획

## 개요
표준 인지 시드 설계 가이드 v1.1에 따라 8개의 Level 0 (Atomic) 시드를 구현합니다.

## Level 0 시드 목록

### SEED-A01 — Edge Detector
- **Category**: Pattern
- **Bit**: INT8
- **Params**: ~128
- **Purpose**: 급격한 변화/경계 검출
- **I/O**: [B,*,D] → [B,*,D]
- **Invariance**: 부분적 평행이동
- **Train**: contrastive edge targets, self-supervised gradients
- **Eval**: Edge F1≥0.90
- **Use**: 세그먼트·경계·이벤트 분절 전처리

### SEED-A02 — Symmetry Detector
- **Category**: Spatial
- **Bit**: INT8
- **Params**: ~256
- **Purpose**: 반사/회전/병진 대칭성 추정
- **Invariance**: 회전(부분), 스케일(정규화 후)
- **Eval**: 축/정도 추정 정확도

### SEED-A03 — Recurrence Spotter
- **Category**: Temporal
- **Bit**: INT8
- **Params**: ~192
- **Purpose**: 반복/주기/모티프 검출
- **Eval**: 패턴 리콜/프리시전

### SEED-A04 — Contrast Amplifier
- **Category**: Pattern
- **Bit**: INT8
- **Params**: ~64
- **Purpose**: 신호 대비 증폭·노이즈 억제
- **Eval**: SNR 개선, 다운스트림 성능 ↑

### SEED-A05 — Grouping Nucleus
- **Category**: Relation
- **Bit**: INT8
- **Params**: ~256
- **Purpose**: 유사도 기반 클러스터 시드
- **Eval**: ARI/AMI, 바인딩 안정성

### SEED-A06 — Sequence Tracker
- **Category**: Temporal
- **Bit**: INT8
- **Params**: ~320
- **Purpose**: 순서 추적·다음 상태 예측
- **Eval**: next-step accuracy, perplexity

### SEED-A07 — Scale Normalizer
- **Category**: Abstraction
- **Bit**: INT8
- **Params**: ~128
- **Purpose**: 스케일/단위 정규화
- **Eval**: 분산 안정화, 폭주/언더플로 방지

### SEED-A08 — Binary Comparator
- **Category**: Logic
- **Bit**: INT8
- **Params**: ~96
- **Purpose**: 대소/동등 비교 원자 연산
- **Eval**: 임계 기반 분류 정확

## 구현 전략

### 1. 디렉토리 구조
```
seeds/
  atomic/
    SEED-A01_edge_detector/
      __init__.py
      model.py
      metadata.json
      README.md
    SEED-A02_symmetry_detector/
      ...
    (A03~A08)
```

### 2. 공통 아키텍처 요소
- **MGP (Multi-Geometry Projection)**: Euclidean, Hyperbolic, Spherical 투영
- **CSE (Continuous Scale-Equivariant)**: 스케일 조건부 정규화
- **Quantization**: INT8 추론 지원

### 3. 구현 순서
1. Base Seed 클래스 정의
2. MGP/CSE 블록 구현
3. 각 시드별 모델 구현
4. 메타데이터 및 문서화
5. 테스트 코드 작성

## 수용 기준
- **Level 0**: F1 ≥ 0.90, latency < 1ms/32샘플
- 각 시드는 독립적으로 실행 가능
- 표준 인터페이스 준수
- 메타데이터 및 문서 완비

