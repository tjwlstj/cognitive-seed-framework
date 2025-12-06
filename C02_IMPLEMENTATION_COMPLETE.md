# C02 Counterfactual Reasoner - 구현 완료 보고서

**시드 ID**: SEED-C02  
**시드 이름**: Counterfactual Reasoner  
**레벨**: 2 (Cellular)  
**카테고리**: Logic  
**구현일**: 2025-12-06  
**작성자**: Manus AI

---

## 1. 개요

**Counterfactual Reasoner**는 반사실 추론을 통해 "만약 ~했다면" 시나리오를 생성하고 평가하는 Cellular 레벨 시드입니다. 인과 구조를 파악하고 개입 효과를 시뮬레이션하여 대안 시나리오를 생성합니다.

### 1.1. 핵심 기능

- **인과 구조 파악**: M02를 활용하여 시계열 데이터의 인과 관계 추정
- **반사실 시나리오 생성**: 개입 지점 식별 및 대안 시나리오 생성
- **개입 효과 시뮬레이션**: 개입이 시스템에 미치는 영향 추정
- **일관성 검증**: 시간적/인과적 일관성 보장
- **다중 시나리오 생성**: 여러 대안 시나리오 생성 및 조화

### 1.2. 구성 시드

| 시드 ID | 시드 이름 | 역할 |
|---------|-----------|------|
| M02 | Causality Detector | 인과 구조 파악 및 개입 효과 추정 |
| M08 | Conflict Resolver | 대안 시나리오 간 충돌 해소 |
| A08 | Binary Comparator | 사실/반사실 시나리오 비교 |

---

## 2. 아키텍처 설계

### 2.1. 전체 구조

```
Input [B, T, D]
    ↓
┌───────────────────────────────────┐
│  Causality Detector (M02)         │
│  - 인과 구조 파악                  │
│  - 인과 그래프 생성                │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  Intervention Module              │
│  - 개입 지점 식별                  │
│  - 개입 값 생성                    │
│  - 개입 강도 조절                  │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  Scenario Generator (GRU)         │
│  - 반사실 시나리오 생성            │
│  - 개입 이후 시퀀스 재생성         │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  Consistency Checker              │
│  - 시간적 일관성 검증              │
│  - 인과적 일관성 검증              │
└───────────────┬───────────────────┘
                ↓
┌───────────────────────────────────┐
│  Binary Comparator (A08)          │
│  - 사실/반사실 비교                │
└───────────────┬───────────────────┘
                ↓
Output: Counterfactual [B, T, D]
        + Metadata
```

### 2.2. 주요 컴포넌트

#### 2.2.1. Scenario Generator
- **구조**: 2-layer GRU
- **기능**: 개입 이후 시퀀스 재생성
- **특징**: 인과 구조를 고려한 시나리오 생성

#### 2.2.2. Intervention Module
- **Intervention Locator**: 최적 개입 지점 식별
- **Intervention Effect Estimator**: 개입 효과 추정
- **Intervention Modulator**: 개입 강도 조절

#### 2.2.3. Consistency Checker
- **Temporal Consistency**: 인접 시점 간 일관성
- **Causal Consistency**: 인과 구조 보존
- **Overall Consistency**: 전체 일관성 평가

---

## 3. 입출력 규격

### 3.1. 입력

```python
x: torch.Tensor  # [B, T, D] - 사실적 시계열 데이터
scale: Optional[torch.Tensor]  # [B, 1] - 스케일 매개변수
intervention: Optional[Dict]  # 개입 정보
    - 'time_step': int - 개입 시점
    - 'value': [B, 1, D] - 개입 값
    - 'strength': float (optional) - 개입 강도
```

### 3.2. 출력

```python
counterfactual: torch.Tensor  # [B, T, D] - 반사실 시나리오

metadata: Dict
    - 'causal_graph': [B, T, T] - 인과 그래프
    - 'intervention_effects': [B, T, D] - 개입 효과
    - 'consistency_scores': Dict
        - 'temporal': [B, T] - 시간적 일관성
        - 'causal': [B, T] - 인과적 일관성
        - 'overall': [B] - 전체 일관성
    - 'comparison': Dict
        - 'features': 비교 특징
        - 'difference_magnitude': [B, T] - 차이 크기
        - 'similarity': [B] - 유사도
    - 'intervention_time': [B] - 개입 시점
```

---

## 4. 파라미터 분석

### 4.1. 파라미터 수

| 컴포넌트 | 파라미터 수 | 비율 |
|----------|-------------|------|
| Causality Detector (M02) | ~600K | 9.6% |
| Conflict Resolver (M08) | ~800K | 12.8% |
| Binary Comparator (A08) | ~96K | 1.5% |
| Scenario Generator | ~200K | 3.2% |
| Intervention Module | ~150K | 2.4% |
| Consistency Checker | ~100K | 1.6% |
| MGP/CSE | ~50K | 0.8% |
| 기타 | ~4,250K | 68.1% |
| **총합** | **~6.25M** | **100%** |

### 4.2. 메모리 사용량 (FP32 기준)

- **모델 파라미터**: ~25 MB
- **활성화 (batch=4, seq_len=10)**: ~5 MB
- **총 메모리**: ~30 MB

---

## 5. 구현 세부사항

### 5.1. 반사실 시나리오 생성 알고리즘

```python
def _generate_counterfactual_scenario(factual, intervention_time, intervention_value):
    # 1. 개입 시점까지 사실적 시나리오 유지
    counterfactual = factual.detach().clone()
    
    # 2. 개입 적용 (벡터화된 연산)
    intervention_mask = compute_intervention_mask(intervention_time)
    counterfactual = apply_intervention(
        counterfactual, 
        intervention_value, 
        intervention_mask
    )
    
    # 3. 개입 이후 시퀀스 재생성 (GRU)
    for each batch:
        prefix = counterfactual[:intervention_time]
        _, hidden = scenario_generator(prefix)
        
        # 나머지 시점 생성
        for step in range(intervention_time+1, seq_len):
            output, hidden = scenario_generator(current_input, hidden)
            append output to generated_steps
    
    return counterfactual
```

### 5.2. 일관성 검증

#### 시간적 일관성
- 인접 시점 간 특징 유사도 계산
- Consistency Checker를 통한 점수화

#### 인과적 일관성
- 인과 구조 보존 여부 확인
- 사실/반사실/인과 특징 3-way 비교

### 5.3. 주요 최적화

1. **Inplace 연산 제거**: 그래디언트 계산 안정성 확보
2. **벡터화된 개입**: 효율적인 배치 처리
3. **메모리 효율**: detach().clone()으로 불필요한 그래프 제거

---

## 6. 테스트 결과

### 6.1. 단위 테스트 (13개)

| 테스트 | 상태 | 설명 |
|--------|------|------|
| test_initialization | ✅ PASS | 초기화 검증 |
| test_forward_without_intervention | ✅ PASS | 자동 개입 생성 |
| test_forward_with_intervention | ✅ PASS | 명시적 개입 |
| test_intervention_effect_estimation | ✅ PASS | 개입 효과 추정 |
| test_consistency_checking | ✅ PASS | 일관성 검증 |
| test_scenario_comparison | ✅ PASS | 시나리오 비교 |
| test_multiple_scenarios_generation | ✅ PASS | 다중 시나리오 생성 |
| test_gradient_flow | ✅ PASS | 그래디언트 흐름 |
| test_parameter_count | ✅ PASS | 파라미터 수 검증 |
| test_create_function | ✅ PASS | 생성 함수 |
| test_config_dataclass | ✅ PASS | Config 데이터클래스 |
| test_intervention_value_generation | ✅ PASS | 자동 개입 값 생성 |
| test_counterfactual_scenario_generation | ✅ PASS | 반사실 시나리오 생성 |

**결과**: 13/13 통과 (100%)

### 6.2. 성능 테스트

```python
# 입력: batch=4, seq_len=10, dim=128
# 환경: CPU (Intel Xeon)

Forward pass: ~150ms
Backward pass: ~200ms
Memory: ~30MB
```

---

## 7. 사용 예제

### 7.1. 기본 사용법

```python
from seeds.cellular.c02_counterfactual_reasoner import CounterfactualReasoner

# 모델 생성
reasoner = CounterfactualReasoner(input_dim=128)

# 사실적 시나리오
factual = torch.randn(4, 10, 128)

# 반사실 추론 (자동 개입)
counterfactual, metadata = reasoner(factual)

print(f"Counterfactual shape: {counterfactual.shape}")
print(f"Consistency score: {metadata['consistency_scores']['overall'].mean():.3f}")
```

### 7.2. 명시적 개입

```python
# 개입 정의
intervention = {
    'time_step': 5,  # 5번째 시점에 개입
    'value': torch.randn(4, 1, 128)  # 개입 값
}

# 반사실 추론
counterfactual, metadata = reasoner(factual, intervention=intervention)

# 개입 효과 분석
effects = metadata['intervention_effects']
print(f"Intervention effects shape: {effects.shape}")
```

### 7.3. 다중 시나리오 생성

```python
# 5개의 대안 시나리오 생성
scenarios, metadata_list = reasoner.generate_multiple_scenarios(
    factual, 
    num_scenarios=5
)

print(f"Number of scenarios: {len(scenarios)}")
print(f"Last scenario is harmonized: {metadata_list[-1]['type']}")
```

---

## 8. 통합 및 등록

### 8.1. __init__.py 업데이트

```python
# seeds/cellular/__init__.py

from .c01_metaphor_engine import MetaphorEngine, create_metaphor_engine
from .c02_counterfactual_reasoner import (
    CounterfactualReasoner, 
    create_counterfactual_reasoner
)
from .c03_schema_learner import SchemaLearner, create_schema_learner

__all__ = [
    'MetaphorEngine',
    'create_metaphor_engine',
    'CounterfactualReasoner',
    'create_counterfactual_reasoner',
    'SchemaLearner',
    'create_schema_learner',
]
```

### 8.2. Git 커밋

```bash
git add seeds/cellular/c02_counterfactual_reasoner.py
git add tests/cellular/test_c02_counterfactual_reasoner.py
git add C02_IMPLEMENTATION_COMPLETE.md
git commit -m "feat: Implement C02 Counterfactual Reasoner

- Add counterfactual reasoning with causal structure
- Implement intervention effect simulation
- Add consistency checking (temporal & causal)
- Add multiple scenario generation
- 13 unit tests (100% pass)
- ~6.25M parameters"
```

---

## 9. 향후 개선 사항

### 9.1. 단기 개선

1. **개입 최적화**: 최적 개입 지점 자동 탐색 알고리즘 개선
2. **일관성 강화**: 더 정교한 일관성 검증 메트릭
3. **효율성 향상**: 시퀀스 재생성 속도 최적화

### 9.2. 장기 개선

1. **인과 발견**: 자동 인과 구조 발견 알고리즘 통합
2. **다중 개입**: 여러 시점에 동시 개입 지원
3. **불확실성 추정**: 반사실 시나리오의 불확실성 정량화

---

## 10. 결론

### 10.1. 달성 사항

- ✅ C02 Counterfactual Reasoner 완전 구현
- ✅ 13개 단위 테스트 100% 통과
- ✅ 인과 구조 기반 반사실 추론 구현
- ✅ 다중 시나리오 생성 및 조화 기능
- ✅ 일관성 검증 메커니즘 구축

### 10.2. 프로젝트 진행 상황

| 레벨 | 완료 | 전체 | 진행률 |
|------|------|------|--------|
| Level 0 (Atomic) | 8 | 8 | 100% ✅ |
| Level 1 (Molecular) | 8 | 8 | 100% ✅ |
| Level 2 (Cellular) | 3 | 8 | 37.5% 🟡 |
| Level 3 (Tissue) | 0 | 8 | 0% 📅 |
| **전체** | **19** | **32** | **59.4%** |

### 10.3. 다음 단계

**세션 3**: C06 Attention Director 구현
- 의존 시드: M06, M01, A05 (모두 완료 ✅)
- 예상 기간: 5-7일
- 우선순위: P0-2

---

**구현 완료**: 2025-12-06  
**작성자**: Manus AI  
**검토자**: -  
**승인자**: -
