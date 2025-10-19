# Dynamic Neural Networks: A Survey - 핵심 내용

**출처**: Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, Yulin Wang  
**발행**: arXiv:2102.04906, Feb 2021  
**인용**: 1018회  
**분야**: Computer Vision, Machine Learning

## 핵심 개념

### 동적 신경망 (Dynamic Neural Networks)

**정의**: 추론 시 입력에 따라 구조나 파라미터를 적응적으로 조정할 수 있는 신경망

**정적 모델과의 차이**:
- **정적 모델**: 고정된 계산 그래프와 파라미터
- **동적 모델**: 입력에 따라 구조/파라미터 변경 가능

## 동적 신경망의 3대 장점

### 1. 효율성 (Efficiency)

동적 네트워크는 입력에 따라 계산 자원을 선택적으로 할당:
- 쉬운 샘플: 적은 계산 자원 사용
- 어려운 샘플: 더 많은 계산 자원 사용
- 결과: 전체 계산 비용 감소하면서 정확도 유지

**구체적 메커니즘**:
- Early Exiting: 쉬운 샘플은 초기 레이어에서 종료
- Layer Skipping: 불필요한 레이어 건너뛰기
- Dynamic Depth: 샘플별 네트워크 깊이 조정

### 2. 표현력 (Representation Power)

동적 네트워크는 더 강력한 표현 능력:
- 입력별 최적 구조 선택
- 다양한 데이터 분포에 적응
- 정적 모델 대비 더 높은 정확도

### 3. 적응성 (Adaptiveness)

동적 네트워크는 다양한 환경에 적응:
- 하드웨어 제약 (모바일, 엣지 디바이스)
- 변화하는 환경 (조명, 센서 변화)
- 다양한 정확도-효율성 트레이드오프 요구사항

## 동적 네트워크의 3가지 분류

### 1. Sample-wise Dynamic Networks

**각 샘플마다 데이터 의존적 구조/파라미터 사용**

#### 주요 기법

**Dynamic Depth (동적 깊이)**:
- Early Exiting: 중간 레이어에서 조기 종료
- Layer Skipping: 특정 레이어 건너뛰기
- Adaptive Networks: 샘플별 레이어 수 조정

**Dynamic Width (동적 너비)**:
- Channel Pruning: 채널 선택적 활성화
- Dynamic Convolution: 입력별 커널 조합

**Dynamic Routing (동적 라우팅)**:
- Multi-branch: 여러 경로 중 선택
- Tree structures: 트리 구조 탐색
- **Gating Functions**: 게이트로 경로 결정

### 2. Spatial-wise Dynamic Networks

**입력의 공간적/시간적 위치에 따라 적응**

#### 주요 기법

**Adaptive Inference**:
- 중요한 공간 영역에 더 많은 계산 할당
- 배경/덜 중요한 영역은 간단히 처리

**Attention Mechanisms**:
- Soft Attention: 가중치 기반 선택
- Hard Attention: 명시적 영역 선택

### 3. Temporal-wise Dynamic Networks

**시계열 데이터에서 시간 차원 적응**

#### 주요 기법

**Recurrent Attention**: RNN/LSTM에서 중요 시점 선택
**Frame Glimpsing**: 비디오에서 키프레임 선택
**Early Exiting**: 시퀀스 조기 종료

## 코어 아키텍처 설계 요소

### 1. Gating Functions (게이팅 함수)

**역할**: 어떤 모듈을 활성화할지 결정

**구현 방식**:
```python
# 의사코드: 게이팅 함수
def gating_function(x, modules):
    # 입력 분석
    features = feature_extractor(x)
    
    # 게이트 점수 계산
    gate_scores = gate_network(features)  # [0, 1] 범위
    
    # 모듈 선택
    selected_modules = []
    for i, score in enumerate(gate_scores):
        if score > threshold:
            selected_modules.append(modules[i])
    
    return selected_modules
```

**학습 방법**:
- Reinforcement Learning: 정책 기반
- Gumbel-Softmax: 미분 가능한 샘플링
- Straight-Through Estimator: 이산 선택 근사

### 2. Module Registry (모듈 레지스트리)

**역할**: 사용 가능한 모듈 관리

**설계 패턴**:
```python
# 의사코드: 모듈 레지스트리
class ModuleRegistry:
    def __init__(self):
        self.modules = {}
        
    def register(self, name, module, metadata):
        self.modules[name] = {
            'module': module,
            'level': metadata['level'],
            'bitwidth': metadata['bitwidth'],
            'dependencies': metadata['dependencies']
        }
    
    def get(self, name):
        return self.modules[name]['module']
    
    def query(self, criteria):
        # 조건에 맞는 모듈 검색
        return [m for m in self.modules.values() 
                if meets_criteria(m, criteria)]
```

### 3. Composition Strategy (조합 전략)

**역할**: 선택된 모듈을 어떻게 조합할지 결정

**주요 패턴**:
- **Sequential**: 순차적 실행
- **Parallel**: 병렬 실행 후 융합
- **Hierarchical**: 계층적 조합
- **DAG (Directed Acyclic Graph)**: 의존성 그래프

## 인지 시드 코어 설계 시사점

### 핵심 컴포넌트

본 서베이를 기반으로 인지 시드 코어는 다음 컴포넌트 필요:

#### 1. Seed Router (시드 라우터)

**기능**: 입력과 태스크에 따라 적절한 시드 선택

**설계 근거**: Sample-wise Dynamic Routing

**구현 요소**:
- Task Analyzer: 태스크 요구사항 분석
- Seed Selector: 게이팅 함수로 시드 선택
- Efficiency Optimizer: 계산 비용 최소화

#### 2. Seed Registry (시드 레지스트리)

**기능**: 32개 시드 등록 및 메타데이터 관리

**메타데이터 항목**:
- Level (0-3): Atomic, Molecular, Cellular, Tissue
- Bitwidth: INT8, FP8, FP16, BF16
- Dependencies: 의존하는 하위 시드
- Geometry: E/H/S (Euclidean/Hyperbolic/Spherical)
- Scale Range: 처리 가능한 스케일 범위

#### 3. Composition Engine (조합 엔진)

**기능**: 선택된 시드들을 조합하여 실행

**조합 패턴**:
- **Bottom-up**: L0 → L1 → L2 → L3 순차 조합
- **Top-down**: 고수준 시드가 하위 시드 호출
- **Hybrid**: 태스크에 따라 동적 조합

#### 4. Cache Manager (캐시 관리자)

**기능**: 중간 결과 캐싱으로 효율성 향상

**캐싱 전략**:
- Seed-level caching: 시드 출력 캐싱
- Feature-level caching: 중간 특징 캐싱
- Result-level caching: 최종 결과 캐싱

## 학습 및 최적화 전략

### 1. End-to-End Training

**장점**: 모든 컴포넌트 동시 최적화
**단점**: 계산 비용 높음, 수렴 어려움

### 2. Modular Training

**장점**: 각 시드 독립적 학습 가능
**단점**: 조합 시 최적성 보장 어려움

### 3. Hybrid Training (권장)

**Phase 1**: 각 시드 독립적 사전학습
**Phase 2**: Router와 Composition Engine 학습
**Phase 3**: End-to-end 미세조정

## 효율성 최적화 기법

### 1. Early Exiting

**적용**: Tissue 레벨 시드에서 조기 종료
- 쉬운 태스크: Cellular 레벨에서 종료
- 어려운 태스크: Tissue 레벨까지 실행

### 2. Layer Skipping

**적용**: 불필요한 시드 건너뛰기
- 태스크 특성 분석 후 필수 시드만 실행

### 3. Dynamic Bitwidth

**적용**: 시드별 동적 양자화
- 중요 시드: FP16
- 일반 시드: INT8/FP8

## 벤치마크 및 평가

### 평가 지표

1. **Accuracy**: 태스크 정확도
2. **Efficiency**: FLOPs, 메모리, 지연시간
3. **Adaptiveness**: 다양한 입력에 대한 강건성
4. **Interpretability**: 시드 선택의 해석 가능성

### 비교 대상

- **Static Baseline**: 모든 시드 항상 실행
- **Oracle**: 최적 시드 조합 (상한선)
- **Random**: 무작위 시드 선택 (하한선)

## 오픈 문제 및 향후 연구

### 1. Architecture Search

**문제**: 최적 시드 조합 자동 탐색
**방향**: NAS (Neural Architecture Search) 기법 적용

### 2. Decision Making

**문제**: 게이팅 함수 학습 어려움
**방향**: Meta-learning, Few-shot learning

### 3. Optimization

**문제**: 동적 네트워크 최적화 복잡
**방향**: 효율적 학습 알고리즘 개발

### 4. Applications

**문제**: 실제 태스크 적용 검증 부족
**방향**: 다양한 도메인 벤치마크 구축

## 구현 참고 자료

### 오픈소스 프레임워크

1. **PyTorch Dynamic Networks**: 동적 그래프 지원
2. **TensorFlow Eager Execution**: 동적 실행
3. **JAX**: 함수형 동적 네트워크

### 주요 라이브러리

- **torch.nn.ModuleDict**: 모듈 딕셔너리
- **torch.nn.ModuleList**: 모듈 리스트
- **torch.jit**: 동적 컴파일

## 시드 코어 아키텍처 권장사항

### 계층 구조

```
CognitiveSeedCore
├── SeedRouter (게이팅 네트워크)
├── SeedRegistry (32개 시드 등록)
├── CompositionEngine (조합 실행)
├── CacheManager (중간 결과 캐싱)
└── MetricsCollector (성능 모니터링)
```

### 실행 흐름

1. **입력 분석**: Task Analyzer가 태스크 요구사항 파악
2. **시드 선택**: Seed Router가 필요한 시드 선택
3. **의존성 해결**: 하위 시드 자동 포함
4. **실행 계획**: Composition Engine이 실행 순서 결정
5. **캐시 활용**: 이전 결과 재사용
6. **결과 반환**: 최종 출력 생성

### 확장성 고려

- **플러그인 방식**: 새 시드 동적 추가
- **버전 관리**: 시드 버전별 호환성
- **분산 실행**: 여러 디바이스에 시드 분산

---

**Note**: 이 서베이는 1018회 인용된 권위 있는 자료로, 인지 시드 코어의 동적 라우팅 메커니즘 설계에 핵심 이론적 기반을 제공함.

