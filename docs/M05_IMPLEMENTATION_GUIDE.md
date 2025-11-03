# M05: Concept Crystallizer - 구현 가이드

## 문서 분류

이 문서는 **구현 가이드**입니다.
- 📚 **정보 자료**: `M05_RESEARCH_MATERIALS.md`
- 📖 **구현 가이드**: 본 문서 (M05_IMPLEMENTATION_GUIDE.md)
- 💻 **프로젝트 코드**: `seeds/molecular/m05_concept_crystallizer.py`
- 🧪 **활용 예제**: `examples/m05_usage_examples.py`

---

## 목차

1. [개요](#1-개요)
2. [설계 명세](#2-설계-명세)
3. [단계별 구현 가이드](#3-단계별-구현-가이드)
4. [테스트 전략](#4-테스트-전략)
5. [성능 최적화](#5-성능-최적화)
6. [참고 자료](#6-참고-자료)

---

## 1. 개요

### 1.1 기본 정보

- **시드 ID**: SEED-M05
- **이름**: Concept Crystallizer
- **Level**: 1 (Molecular)
- **Category**: Abstraction
- **Target Params**: ~700K
- **Bit Depth**: FP8

### 1.2 목적

소수의 예제로부터 **개념의 프로토타입 표현을 학습**하고, 새로운 인스턴스를 해당 개념으로 분류합니다. Few-shot learning을 통해 데이터가 제한된 환경에서도 강력한 일반화 능력을 발휘합니다.

### 1.3 구성 시드

- **A05**: Grouping Nucleus (군집화 및 그룹 표현)
- **M03**: Pattern Completer (패턴 완성 및 보간)
- **M01**: Hierarchy Builder (계층적 구조 학습)

### 1.4 핵심 기능

1. **Prototype Learning**
   - Few-shot 예제로부터 클래스 프로토타입 학습
   - Embedding space에서 거리 기반 분류

2. **Concept Abstraction**
   - 계층적 개념 표현 (M01)
   - 패턴 기반 일반화 (M03)

3. **Meta-Learning**
   - Episode-based 학습
   - Support set과 Query set 구분

4. **Distance-based Classification**
   - Euclidean distance 또는 Cosine similarity
   - Softmax over distances

---

## 2. 설계 명세

### 2.1 아키텍처 다이어그램

```
Support Set [N, K, D]    Query Set [Q, D]
    │                         │
    ├─────────────────────────┤
    │                         │
    ▼                         ▼
┌─────────────────┐   ┌─────────────────┐
│  Embedding Net  │   │  Embedding Net  │
│  (공유)         │   │  (공유)         │
│  - A05 (그룹)   │   │  - A05 (그룹)   │
│  - M03 (패턴)   │   │  - M03 (패턴)   │
│  - M01 (계층)   │   │  - M01 (계층)   │
└─────────────────┘   └─────────────────┘
    │                         │
    ▼                         │
┌─────────────────┐           │
│ Prototype       │           │
│ Computation     │           │
│ (mean pooling)  │           │
└─────────────────┘           │
    │                         │
    └──────────┬──────────────┘
               ▼
       ┌───────────────┐
       │ Distance      │
       │ Computation   │
       │ (Euclidean)   │
       └───────────────┘
               ▼
       ┌───────────────┐
       │ Classification│
       │ (Softmax)     │
       └───────────────┘
               ▼
       Output [Q, N]
```

### 2.2 입출력 명세

#### 입력 (Episode 기반)
- `support_set`: `[N, K, D]` - N개 클래스, 각 K개 예제
- `query_set`: `[Q, D]` - Q개 쿼리 샘플
- `N`: N-way (클래스 수)
- `K`: K-shot (클래스당 예제 수)

#### 출력
- `logits`: `[Q, N]` - 각 쿼리의 클래스별 로짓
- `predictions`: `[Q]` - 예측된 클래스 인덱스

#### 메타데이터
```python
{
    'prototypes': Tensor,         # [N, D] - 각 클래스의 프로토타입
    'embeddings': Tensor,         # [N*K+Q, D] - 모든 임베딩
    'distances': Tensor,          # [Q, N] - 쿼리-프로토타입 거리
    'support_embeddings': Tensor, # [N, K, D]
    'query_embeddings': Tensor    # [Q, D]
}
```

### 2.3 파라미터 예산

| 컴포넌트 | 파라미터 수 | 비율 |
|---------|-----------|------|
| A05 (Grouping Nucleus) | ~100K | 14% |
| M03 (Pattern Completer) | ~550K | 79% |
| M01 (Hierarchy Builder) | ~426K | 61% |
| **기존 시드 합계** | **~1,076K** | **154%** |
| Shared Embedding Layers | -376K | -54% |
| Distance Metric Layer | ~0.5K | 0.1% |
| **실제 총합** | **~700K** | **100%** |

**참고**: 시드들이 공유 임베딩 레이어를 사용하여 중복 제거

---

## 3. 단계별 구현 가이드

### Step 1: 프로젝트 구조 준비

#### 1.1 파일 생성

```bash
# 메인 구현 파일
touch seeds/molecular/m05_concept_crystallizer.py

# 활용 예제 파일
mkdir -p examples
touch examples/m05_usage_examples.py

# 테스트 파일
mkdir -p tests/molecular
touch tests/molecular/test_m05_concept_crystallizer.py
```

#### 1.2 기본 클래스 구조

```python
# seeds/molecular/m05_concept_crystallizer.py
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from ..atomic.a05_grouping_nucleus import GroupingNucleus
from .m03_pattern_completer import PatternCompleter
from .m01_hierarchy_builder import HierarchyBuilder

class ConceptCrystallizer(nn.Module):
    """
    M05: Concept Crystallizer
    
    Few-shot learning을 통해 개념의 프로토타입을 학습하고
    새로운 인스턴스를 분류합니다.
    
    Args:
        input_dim: 입력 차원
        hidden_dim: 은닉 차원
        n_way: N-way classification (클래스 수)
        k_shot: K-shot learning (클래스당 예제 수)
        distance_metric: 'euclidean' or 'cosine'
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        n_way: int = 5,
        k_shot: int = 5,
        distance_metric: str = 'euclidean'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        self.k_shot = k_shot
        self.distance_metric = distance_metric
        
        # 구성 시드들
        self.grouping = GroupingNucleus(input_dim, hidden_dim)
        self.pattern_completer = PatternCompleter(hidden_dim)
        self.hierarchy = HierarchyBuilder(hidden_dim)
        
        # 임베딩 네트워크 (공유)
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def compute_prototypes(
        self, 
        support_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Support set으로부터 프로토타입 계산
        
        Args:
            support_embeddings: [N, K, D]
        
        Returns:
            prototypes: [N, D]
        """
        # 각 클래스의 평균 임베딩을 프로토타입으로 사용
        prototypes = support_embeddings.mean(dim=1)
        return prototypes
    
    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        쿼리와 프로토타입 간 거리 계산
        
        Args:
            query_embeddings: [Q, D]
            prototypes: [N, D]
        
        Returns:
            distances: [Q, N]
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance
            # [Q, 1, D] - [1, N, D] -> [Q, N, D]
            diff = query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
            distances = torch.norm(diff, dim=2)
        elif self.distance_metric == 'cosine':
            # Cosine similarity (음수로 변환하여 거리처럼 사용)
            query_norm = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
            proto_norm = prototypes / prototypes.norm(dim=1, keepdim=True)
            distances = -torch.mm(query_norm, proto_norm.t())
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(
        self,
        support_set: torch.Tensor,
        query_set: torch.Tensor,
        return_metadata: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass
        
        Args:
            support_set: [N, K, D] - Support set
            query_set: [Q, D] - Query set
            return_metadata: 메타데이터 반환 여부
        
        Returns:
            logits: [Q, N] - 클래스별 로짓
            metadata: 메타데이터 (선택적)
        """
        N, K, D = support_set.shape
        Q = query_set.shape[0]
        
        # 1. Support set 임베딩
        # [N, K, D] -> [N*K, D]
        support_flat = support_set.view(N * K, D)
        support_emb = self.embedding_net(support_flat)
        
        # 구성 시드 적용
        support_emb = self.grouping(support_emb.unsqueeze(0)).squeeze(0)
        support_emb = self.pattern_completer(support_emb.unsqueeze(0)).squeeze(0)
        support_emb = self.hierarchy(support_emb.unsqueeze(0)).squeeze(0)
        
        # [N*K, D] -> [N, K, D]
        support_embeddings = support_emb.view(N, K, -1)
        
        # 2. Query set 임베딩
        query_emb = self.embedding_net(query_set)
        query_emb = self.grouping(query_emb.unsqueeze(0)).squeeze(0)
        query_emb = self.pattern_completer(query_emb.unsqueeze(0)).squeeze(0)
        query_emb = self.hierarchy(query_emb.unsqueeze(0)).squeeze(0)
        query_embeddings = query_emb
        
        # 3. 프로토타입 계산
        prototypes = self.compute_prototypes(support_embeddings)
        
        # 4. 거리 계산
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # 5. 로짓 계산 (거리의 음수를 로짓으로 사용)
        logits = -distances
        
        # 6. 예측
        predictions = torch.argmax(logits, dim=1)
        
        if return_metadata:
            metadata = {
                'prototypes': prototypes,
                'embeddings': torch.cat([support_emb, query_emb], dim=0),
                'distances': distances,
                'support_embeddings': support_embeddings,
                'query_embeddings': query_embeddings,
                'predictions': predictions
            }
            return logits, metadata
        
        return logits, None
```

### Step 2: 테스트 코드 작성

```python
# tests/molecular/test_m05_concept_crystallizer.py
import torch
import pytest
from seeds.molecular.m05_concept_crystallizer import ConceptCrystallizer

def test_concept_crystallizer_basic():
    """기본 동작 테스트"""
    model = ConceptCrystallizer(
        input_dim=64,
        hidden_dim=128,
        n_way=5,
        k_shot=5
    )
    
    # 5-way 5-shot
    support_set = torch.randn(5, 5, 64)
    query_set = torch.randn(10, 64)
    
    logits, metadata = model(support_set, query_set, return_metadata=True)
    
    assert logits.shape == (10, 5)
    assert metadata['prototypes'].shape == (5, 128)
    assert metadata['predictions'].shape == (10,)

def test_few_shot_learning():
    """Few-shot 학습 시뮬레이션"""
    model = ConceptCrystallizer(n_way=3, k_shot=3)
    
    # 간단한 합성 데이터
    # 클래스 0: [1, 0, 0, ...]
    # 클래스 1: [0, 1, 0, ...]
    # 클래스 2: [0, 0, 1, ...]
    support_set = torch.zeros(3, 3, 64)
    for i in range(3):
        support_set[i, :, i] = 1.0
    
    query_set = torch.zeros(3, 64)
    query_set[0, 0] = 1.0  # 클래스 0
    query_set[1, 1] = 1.0  # 클래스 1
    query_set[2, 2] = 1.0  # 클래스 2
    
    logits, metadata = model(support_set, query_set, return_metadata=True)
    predictions = metadata['predictions']
    
    # 정확히 분류되어야 함
    assert predictions[0] == 0
    assert predictions[1] == 1
    assert predictions[2] == 2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### Step 3: 활용 예제 작성

```python
# examples/m05_usage_examples.py
import torch
from seeds.molecular.m05_concept_crystallizer import ConceptCrystallizer

def example_omniglot_style():
    """Omniglot 스타일 문자 인식"""
    print("=== Omniglot-style Character Recognition ===")
    
    model = ConceptCrystallizer(
        input_dim=784,  # 28x28 이미지
        hidden_dim=256,
        n_way=5,
        k_shot=1  # 1-shot learning
    )
    
    # 5개 클래스, 각 1개 예제
    support_set = torch.randn(5, 1, 784)
    query_set = torch.randn(20, 784)
    
    logits, metadata = model(support_set, query_set, return_metadata=True)
    
    print(f"Support set: {support_set.shape}")
    print(f"Query set: {query_set.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Predictions: {metadata['predictions']}")
    print(f"Prototypes: {metadata['prototypes'].shape}")

def example_concept_learning():
    """개념 학습 시뮬레이션"""
    print("\n=== Concept Learning Simulation ===")
    
    model = ConceptCrystallizer(n_way=3, k_shot=5)
    
    # 3개 개념, 각 5개 예제
    support_set = torch.randn(3, 5, 64)
    query_set = torch.randn(15, 64)
    
    logits, metadata = model(support_set, query_set, return_metadata=True)
    
    # 각 쿼리의 가장 가까운 프로토타입 확인
    distances = metadata['distances']
    min_distances, predictions = torch.min(distances, dim=1)
    
    print(f"Query predictions: {predictions}")
    print(f"Min distances: {min_distances}")

if __name__ == '__main__':
    example_omniglot_style()
    example_concept_learning()
```

---

## 4. 테스트 전략

### 4.1 단위 테스트

1. **프로토타입 계산 테스트**
   - Support set으로부터 정확한 평균 계산 확인

2. **거리 계산 테스트**
   - Euclidean 및 Cosine 거리 정확성 검증

3. **Few-shot 분류 테스트**
   - 간단한 합성 데이터로 정확한 분류 확인

### 4.2 통합 테스트

1. **N-way K-shot 변형 테스트**
   - 다양한 N, K 조합에서 동작 확인

2. **메타러닝 시뮬레이션**
   - Episode 기반 학습 프로세스 검증

### 4.3 벤치마크

- **Omniglot**: 1-shot, 5-way 분류
- **Mini-ImageNet**: 5-shot, 5-way 분류
- **목표 정확도**: ≥ 85%

---

## 5. 성능 최적화

### 5.1 파라미터 효율성

- 공유 임베딩 레이어 사용
- 경량 거리 계산 모듈

### 5.2 FP8 양자화

```python
# FP8 양자화 적용
model = ConceptCrystallizer(...)
model = model.to(torch.float8_e4m3fn)
```

### 5.3 배치 처리

- Episode 단위 배치 처리
- 병렬 프로토타입 계산

---

## 6. 참고 자료

1. **Prototypical Networks** (Snell et al., 2017)
   - https://arxiv.org/abs/1703.05175

2. **Meta-Learning Survey** (Hospedales et al., 2021)
   - https://ieeexplore.ieee.org/document/9428530

3. **Few-Shot Learning Benchmark**
   - Omniglot, Mini-ImageNet 데이터셋

---

**작성일**: 2025-11-02  
**작성자**: Manus AI  
**버전**: 1.0
