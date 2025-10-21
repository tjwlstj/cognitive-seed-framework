# M06 Context Integrator 구현 가이드

## 문서 분류

이 문서는 **구현 가이드**입니다.
- 📚 **정보 자료**: `M06_RESEARCH_MATERIALS.md`
- 📖 **구현 가이드**: 본 문서 (M06_IMPLEMENTATION_GUIDE.md)
- 💻 **프로젝트 코드**: `seeds/molecular/m06_context_integrator.py`
- 🧪 **활용 예제**: `examples/m06_usage_examples.py`

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

- **시드 ID**: SEED-M06
- **이름**: Context Integrator
- **Level**: 1 (Molecular)
- **Category**: Composition
- **Target Params**: ~650K
- **Bit Depth**: FP8

### 1.2 목적

국소적 맥락(local context)과 전역적 맥락(global context)을 융합하여 **중의성을 해소**하고 이해를 향상시킵니다.

### 1.3 구성 시드

- **A06**: Sequence Tracker (시간적 맥락)
- **M01**: Hierarchy Builder (계층적 맥락)
- **A05**: Grouping Nucleus (그룹 맥락)

### 1.4 핵심 기능

1. **Multi-scale Context Encoding**
   - Local context: 슬라이딩 윈도우 기반
   - Global context: 전체 시퀀스 기반

2. **Hierarchical Context Integration**
   - Temporal context (A06)
   - Hierarchical context (M01)
   - Group context (A05)

3. **Context Fusion**
   - Multi-head attention 기반
   - Cross-attention mechanism

4. **Disambiguation**
   - 맥락 기반 중의성 해소
   - 상호작용 특징 활용

---

## 2. 설계 명세

### 2.1 아키텍처 다이어그램

```
Input [B, L, D]
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌─────────────────┐         ┌──────────────────┐
│  Atomic Seeds   │         │  Local/Global    │
│  (병렬 처리)     │         │  Context Encoder │
│  - A06 (시간)   │         │  - Transformer   │
│  - M01 (계층)   │         │  - Sliding Window│
│  - A05 (그룹)   │         └──────────────────┘
└─────────────────┘                 │
    │                               │
    └───────────┬───────────────────┘
                ▼
        ┌───────────────┐
        │ Context Fusion│
        │ (Multi-head   │
        │  Attention)   │
        └───────────────┘
                ▼
        ┌───────────────┐
        │ Disambiguator │
        │ (중의성 해소) │
        └───────────────┘
                ▼
        Output [B, L, D]
```

### 2.2 입출력 명세

#### 입력
- `x`: `[B, L, D]` - 입력 시퀀스
- `context_window`: `int` - 국소 맥락 윈도우 크기 (기본값: 5)

#### 출력
- `integrated`: `[B, L, D]` - 맥락이 통합된 표현

#### 메타데이터
```python
{
    'local_context': Tensor,      # [B, L, D]
    'global_context': Tensor,     # [B, L, D]
    'temporal_context': Tensor,   # [B, L, D]
    'hierarchical_context': Tensor,  # [B, L, D]
    'group_context': Tensor,      # [B, L, D]
    'fusion_weights': Tensor      # [B, L, num_contexts]
}
```

### 2.3 파라미터 예산

| 컴포넌트 | 파라미터 수 | 비율 |
|---------|-----------|------|
| A06 (Sequence Tracker) | ~120K | 18% |
| M01 (Hierarchy Builder) | ~426K | 66% |
| A05 (Grouping Nucleus) | ~100K | 15% |
| **기존 시드 합계** | **~646K** | **99%** |
| Local Context Encoder | ~3K | 0.5% |
| Global Context Encoder | ~3K | 0.5% |
| Context Fusion (MHA) | ~1K | 0.2% |
| Disambiguator | ~0.5K | 0.1% |
| **추가 레이어 합계** | **~7.5K** | **1%** |
| **총합** | **~653K** | **100%** |

---

## 3. 단계별 구현 가이드

### Step 1: 프로젝트 구조 준비

#### 1.1 파일 생성

```bash
# 메인 구현 파일
touch seeds/molecular/m06_context_integrator.py

# 활용 예제 파일
mkdir -p examples
touch examples/m06_usage_examples.py

# 테스트 파일 (기존 파일에 추가)
# tests/test_molecular_seeds.py
```

#### 1.2 디렉토리 구조

```
cognitive-seed-framework/
├── seeds/
│   └── molecular/
│       ├── __init__.py
│       ├── m01_hierarchy_builder.py
│       ├── m02_causality_detector.py
│       ├── m03_pattern_completer.py
│       ├── m04_spatial_transformer.py
│       └── m06_context_integrator.py  ← 신규
├── examples/
│   └── m06_usage_examples.py          ← 신규
├── tests/
│   └── test_molecular_seeds.py        ← 업데이트
└── docs/
    ├── M06_RESEARCH_MATERIALS.md      ← 정보 자료
    └── M06_IMPLEMENTATION_GUIDE.md    ← 본 문서
```

---

### Step 2: Config 클래스 작성

```python
# seeds/molecular/m06_context_integrator.py

from dataclasses import dataclass
from seeds.base import BaseSeed, SeedConfig

@dataclass
class ContextIntegratorConfig(SeedConfig):
    """Context Integrator 설정"""
    seed_id: str = "SEED-M06"
    name: str = "Context Integrator"
    level: int = 1
    category: str = "Composition"
    bit_depth: str = "FP8"
    params: int = 650000
    input_dim: int = 128
    output_dim: int = 128
    
    # M06 특화 설정
    num_heads: int = 8
    num_encoder_layers: int = 2
    context_window: int = 5
    dropout: float = 0.1
```

**체크포인트 1**: ✅ Config 클래스가 SeedConfig를 상속하는가?

---

### Step 3: 기본 클래스 구조 작성

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from seeds.atomic import SequenceTracker, GroupingNucleus
from seeds.molecular import HierarchyBuilder

class ContextIntegrator(BaseSeed):
    """
    SEED-M06: Context Integrator
    
    다층적 맥락을 통합하여 중의성을 해소합니다.
    
    주요 기능:
    - Multi-scale context encoding (local/global)
    - Hierarchical context integration
    - Multi-head attention fusion
    - Context-based disambiguation
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 2,
        context_window: int = 5,
        dropout: float = 0.1
    ):
        config = ContextIntegratorConfig(
            input_dim=input_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            context_window=context_window,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        
        # Step 4에서 구현할 컴포넌트들
        self._init_atomic_seeds()
        self._init_context_encoders()
        self._init_fusion_module()
        self._init_disambiguator()
```

**체크포인트 2**: ✅ BaseSeed를 상속하고 config를 전달하는가?

---

### Step 4: Atomic Seeds 초기화

```python
def _init_atomic_seeds(self):
    """Atomic/Molecular seeds 초기화"""
    
    # A06: Sequence Tracker (시간적 맥락)
    self.sequence_tracker = SequenceTracker(self.config.input_dim)
    
    # M01: Hierarchy Builder (계층적 맥락)
    self.hierarchy_builder = HierarchyBuilder(self.config.input_dim)
    
    # A05: Grouping Nucleus (그룹 맥락)
    self.grouping_nucleus = GroupingNucleus(self.config.input_dim)
```

**체크포인트 3**: ✅ 모든 구성 시드가 올바르게 import되고 초기화되는가?

---

### Step 5: Context Encoders 구현

#### 5.1 Local Context Encoder

```python
def _init_context_encoders(self):
    """Local/Global context encoders 초기화"""
    
    # Local context encoder (Transformer)
    local_encoder_layer = nn.TransformerEncoderLayer(
        d_model=self.config.input_dim,
        nhead=self.config.num_heads,
        dim_feedforward=self.config.input_dim * 4,
        dropout=self.config.dropout,
        batch_first=True
    )
    self.local_context_encoder = nn.TransformerEncoder(
        local_encoder_layer,
        num_layers=self.config.num_encoder_layers
    )
    
    # Global context encoder (Transformer)
    global_encoder_layer = nn.TransformerEncoderLayer(
        d_model=self.config.input_dim,
        nhead=self.config.num_heads,
        dim_feedforward=self.config.input_dim * 4,
        dropout=self.config.dropout,
        batch_first=True
    )
    self.global_context_encoder = nn.TransformerEncoder(
        global_encoder_layer,
        num_layers=self.config.num_encoder_layers
    )

def encode_local_context(
    self,
    x: torch.Tensor,
    window_size: Optional[int] = None
) -> torch.Tensor:
    """
    슬라이딩 윈도우 기반 국소 맥락 인코딩
    
    Args:
        x: [B, L, D] - 입력 시퀀스
        window_size: 윈도우 크기 (None이면 config 값 사용)
    Returns:
        local_context: [B, L, D] - 국소 맥락
    """
    if window_size is None:
        window_size = self.config.context_window
    
    B, L, D = x.shape
    
    # 패딩 (양쪽에 window_size // 2씩)
    pad_size = window_size // 2
    padded = F.pad(x, (0, 0, pad_size, pad_size))  # [B, L+2*pad_size, D]
    
    # 슬라이딩 윈도우 추출
    local_contexts = []
    for i in range(L):
        window = padded[:, i:i+window_size, :]  # [B, window_size, D]
        local_contexts.append(window)
    
    local_contexts = torch.stack(local_contexts, dim=1)  # [B, L, window_size, D]
    
    # Transformer 인코딩 (각 윈도우 독립적으로)
    local_contexts_flat = local_contexts.view(B * L, window_size, D)
    encoded = self.local_context_encoder(local_contexts_flat)  # [B*L, window_size, D]
    
    # 평균 풀링으로 윈도우 요약
    local_context = encoded.mean(dim=1).view(B, L, D)  # [B, L, D]
    
    return local_context

def encode_global_context(self, x: torch.Tensor) -> torch.Tensor:
    """
    전체 시퀀스를 고려한 전역 맥락 인코딩
    
    Args:
        x: [B, L, D] - 입력 시퀀스
    Returns:
        global_context: [B, L, D] - 전역 맥락
    """
    # Transformer 인코딩 (전체 시퀀스)
    global_context = self.global_context_encoder(x)  # [B, L, D]
    
    return global_context
```

**체크포인트 4**: ✅ Local context가 슬라이딩 윈도우로 올바르게 추출되는가?

---

### Step 6: Context Fusion Module 구현

```python
def _init_fusion_module(self):
    """Context fusion module 초기화"""
    
    # Multi-head attention for fusion
    self.context_fusion = nn.MultiheadAttention(
        embed_dim=self.config.input_dim,
        num_heads=self.config.num_heads,
        dropout=self.config.dropout,
        batch_first=True
    )
    
    # Context weighting network (선택적)
    self.context_weighter = nn.Sequential(
        nn.Linear(self.config.input_dim * 5, 256),
        nn.ReLU(),
        nn.Dropout(self.config.dropout),
        nn.Linear(256, 5),  # 5개 맥락에 대한 가중치
        nn.Softmax(dim=-1)
    )

def fuse_contexts(
    self,
    local: torch.Tensor,
    global_ctx: torch.Tensor,
    temporal: torch.Tensor,
    hierarchical: torch.Tensor,
    group: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    다중 맥락 융합
    
    Args:
        local: [B, L, D] - 국소 맥락
        global_ctx: [B, L, D] - 전역 맥락
        temporal: [B, L, D] - 시간적 맥락
        hierarchical: [B, L, D] - 계층적 맥락
        group: [B, L, D] - 그룹 맥락
    Returns:
        fused: [B, L, D] - 융합된 맥락
        weights: [B, L, 5] - 각 맥락의 가중치
    """
    B, L, D = local.shape
    
    # 방법 1: Cross-attention 기반 융합
    # local을 query, 나머지를 key/value로 사용
    contexts = torch.stack([global_ctx, temporal, hierarchical, group], dim=2)  # [B, L, 4, D]
    contexts_flat = contexts.reshape(B, L, -1)  # [B, L, 4*D]
    
    # Cross-attention
    fused, attn_weights = self.context_fusion(
        query=local,           # [B, L, D]
        key=contexts_flat,     # [B, L, 4*D]
        value=contexts_flat    # [B, L, 4*D]
    )
    
    # 방법 2: Weighted sum (추가 옵션)
    # 모든 맥락 결합
    all_contexts = torch.cat([local, global_ctx, temporal, hierarchical, group], dim=-1)  # [B, L, 5*D]
    
    # 가중치 계산
    weights = self.context_weighter(all_contexts)  # [B, L, 5]
    
    # 가중 합산
    contexts_stacked = torch.stack([local, global_ctx, temporal, hierarchical, group], dim=-1)  # [B, L, D, 5]
    weighted_fused = torch.sum(contexts_stacked * weights.unsqueeze(2), dim=-1)  # [B, L, D]
    
    # 두 방법 결합 (선택적)
    final_fused = (fused + weighted_fused) / 2
    
    return final_fused, weights
```

**체크포인트 5**: ✅ 다중 맥락이 올바르게 융합되는가?

---

### Step 7: Disambiguator 구현

```python
def _init_disambiguator(self):
    """Disambiguator 초기화"""
    
    self.disambiguator = nn.Sequential(
        nn.Linear(self.config.input_dim * 3, 256),
        nn.ReLU(),
        nn.Dropout(self.config.dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(self.config.dropout),
        nn.Linear(128, self.config.input_dim)
    )

def disambiguate(
    self,
    x: torch.Tensor,
    context: torch.Tensor
) -> torch.Tensor:
    """
    맥락을 활용한 중의성 해소
    
    Args:
        x: [B, L, D] - 원본 입력
        context: [B, L, D] - 융합된 맥락
    Returns:
        disambiguated: [B, L, D] - 중의성이 해소된 표현
    """
    # 원본, 맥락, 그리고 둘의 상호작용 결합
    interaction = x * context  # Element-wise multiplication
    combined = torch.cat([x, context, interaction], dim=-1)  # [B, L, 3*D]
    
    # Disambiguation network
    disambiguated = self.disambiguator(combined)  # [B, L, D]
    
    # Residual connection
    output = x + disambiguated
    
    return output
```

**체크포인트 6**: ✅ 상호작용 특징이 올바르게 계산되는가?

---

### Step 8: Forward Pass 구현

```python
def forward(
    self,
    x: torch.Tensor,
    context_window: Optional[int] = None,
    return_metadata: bool = False
) -> torch.Tensor:
    """
    Forward pass
    
    Args:
        x: [B, L, D] - 입력 시퀀스
        context_window: 국소 맥락 윈도우 크기
        return_metadata: 메타데이터 반환 여부
    Returns:
        integrated: [B, L, D] - 맥락이 통합된 표현
        (선택적) metadata: 중간 결과 딕셔너리
    """
    # 1. Atomic/Molecular seeds로 다양한 맥락 추출
    temporal_context = self.sequence_tracker(x)        # [B, L, D]
    hierarchical_context = self.hierarchy_builder(x)   # [B, L, D]
    group_context = self.grouping_nucleus(x)           # [B, L, D]
    
    # 2. Local/Global context encoding
    local_context = self.encode_local_context(x, context_window)  # [B, L, D]
    global_context = self.encode_global_context(x)                # [B, L, D]
    
    # 3. Context fusion
    fused_context, fusion_weights = self.fuse_contexts(
        local_context,
        global_context,
        temporal_context,
        hierarchical_context,
        group_context
    )  # [B, L, D], [B, L, 5]
    
    # 4. Disambiguation
    integrated = self.disambiguate(x, fused_context)  # [B, L, D]
    
    if return_metadata:
        metadata = {
            'local_context': local_context,
            'global_context': global_context,
            'temporal_context': temporal_context,
            'hierarchical_context': hierarchical_context,
            'group_context': group_context,
            'fused_context': fused_context,
            'fusion_weights': fusion_weights
        }
        return integrated, metadata
    
    return integrated
```

**체크포인트 7**: ✅ Forward pass가 모든 단계를 올바르게 실행하는가?

---

### Step 9: 추가 유틸리티 메서드

```python
def get_context_importance(
    self,
    x: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    각 맥락의 중요도 분석
    
    Args:
        x: [B, L, D]
    Returns:
        importance: 맥락별 중요도 딕셔너리
    """
    _, metadata = self.forward(x, return_metadata=True)
    
    fusion_weights = metadata['fusion_weights']  # [B, L, 5]
    
    # 평균 중요도 계산
    avg_importance = fusion_weights.mean(dim=[0, 1])  # [5]
    
    importance = {
        'local': avg_importance[0].item(),
        'global': avg_importance[1].item(),
        'temporal': avg_importance[2].item(),
        'hierarchical': avg_importance[3].item(),
        'group': avg_importance[4].item()
    }
    
    return importance

def visualize_context_attention(
    self,
    x: torch.Tensor,
    position: int
) -> Dict[str, torch.Tensor]:
    """
    특정 위치의 맥락 attention 시각화
    
    Args:
        x: [B, L, D]
        position: 분석할 위치
    Returns:
        attention_maps: Attention 맵 딕셔너리
    """
    _, metadata = self.forward(x, return_metadata=True)
    
    # 해당 위치의 각 맥락 추출
    attention_maps = {
        'local': metadata['local_context'][:, position, :],
        'global': metadata['global_context'][:, position, :],
        'temporal': metadata['temporal_context'][:, position, :],
        'hierarchical': metadata['hierarchical_context'][:, position, :],
        'group': metadata['group_context'][:, position, :]
    }
    
    return attention_maps
```

**체크포인트 8**: ✅ 유틸리티 메서드가 올바른 정보를 반환하는가?

---

### Step 10: __init__.py 업데이트

```python
# seeds/molecular/__init__.py

from .m01_hierarchy_builder import HierarchyBuilder, create_hierarchy_builder
from .m02_causality_detector import CausalityDetector, create_causality_detector
from .m03_pattern_completer import PatternCompleter
from .m04_spatial_transformer import SpatialTransformer, create_spatial_transformer
from .m06_context_integrator import ContextIntegrator  # 추가

__all__ = [
    "HierarchyBuilder",
    "create_hierarchy_builder",
    "CausalityDetector",
    "create_causality_detector",
    "PatternCompleter",
    "SpatialTransformer",
    "create_spatial_transformer",
    "ContextIntegrator",  # 추가
]
```

**체크포인트 9**: ✅ Import가 올바르게 추가되었는가?

---

## 4. 테스트 전략

### 4.1 단위 테스트 작성

```python
# tests/test_molecular_seeds.py

def test_context_integrator_forward(self):
    """M06: Forward pass 테스트"""
    print("\n[X/Y] Testing Context Integrator - Forward pass...")
    
    seed = ContextIntegrator(input_dim=self.input_dim)
    seed = seed.to(self.device)
    
    # 입력: [B, L, D]
    seq_len = 50
    x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
    
    # Forward
    output = seed(x)
    
    # 출력 shape 확인
    assert output.shape == (self.batch_size, seq_len, self.input_dim)
    assert not torch.isnan(output).any()
    
    print("✓ Forward pass successful")

def test_context_integrator_with_metadata(self):
    """M06: 메타데이터 반환 테스트"""
    print("\n[X/Y] Testing Context Integrator - Metadata...")
    
    seed = ContextIntegrator(input_dim=self.input_dim)
    seed = seed.to(self.device)
    
    seq_len = 50
    x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
    
    # Forward with metadata
    output, metadata = seed(x, return_metadata=True)
    
    # 메타데이터 확인
    assert 'local_context' in metadata
    assert 'global_context' in metadata
    assert 'temporal_context' in metadata
    assert 'hierarchical_context' in metadata
    assert 'group_context' in metadata
    assert 'fusion_weights' in metadata
    
    # Fusion weights shape 확인
    assert metadata['fusion_weights'].shape == (self.batch_size, seq_len, 5)
    
    print("✓ Metadata return successful")

def test_context_integrator_importance(self):
    """M06: 맥락 중요도 분석 테스트"""
    print("\n[X/Y] Testing Context Integrator - Context importance...")
    
    seed = ContextIntegrator(input_dim=self.input_dim)
    seed = seed.to(self.device)
    
    seq_len = 50
    x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
    
    # 중요도 분석
    importance = seed.get_context_importance(x)
    
    # 결과 확인
    assert 'local' in importance
    assert 'global' in importance
    assert 'temporal' in importance
    assert 'hierarchical' in importance
    assert 'group' in importance
    
    # 중요도 합이 1에 가까운지 확인
    total = sum(importance.values())
    assert abs(total - 1.0) < 0.01
    
    print("✓ Context importance analysis successful")
```

### 4.2 통합 테스트

```python
def test_context_integrator_integration(self):
    """M06: 다른 시드와의 통합 테스트"""
    print("\n[X/Y] Testing Context Integrator - Integration...")
    
    # M06 생성
    integrator = ContextIntegrator(input_dim=self.input_dim).to(self.device)
    
    # M03 (Pattern Completer)와 연계
    completer = PatternCompleter(input_dim=self.input_dim).to(self.device)
    
    seq_len = 50
    x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
    
    # M03 -> M06 파이프라인
    completed = completer(x)
    integrated = integrator(completed)
    
    # 결과 확인
    assert integrated.shape == (self.batch_size, seq_len, self.input_dim)
    assert not torch.isnan(integrated).any()
    
    print("✓ Integration with other seeds successful")
```

**체크포인트 10**: ✅ 모든 테스트가 통과하는가?

---

## 5. 성능 최적화

### 5.1 메모리 최적화

```python
# Gradient checkpointing 적용
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    """메모리 효율적인 forward pass"""
    
    # Checkpoint를 사용하여 중간 활성화 저장 최소화
    temporal_context = checkpoint(self.sequence_tracker, x)
    hierarchical_context = checkpoint(self.hierarchy_builder, x)
    group_context = checkpoint(self.grouping_nucleus, x)
    
    # ... 나머지 동일
```

### 5.2 계산 효율화

```python
# Cached context encoding
@torch.no_grad()
def precompute_global_context(self, x):
    """전역 맥락 사전 계산 (추론 시)"""
    return self.encode_global_context(x)

# Batch processing
def forward_batch_efficient(self, x, batch_size=32):
    """대용량 시퀀스 배치 처리"""
    B, L, D = x.shape
    
    if L > batch_size:
        # 시퀀스를 청크로 분할
        chunks = torch.split(x, batch_size, dim=1)
        results = []
        
        for chunk in chunks:
            result = self.forward(chunk)
            results.append(result)
        
        return torch.cat(results, dim=1)
    else:
        return self.forward(x)
```

### 5.3 양자화 준비

```python
# FP8 양자화를 위한 준비
def prepare_for_quantization(self):
    """양자화 준비"""
    
    # Batch normalization 추가 (선택적)
    self.bn_local = nn.BatchNorm1d(self.config.input_dim)
    self.bn_global = nn.BatchNorm1d(self.config.input_dim)
    
    # Quantization-aware training 설정
    # torch.quantization.prepare_qat(self, inplace=True)
```

---

## 6. 참고 자료

### 6.1 관련 문서

- **정보 자료**: `docs/M06_RESEARCH_MATERIALS.md`
- **설계 가이드**: `docs/LEVEL1_IMPLEMENTATION_GUIDE.md`
- **활용 예제**: `examples/m06_usage_examples.py`

### 6.2 핵심 논문

1. Jin et al., "MoH: Multi-Head Attention as Mixture-of-Head Attention", ICML 2025
2. Yang et al., "Context aware hierarchical attention", Nature 2025
3. Xu et al., "HCF-Net: Hierarchical Context Fusion Network", arXiv 2024

### 6.3 구현 참고

- PyTorch Transformer: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
- Multi-head Attention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

---

## 체크리스트

구현 완료 전 확인 사항:

- [ ] Step 1: 프로젝트 구조 준비
- [ ] Step 2: Config 클래스 작성
- [ ] Step 3: 기본 클래스 구조 작성
- [ ] Step 4: Atomic Seeds 초기화
- [ ] Step 5: Context Encoders 구현
- [ ] Step 6: Context Fusion Module 구현
- [ ] Step 7: Disambiguator 구현
- [ ] Step 8: Forward Pass 구현
- [ ] Step 9: 추가 유틸리티 메서드
- [ ] Step 10: __init__.py 업데이트
- [ ] 테스트 작성 및 통과
- [ ] 문서 작성 (README 업데이트)
- [ ] 활용 예제 작성

---

**작성일**: 2025-10-21  
**작성자**: Manus AI (누스양)  
**다음 업데이트**: M06 구현 완료 시

