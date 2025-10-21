# M06 Context Integrator - 프로젝트 통합 가이드

## 문서 분류

이 문서는 **프로젝트 통합 가이드**입니다.
- 📚 **정보 자료**: `M06_RESEARCH_MATERIALS.md`
- 📖 **구현 가이드**: `M06_IMPLEMENTATION_GUIDE.md`
- 🔗 **프로젝트 통합**: 본 문서 (M06_PROJECT_INTEGRATION.md)
- 💻 **메인 코드**: `seeds/molecular/m06_context_integrator.py`
- 🧪 **활용 예제**: `examples/m06_usage_examples.py`

---

## 목차

1. [통합 개요](#1-통합-개요)
2. [메인 프로그램 구조](#2-메인-프로그램-구조)
3. [다른 시드와의 연계](#3-다른-시드와의-연계)
4. [상위 레벨 시드 준비](#4-상위-레벨-시드-준비)
5. [API 설계](#5-api-설계)
6. [배포 전략](#6-배포-전략)

---

## 1. 통합 개요

### 1.1 M06의 역할

M06 Context Integrator는 **Level 1 (Molecular)의 핵심 통합 시드**로서:

1. **Phase 2 완료**: M03과 함께 Phase 2를 완료
2. **상위 시드 기반**: M08 (Conflict Resolver)의 구성 요소
3. **범용 맥락 통합**: 다양한 시드에서 재사용 가능

### 1.2 의존성 그래프

```
Level 0 (Atomic)
    A05 (Grouping Nucleus) ────┐
    A06 (Sequence Tracker) ─────┼──→ M06 (Context Integrator)
                                │
Level 1 (Molecular)             │
    M01 (Hierarchy Builder) ────┘
                                ↓
                            M08 (Conflict Resolver)
                                ↓
Level 2 (Cellular)
    (미정의)
```

### 1.3 통합 시점

- **현재 Phase**: Phase 2
- **M06 구현 후**: Phase 2 완료
- **다음 단계**: Phase 3 (M05, M07) 또는 Phase 4 (M08)

---

## 2. 메인 프로그램 구조

### 2.1 프로젝트 파일 구조

```
cognitive-seed-framework/
├── seeds/
│   ├── base.py                          # 기본 클래스
│   ├── atomic/                          # Level 0
│   │   ├── __init__.py
│   │   ├── a01_edge_detector.py
│   │   ├── a05_grouping_nucleus.py
│   │   ├── a06_sequence_tracker.py
│   │   └── ...
│   └── molecular/                       # Level 1
│       ├── __init__.py                  # ← M06 추가
│       ├── m01_hierarchy_builder.py
│       ├── m02_causality_detector.py
│       ├── m03_pattern_completer.py
│       ├── m04_spatial_transformer.py
│       └── m06_context_integrator.py    # ← 메인 코드
├── tests/
│   └── test_molecular_seeds.py          # ← M06 테스트 추가
├── examples/
│   └── m06_usage_examples.py            # ← 활용 예제
├── docs/
│   ├── M06_RESEARCH_MATERIALS.md        # ← 정보 자료
│   ├── M06_IMPLEMENTATION_GUIDE.md      # ← 구현 가이드
│   └── M06_PROJECT_INTEGRATION.md       # ← 본 문서
└── README.md                            # ← 업데이트 필요
```

### 2.2 Import 구조

```python
# seeds/molecular/__init__.py

from .m01_hierarchy_builder import HierarchyBuilder, create_hierarchy_builder
from .m02_causality_detector import CausalityDetector, create_causality_detector
from .m03_pattern_completer import PatternCompleter
from .m04_spatial_transformer import SpatialTransformer, create_spatial_transformer
from .m06_context_integrator import ContextIntegrator  # ← 추가

__all__ = [
    "HierarchyBuilder",
    "create_hierarchy_builder",
    "CausalityDetector",
    "create_causality_detector",
    "PatternCompleter",
    "SpatialTransformer",
    "create_spatial_transformer",
    "ContextIntegrator",  # ← 추가
]
```

### 2.3 메인 프로그램 예시

```python
# main.py (프로젝트 루트)

"""
Cognitive Seed Framework - Main Program

모든 시드를 통합하여 사용하는 메인 프로그램
"""

import torch
from seeds.molecular import (
    HierarchyBuilder,
    CausalityDetector,
    PatternCompleter,
    SpatialTransformer,
    ContextIntegrator
)


class CognitivePipeline:
    """인지 처리 파이프라인"""
    
    def __init__(self, input_dim=128):
        self.input_dim = input_dim
        
        # Molecular seeds 초기화
        self.hierarchy_builder = HierarchyBuilder(input_dim)
        self.causality_detector = CausalityDetector(input_dim)
        self.pattern_completer = PatternCompleter(input_dim)
        self.spatial_transformer = SpatialTransformer(input_dim)
        self.context_integrator = ContextIntegrator(input_dim)
    
    def process(self, x, task='context_integration'):
        """
        데이터 처리
        
        Args:
            x: 입력 데이터
            task: 처리 태스크
        Returns:
            결과
        """
        if task == 'context_integration':
            return self.context_integrator(x)
        
        elif task == 'pattern_completion_with_context':
            # 파이프라인: 패턴 완성 -> 맥락 통합
            completed = self.pattern_completer(x)
            integrated = self.context_integrator(completed)
            return integrated
        
        elif task == 'hierarchical_context':
            # 계층 구조 파악 후 맥락 통합
            hierarchy = self.hierarchy_builder(x)
            integrated = self.context_integrator(hierarchy)
            return integrated
        
        else:
            raise ValueError(f"Unknown task: {task}")


def main():
    """메인 함수"""
    
    # 파이프라인 생성
    pipeline = CognitivePipeline(input_dim=128)
    
    # 입력 데이터
    x = torch.randn(4, 50, 128)
    
    # 태스크 실행
    print("=" * 60)
    print("Cognitive Seed Framework - Main Program")
    print("=" * 60)
    
    # 1. 맥락 통합
    print("\n[Task 1] Context Integration")
    result1 = pipeline.process(x, task='context_integration')
    print(f"Output shape: {result1.shape}")
    
    # 2. 패턴 완성 + 맥락 통합
    print("\n[Task 2] Pattern Completion with Context")
    result2 = pipeline.process(x, task='pattern_completion_with_context')
    print(f"Output shape: {result2.shape}")
    
    # 3. 계층적 맥락
    print("\n[Task 3] Hierarchical Context")
    result3 = pipeline.process(x, task='hierarchical_context')
    print(f"Output shape: {result3.shape}")
    
    print("\n" + "=" * 60)
    print("All tasks completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 3. 다른 시드와의 연계

### 3.1 M01 (Hierarchy Builder)와 연계

```python
# 계층 구조를 파악한 후 맥락 통합

from seeds.molecular import HierarchyBuilder, ContextIntegrator

hierarchy_builder = HierarchyBuilder(input_dim=128)
context_integrator = ContextIntegrator(input_dim=128)

# 입력
x = torch.randn(4, 50, 128)

# 계층 구조 추출
hierarchy = hierarchy_builder(x)

# 계층 정보를 활용한 맥락 통합
# (M06 내부에서 M01을 사용하므로 자동으로 통합됨)
integrated = context_integrator(hierarchy)
```

### 3.2 M03 (Pattern Completer)와 연계

```python
# 결손 패턴을 완성한 후 맥락 통합

from seeds.molecular import PatternCompleter, ContextIntegrator

completer = PatternCompleter(input_dim=128)
integrator = ContextIntegrator(input_dim=128)

# 결손이 있는 입력
x = torch.randn(4, 50, 128)
mask = torch.ones(4, 50)
mask[:, 20:30] = 0  # 결손 구간

# 파이프라인
completed = completer(x, mask=mask)
integrated = integrator(completed)
```

### 3.3 M02 (Causality Detector)와 연계

```python
# 인과 관계를 파악한 후 맥락 통합

from seeds.molecular import CausalityDetector, ContextIntegrator

causality_detector = CausalityDetector(input_dim=128)
context_integrator = ContextIntegrator(input_dim=128)

# 입력
x = torch.randn(4, 50, 128)

# 인과 관계 분석
causal_features = causality_detector(x)

# 인과 정보를 포함한 맥락 통합
integrated = context_integrator(causal_features)
```

### 3.4 M04 (Spatial Transformer)와 연계

```python
# 공간 변환 후 맥락 통합

from seeds.molecular import SpatialTransformer, ContextIntegrator

transformer = SpatialTransformer(input_dim=128)
integrator = ContextIntegrator(input_dim=128)

# 입력
x = torch.randn(4, 50, 128)

# 공간 정규화
transformed = transformer(x)

# 정규화된 공간에서 맥락 통합
integrated = integrator(transformed)
```

---

## 4. 상위 레벨 시드 준비

### 4.1 M08 (Conflict Resolver) 준비

M06은 **M08의 핵심 구성 요소**입니다.

#### M08 설계 (예상)

```python
# seeds/molecular/m08_conflict_resolver.py (미구현)

from seeds.atomic import BinaryComparator
from seeds.molecular import ContextIntegrator, CausalityDetector

class ConflictResolver(BaseSeed):
    """
    SEED-M08: Conflict Resolver
    
    상충하는 정보를 맥락과 인과 관계를 고려하여 해소합니다.
    
    Composed From:
    - A08 (Binary Comparator)
    - M06 (Context Integrator)  ← M06 사용
    - M02 (Causality Detector)
    """
    
    def __init__(self, input_dim=128):
        self.comparator = BinaryComparator(input_dim)
        self.context_integrator = ContextIntegrator(input_dim)  # ← M06
        self.causality_detector = CausalityDetector(input_dim)
        
        # Conflict resolution network
        self.resolver = nn.Sequential(
            nn.Linear(input_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x1, x2):
        """
        두 상충하는 입력을 해소
        
        Args:
            x1, x2: 상충하는 입력
        Returns:
            resolved: 해소된 결과
        """
        # 1. 비교
        comparison = self.comparator(x1, x2)
        
        # 2. 맥락 통합 (M06 사용)
        context1 = self.context_integrator(x1)
        context2 = self.context_integrator(x2)
        
        # 3. 인과 관계 분석
        causal1 = self.causality_detector(x1)
        causal2 = self.causality_detector(x2)
        
        # 4. 해소
        combined = torch.cat([context1, context2, comparison], dim=-1)
        resolved = self.resolver(combined)
        
        return resolved
```

### 4.2 Level 2 (Cellular) 준비

M06은 Level 2 시드에서도 재사용 가능합니다.

```python
# 예상 구조

class CellularSeed(BaseSeed):
    """Level 2 시드 예시"""
    
    def __init__(self, input_dim=128):
        # Molecular seeds 조합
        self.context_integrator = ContextIntegrator(input_dim)  # ← M06 재사용
        self.pattern_completer = PatternCompleter(input_dim)
        # ...
```

---

## 5. API 설계

### 5.1 공개 API

```python
# seeds/molecular/m06_context_integrator.py

class ContextIntegrator(BaseSeed):
    """
    공개 API:
    - __init__(): 초기화
    - forward(): 메인 처리
    - get_context_importance(): 맥락 중요도 분석
    - visualize_context_attention(): 시각화
    """
    
    def __init__(self, input_dim=128, **kwargs):
        """초기화"""
        pass
    
    def forward(self, x, context_window=None, return_metadata=False):
        """메인 처리"""
        pass
    
    def get_context_importance(self, x):
        """맥락 중요도 분석"""
        pass
    
    def visualize_context_attention(self, x, position):
        """시각화"""
        pass
```

### 5.2 내부 API

```python
# 내부 메서드 (private)

class ContextIntegrator(BaseSeed):
    """
    내부 API:
    - _init_atomic_seeds(): Atomic seeds 초기화
    - _init_context_encoders(): Context encoders 초기화
    - _init_fusion_module(): Fusion module 초기화
    - _init_disambiguator(): Disambiguator 초기화
    - encode_local_context(): Local context 인코딩
    - encode_global_context(): Global context 인코딩
    - fuse_contexts(): 맥락 융합
    - disambiguate(): 중의성 해소
    """
    
    def _init_atomic_seeds(self):
        """내부: Atomic seeds 초기화"""
        pass
    
    # ... 기타 내부 메서드
```

### 5.3 사용자 인터페이스

```python
# 사용자가 사용하는 간단한 인터페이스

from seeds.molecular import ContextIntegrator

# 1. 기본 사용
integrator = ContextIntegrator()
output = integrator(input_data)

# 2. 메타데이터 포함
output, metadata = integrator(input_data, return_metadata=True)

# 3. 맥락 중요도 분석
importance = integrator.get_context_importance(input_data)

# 4. 시각화
attention_maps = integrator.visualize_context_attention(input_data, position=10)
```

---

## 6. 배포 전략

### 6.1 버전 관리

```python
# seeds/molecular/m06_context_integrator.py

__version__ = "1.0.0"
__author__ = "Cognitive Seed Framework Team"
__status__ = "Production"  # Development | Beta | Production
```

### 6.2 문서화

#### README 업데이트

```markdown
# Cognitive Seed Framework

## Level 1 (Molecular) Seeds

### 구현 완료 (5/8)

- ✅ M01: Hierarchy Builder
- ✅ M02: Causality Detector
- ✅ M03: Pattern Completer
- ✅ M04: Spatial Transformer
- ✅ M06: Context Integrator  ← 추가

### 구현 예정 (3/8)

- ⏳ M05: Concept Crystallizer
- ⏳ M07: Analogy Mapper
- ⏳ M08: Conflict Resolver
```

#### CHANGELOG 업데이트

```markdown
# Changelog

## [Unreleased]

### Added
- M06 Context Integrator implementation
- Multi-scale context encoding (local/global)
- Hierarchical context integration
- Context fusion with multi-head attention
- Context-based disambiguation
- 8 comprehensive usage examples
- Visualization tools for fusion weights

### Changed
- Updated molecular __init__.py
- Enhanced test suite with M06 tests

### Documentation
- M06_RESEARCH_MATERIALS.md
- M06_IMPLEMENTATION_GUIDE.md
- M06_PROJECT_INTEGRATION.md
- examples/m06_usage_examples.py
```

### 6.3 테스트 커버리지

```bash
# 테스트 실행
python tests/test_molecular_seeds.py

# 커버리지 측정 (선택적)
pip install pytest-cov
pytest tests/test_molecular_seeds.py --cov=seeds.molecular --cov-report=html
```

### 6.4 CI/CD 통합

```yaml
# .github/workflows/test.yml (예시)

name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install torch
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python tests/test_molecular_seeds.py
```

---

## 7. 체크리스트

### 7.1 구현 체크리스트

- [ ] M06 메인 코드 작성
- [ ] 테스트 작성 및 통과
- [ ] 활용 예제 작성
- [ ] 문서 작성 (3개)
- [ ] __init__.py 업데이트
- [ ] README 업데이트
- [ ] CHANGELOG 업데이트

### 7.2 통합 체크리스트

- [ ] 다른 시드와의 연계 테스트
- [ ] 메인 프로그램 작성
- [ ] API 문서화
- [ ] 성능 벤치마크
- [ ] 메모리 프로파일링

### 7.3 배포 체크리스트

- [ ] 버전 번호 설정
- [ ] 라이선스 확인
- [ ] 의존성 명시
- [ ] CI/CD 설정
- [ ] 릴리스 노트 작성

---

## 8. 다음 단계

### 8.1 Phase 2 완료

M06 구현 완료 시:
- ✅ M03: Pattern Completer
- ✅ M06: Context Integrator
- **Phase 2 완료!**

### 8.2 Phase 3 진행

다음 구현 대상:
- M05: Concept Crystallizer (A05 + M03 + M01)
- M07: Analogy Mapper (M01 + A08 + M05)

### 8.3 Phase 4 진행

최종 구현:
- M08: Conflict Resolver (A08 + M06 + M02)

---

**작성일**: 2025-10-21  
**작성자**: Manus AI (누스양)  
**업데이트**: M06 구현 완료 시

