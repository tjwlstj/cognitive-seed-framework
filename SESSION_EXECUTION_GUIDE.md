# 개발 세션 실행 가이드

**작성일**: 2025-11-13  
**작성자**: Manus AI  
**목적**: 각 개발 세션의 구체적인 실행 방법 제공

---

## 개요

본 가이드는 `DEVELOPMENT_PLAN.md`에 정의된 4개의 개발 세션을 실행하기 위한 구체적인 단계별 지침을 제공합니다. 각 세션은 독립적으로 실행 가능하며, 명확한 입력과 출력을 가집니다.

---

## 세션 1: M08 Conflict Resolver 구현

### 사전 준비

프로젝트 디렉토리로 이동하고 가상환경을 활성화합니다.

```bash
cd /home/ubuntu/cognitive-seed-framework
source venv/bin/activate
```

### 단계 1: 의존 시드 코드 리뷰 (30분)

다음 파일들을 읽고 인터페이스를 이해합니다.

```bash
# A08 Binary Comparator
cat seeds/atomic/a08_binary_comparator.py

# M06 Context Integrator
cat seeds/molecular/m06_context_integrator.py

# M02 Causality Detector
cat seeds/molecular/m02_causality_detector.py
```

**핵심 확인 사항**:
- 각 시드의 입출력 형태 (shape)
- forward() 메서드 시그니처
- return_metadata 옵션 사용법

### 단계 2: 아키텍처 설계 (1시간)

`seeds/molecular/m08_conflict_resolver.py` 파일을 생성하고 기본 구조를 작성합니다.

```python
"""
SEED-M08: Conflict Resolver
Category: Logic
Composed From: A08 (Binary Comparator) + M06 (Context Integrator) + M02 (Causality Detector)
"""

import torch
import torch.nn as nn
from ..base import BaseSeed
from ..atomic.a08_binary_comparator import BinaryComparator
from .m06_context_integrator import ContextIntegrator
from .m02_causality_detector import CausalityDetector

class ConflictResolver(BaseSeed):
    """
    M08 Conflict Resolver
    
    제약 충돌을 탐지하고 타협 솔루션을 생성합니다.
    
    Architecture:
        Input [B, N, D]  # N개의 제약 조건
            │
            ├─→ Constraint Encoder
            ├─→ Conflict Detector (A08)
            ├─→ Context Analyzer (M06)
            ├─→ Causality Reasoner (M02)
            └─→ Resolution Generator
            │
        Output [B, N, D]  # 해결된 제약 조건
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_constraints: int = 10,
        resolution_strategies: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_constraints = num_constraints
        self.resolution_strategies = resolution_strategies
        
        # TODO: 컴포넌트 초기화
        
    def forward(self, x, return_metadata=False):
        # TODO: 구현
        pass
```

### 단계 3: 핵심 컴포넌트 구현 (3시간)

다음 컴포넌트들을 순차적으로 구현합니다.

1. **Constraint Encoder**: 제약 조건 인코딩
2. **Conflict Detector**: A08을 사용한 충돌 탐지
3. **Context Analyzer**: M06을 사용한 맥락 분석
4. **Causality Reasoner**: M02를 사용한 인과 추론
5. **Resolution Generator**: 해결책 생성

### 단계 4: 테스트 코드 작성 (1.5시간)

`tests/molecular/test_m08_conflict_resolver.py` 파일을 생성합니다.

```python
import torch
import pytest
from seeds.molecular.m08_conflict_resolver import ConflictResolver

def test_m08_initialization():
    """M08 초기화 테스트"""
    model = ConflictResolver(input_dim=64, hidden_dim=128)
    assert model is not None
    # 파라미터 수 확인 (~800K 목표)
    total_params = sum(p.numel() for p in model.parameters())
    assert 700_000 <= total_params <= 900_000

def test_m08_forward():
    """M08 forward pass 테스트"""
    model = ConflictResolver(input_dim=64)
    x = torch.randn(4, 10, 64)  # 4 batch, 10 constraints
    output = model(x)
    assert output.shape == x.shape

def test_m08_conflict_detection():
    """충돌 탐지 테스트"""
    # TODO: 구현

def test_m08_resolution_generation():
    """해결책 생성 테스트"""
    # TODO: 구현

def test_m08_metadata():
    """메타데이터 반환 테스트"""
    # TODO: 구현
```

테스트 실행:

```bash
pytest tests/molecular/test_m08_conflict_resolver.py -v
```

### 단계 5: 파라미터 수 검증 및 조정 (1시간)

파라미터 수를 확인하고 목표 범위(~800K)에 맞춥니다.

```python
# 파라미터 수 확인 스크립트
from seeds.molecular.m08_conflict_resolver import ConflictResolver

model = ConflictResolver(input_dim=128, hidden_dim=256)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# hidden_dim 조정하여 목표 파라미터 수 달성
```

### 단계 6: 문서화 (1.5시간)

`M08_IMPLEMENTATION_COMPLETE.md` 파일을 작성합니다. `M05_IMPLEMENTATION_COMPLETE.md`를 템플릿으로 사용합니다.

### 단계 7: 프로젝트 통합 (30분)

`seeds/__init__.py`와 `seeds/molecular/__init__.py`를 업데이트합니다.

```python
# seeds/molecular/__init__.py에 추가
from .m08_conflict_resolver import ConflictResolver

__all__ = [
    # ... 기존 항목들 ...
    'ConflictResolver',
]
```

### 단계 8: Git 커밋 및 푸시 (15분)

```bash
git add seeds/molecular/m08_conflict_resolver.py
git add tests/molecular/test_m08_conflict_resolver.py
git add M08_IMPLEMENTATION_COMPLETE.md
git add seeds/__init__.py seeds/molecular/__init__.py

git commit -m "feat: Implement M08 Conflict Resolver

- Add M08: Conflict Resolver (Logic, ~800K params)
- Compose A08 + M06 + M02 for constraint conflict resolution
- Add comprehensive unit tests (5 test cases)
- Add implementation completion report
- Update seed registry"

git push origin main
```

### 예상 소요 시간

- 코드 리뷰: 30분
- 설계: 1시간
- 구현: 3시간
- 테스트: 1.5시간
- 파라미터 조정: 1시간
- 문서화: 1.5시간
- 통합 및 커밋: 45분
- **총합**: ~9.5시간 (1-2일)

---

## 세션 2: M07 Analogy Mapper 구현

### 사전 준비

```bash
cd /home/ubuntu/cognitive-seed-framework
source venv/bin/activate
```

### 단계 1: 의존 시드 코드 리뷰 (30분)

```bash
# M01 Hierarchy Builder
cat seeds/molecular/m01_hierarchy_builder.py

# A08 Binary Comparator
cat seeds/atomic/a08_binary_comparator.py

# M05 Concept Crystallizer
cat seeds/molecular/m05_concept_crystallizer.py
```

### 단계 2: 아키텍처 설계 (1시간)

`seeds/molecular/m07_analogy_mapper.py` 파일을 생성합니다.

```python
"""
SEED-M07: Analogy Mapper
Category: Analogy
Composed From: M01 (Hierarchy Builder) + A08 (Binary Comparator) + M05 (Concept Crystallizer)
"""

import torch
import torch.nn as nn
from ..base import BaseSeed
from .m01_hierarchy_builder import HierarchyBuilder
from ..atomic.a08_binary_comparator import BinaryComparator
from .m05_concept_crystallizer import ConceptCrystallizer

class AnalogyMapper(BaseSeed):
    """
    M07 Analogy Mapper
    
    구조적 유사성을 매핑하고 유추 추론을 수행합니다.
    
    Architecture:
        Source [B, N, D]    Target [B, M, D]
            │                   │
            ├─→ Structure Encoder (M01)
            ├─→ Concept Matcher (M05)
            ├─→ Similarity Scorer (A08)
            └─→ Mapping Generator
            │
        Output [B, N, M]  # 매핑 행렬
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_prototypes: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_prototypes = num_prototypes
        
        # TODO: 컴포넌트 초기화
        
    def forward(self, source, target, return_metadata=False):
        # TODO: 구현
        pass
```

### 단계 3: 핵심 컴포넌트 구현 (3시간)

1. **Structure Encoder**: M01을 사용한 구조 인코딩
2. **Concept Matcher**: M05를 사용한 개념 매칭
3. **Similarity Scorer**: A08을 사용한 유사도 평가
4. **Mapping Generator**: 매핑 행렬 생성

### 단계 4: 테스트 코드 작성 (1.5시간)

`tests/molecular/test_m07_analogy_mapper.py` 파일을 생성합니다.

```python
import torch
import pytest
from seeds.molecular.m07_analogy_mapper import AnalogyMapper

def test_m07_initialization():
    """M07 초기화 테스트"""
    model = AnalogyMapper(input_dim=64, hidden_dim=128)
    assert model is not None
    # 파라미터 수 확인 (~750K 목표)
    total_params = sum(p.numel() for p in model.parameters())
    assert 650_000 <= total_params <= 850_000

def test_m07_forward():
    """M07 forward pass 테스트"""
    model = AnalogyMapper(input_dim=64)
    source = torch.randn(4, 10, 64)
    target = torch.randn(4, 12, 64)
    output = model(source, target)
    assert output.shape == (4, 10, 12)

def test_m07_structure_mapping():
    """구조 매핑 테스트"""
    # TODO: 구현

def test_m07_concept_matching():
    """개념 매칭 테스트"""
    # TODO: 구현

def test_m07_metadata():
    """메타데이터 반환 테스트"""
    # TODO: 구현
```

### 단계 5-8: M08과 동일한 프로세스

파라미터 검증, 문서화, 통합, 커밋을 M08과 동일하게 진행합니다.

### Git 커밋 메시지

```bash
git commit -m "feat: Implement M07 Analogy Mapper

- Add M07: Analogy Mapper (Analogy, ~750K params)
- Compose M01 + A08 + M05 for structural similarity mapping
- Add comprehensive unit tests (5 test cases)
- Add implementation completion report
- Update seed registry"
```

### 예상 소요 시간

**총합**: ~9.5시간 (1-2일)

---

## 세션 3: Level 1 통합 및 벤치마크

### 사전 준비

```bash
cd /home/ubuntu/cognitive-seed-framework
source venv/bin/activate
```

### 단계 1: 전체 시드 통합 테스트 (2시간)

모든 Level 1 시드가 정상 작동하는지 확인합니다.

```python
# integration_test.py
from seeds import load_seed
import torch

# Level 1 시드 목록
molecular_seeds = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08']

for seed_id in molecular_seeds:
    print(f"Testing {seed_id}...")
    seed = load_seed(seed_id, input_dim=128)
    
    # 기본 forward pass 테스트
    if seed_id == 'M07':
        source = torch.randn(2, 10, 128)
        target = torch.randn(2, 12, 128)
        output = seed(source, target)
    elif seed_id == 'M05':
        support = torch.randn(5, 5, 128)
        query = torch.randn(10, 128)
        output = seed(support, query)
    else:
        x = torch.randn(2, 50, 128)
        output = seed(x)
    
    print(f"  ✓ {seed_id} passed")
```

### 단계 2: 벤치마크 데이터셋 준비 (1.5시간)

`benchmarks/` 디렉토리를 생성하고 벤치마크 데이터를 준비합니다.

```bash
mkdir -p benchmarks/data
```

### 단계 3: 평가 스크립트 작성 (2시간)

`benchmarks/level1_benchmark.py` 파일을 작성합니다.

```python
"""
Level 1 (Molecular) Seeds Benchmark

수용 기준:
- AMI/ARI ≥ 0.85
- Latency < 10ms per batch (32 samples)
"""

import torch
import time
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from seeds import load_seed

def benchmark_seed(seed_id, num_runs=100):
    """개별 시드 벤치마크"""
    seed = load_seed(seed_id, input_dim=128)
    
    # Latency 측정
    latencies = []
    for _ in range(num_runs):
        x = torch.randn(32, 50, 128)
        start = time.time()
        with torch.no_grad():
            output = seed(x)
        end = time.time()
        latencies.append((end - start) * 1000)  # ms
    
    avg_latency = sum(latencies) / len(latencies)
    
    return {
        'seed_id': seed_id,
        'avg_latency_ms': avg_latency,
        'latency_passed': avg_latency < 10.0
    }

# 전체 시드 벤치마크 실행
results = []
for seed_id in ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08']:
    result = benchmark_seed(seed_id)
    results.append(result)
    print(f"{seed_id}: {result['avg_latency_ms']:.2f}ms - {'✓' if result['latency_passed'] else '✗'}")
```

### 단계 4: 문서 업데이트 (1.5시간)

다음 파일들을 업데이트합니다:

1. **README.md**: Level 1 완성 표시
2. **CHANGELOG.md**: v1.2.0 변경 사항 추가
3. **ROADMAP.md**: Phase 2 완료 표시

### 단계 5: PR #1 리뷰 및 병합 (30분)

```bash
# PR 체크아웃
gh pr checkout 1

# 변경 사항 확인
git diff main

# 병합
gh pr merge 1 --squash
```

### 단계 6: Git 태그 생성 (15분)

```bash
git tag -a v1.2.0 -m "Release v1.2.0: Level 1 (Molecular) Complete

- Complete all 8 Molecular seeds (M01-M08)
- Add Level 1 benchmark suite
- Add comprehensive documentation
- Total parameters: ~6.37M"

git push origin v1.2.0
```

### 예상 소요 시간

**총합**: ~7.5시간 (1일)

---

## 세션 4: 보안 강화 및 유지보수

### 사전 준비

```bash
cd /home/ubuntu/cognitive-seed-framework
```

### 단계 1: SECURITY.md 작성 (1시간)

```markdown
# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 1.2.x | ✅ |
| 1.1.x | ✅ |
| < 1.1 | ❌ |

## Reporting a Vulnerability

보안 취약점을 발견하신 경우, 다음 방법으로 보고해 주세요:

1. **GitHub Security Advisories** (권장)
   - https://github.com/tjwlstj/cognitive-seed-framework/security/advisories/new

2. **이메일**
   - [보안 담당자 이메일]

3. **비공개 이슈**
   - 심각한 취약점은 공개 이슈로 보고하지 마세요

## Response Timeline

- 초기 응답: 48시간 이내
- 취약점 확인: 7일 이내
- 패치 배포: 심각도에 따라 1-30일

## Security Update Policy

보안 패치는 지원되는 모든 버전에 백포트됩니다.
```

### 단계 2: CI/CD 워크플로우 작성 (2시간)

`.github/workflows/tests.yml` 파일을 작성합니다.

```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=seeds --cov=core
    
    - name: Security scan
      run: |
        pip install bandit
        bandit -r core/ seeds/ -f json -o bandit-report.json
```

### 단계 3: 커뮤니티 문서 작성 (1.5시간)

1. **CONTRIBUTING.md**: 기여 가이드라인
2. **CODE_OF_CONDUCT.md**: 행동 강령
3. **Issue/PR 템플릿**

### 단계 4: Git 커밋 및 푸시 (15분)

```bash
git add SECURITY.md CONTRIBUTING.md CODE_OF_CONDUCT.md
git add .github/

git commit -m "docs: Add security policy and community guidelines

- Add SECURITY.md with vulnerability reporting process
- Add CI/CD workflows for automated testing
- Add CONTRIBUTING.md and CODE_OF_CONDUCT.md
- Add issue and PR templates"

git push origin main
```

### 예상 소요 시간

**총합**: ~4.5시간 (0.5일)

---

## 전체 일정 요약

| 세션 | 예상 시간 | 우선순위 |
|---|---|---|
| 세션 1: M08 구현 | 1-2일 | P0 |
| 세션 2: M07 구현 | 1-2일 | P0 |
| 세션 3: 통합 및 벤치마크 | 1일 | P1 |
| 세션 4: 보안 및 유지보수 | 0.5일 | P2 |
| **총합** | **3.5-5일** | - |

---

## 문제 해결 가이드

### 파라미터 수 초과 시

1. `hidden_dim` 감소
2. 레이어 수 감소
3. 구성 시드를 경량 버전으로 대체 (M05 사례 참고)

### 테스트 실패 시

1. 입출력 shape 확인
2. 의존 시드 버전 확인
3. 단계별 디버깅 (print 문 추가)

### Git 충돌 시

```bash
git fetch origin
git rebase origin/main
# 충돌 해결 후
git rebase --continue
```

---

**작성일**: 2025-11-13  
**작성자**: Manus AI  
**버전**: 1.0
