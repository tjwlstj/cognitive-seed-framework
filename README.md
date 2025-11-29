# Cognitive Seed Framework

**표준 인지 시드 설계 가이드 v1.1 기반 구현**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)

## 개요

본 프로젝트는 **인지 시드(Cognitive Seed)** 설계 및 구현을 위한 표준 프레임워크입니다. 모듈식 지능 시스템을 위해 **Multi-Geometry Projection (MGP)**, **Continuous Scale-Equivariant (CSE)**, **Seed Routing**을 통합한 현대적 아키텍처를 제공합니다.

## 핵심 특징

- **32개 표준 인지 시드**: 4개 레벨(Atomic, Molecular, Cellular, Tissue)로 구성된 계층적 인지 모듈
- **다중 기하학 투영**: Euclidean, Hyperbolic, Spherical 공간을 병렬로 활용
- **연속 스케일 등변성**: 입력 스케일 변화에 강건한 조건부 정규화
- **동적 시드 라우팅**: 태스크와 맥락에 따라 최적 시드 조합을 선택
- **재현성 보장**: PyTorch DataLoader worker seed 초기화 및 deterministic 설정 지원
- **명명 규칙 통일**: 다양한 시드 ID 형식 지원 (A01, SEED-A01, A01_Edge_Detector 등)

## 아키텍처

### 설계 철학

1. **모듈성 & 재사용성**: 태스크 독립적 핵심 인지 기능을 모듈화
2. **기하학적 적합성**: 데이터 구조에 맞춘 다중 기하학 공간 활용
3. **스케일 강건성**: 연속 스케일 조건부 처리로 입력 변화에 대응
4. **정량 표준**: 명확한 I/O 규격, 벤치마크, 수용 기준
5. **설명가능성**: 각 시드의 기능, 가정, 제약을 투명하게 문서화
6. **재현성**: 완전한 재현성을 위한 시드 관리 및 deterministic 설정

### 핵심 컴포넌트

```
┌─────────────────────────────────────────┐
│         Seed Router                     │
│  (Task/Context → Seed Selection)        │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
┌───▼────────┐         ┌────────▼────┐
│  MGP Block │         │  CSE Block  │
│  E/H/S     │◄────────┤  Scale-     │
│  Projection│         │  Equivariant│
└────────────┘         └─────────────┘
```

## 프로젝트 구조

```
cognitive-seed-framework/
├── seeds/                    # 시드 구현 및 가중치
│   ├── atomic/              # Level 0: 8개 원자 시드
│   ├── molecular/           # Level 1: 8개 분자 시드
│   ├── cellular/            # Level 2: 8개 세포 시드 (예정)
│   └── tissue/              # Level 3: 8개 조직 시드 (예정)
├── core/                    # 코어 아키텍처
│   ├── registry.py          # 시드 레지스트리
│   ├── router.py            # 시드 라우터
│   ├── composition.py       # 조합 엔진 (DAG)
│   ├── cache.py             # 캐시 관리자
│   ├── metrics.py           # 메트릭 수집기
│   └── reproducibility.py   # 재현성 유틸리티
├── examples/                # 사용 예제
├── docs/                    # 문서 및 가이드
└── README.md
```

## 32개 인지 시드 카탈로그

### Level 0 — Atomic (8)

| ID | Name | Category | 핵심 용도 |
|---|---|---|---|
| A01 | Edge Detector | Pattern | 경계/전환 검출 |
| A02 | Symmetry Detector | Spatial | 대칭 축/정도 추정 |
| A03 | Recurrence Spotter | Temporal | 반복/주기 검출 |
| A04 | Contrast Amplifier | Pattern | 대비 증폭·노이즈 억제 |
| A05 | Grouping Nucleus | Relation | 유사도 기반 군집 |
| A06 | Sequence Tracker | Temporal | 순서 추적·예측 |
| A07 | Scale Normalizer | Abstraction | 스케일 정규화 |
| A08 | Binary Comparator | Logic | 대소/동등 비교 |

### Level 1 — Molecular (8)

| ID | Name | Category | 핵심 용도 |
|---|---|---|---|
| M01 | Hierarchy Builder | Relation | 상하 관계 트리/DAG 구축 |
| M02 | Causality Detector | Temporal/Logic | 인과 구조 추정 |
| M03 | Pattern Completer | Pattern | 결손 보간/외삽 |
| M04 | Spatial Transformer | Spatial | 회전·스케일 정렬 |
| M05 | Concept Crystallizer | Abstraction | 프로토타입 학습 |
| M06 | Context Integrator | Composition | 맥락 융합 |
| M07 | Analogy Mapper | Analogy | 구조적 유사성 매핑 |
| M08 | Conflict Resolver | Logic | 제약 충돌 해소 |

### Level 2 — Cellular (8)

| ID | Name | Category | 핵심 용도 |
|---|---|---|---|
| C01 | Metaphor Engine | Analogy | 은유 매핑 |
| C02 | Counterfactual Reasoner | Logic | 반사실 시뮬레이션 |
| C03 | Schema Learner | Abstraction | 스키마 구조 학습 |
| C04 | Perspective Shifter | Spatial/Analogy | 관점 전환 |
| C05 | Narrative Constructor | Composition | 서사 구조화 |
| C06 | Attention Director | Composition | 주의 가중 배분 |
| C07 | Boundary Detector | Pattern | 의미 경계 탐지 |
| C08 | Novelty Assessor | Abstraction | 참신성 평가 |

### Level 3 — Tissue (8)

| ID | Name | Category | 핵심 용도 |
|---|---|---|---|
| T01 | Abductive Reasoner | Logic | 최선 설명 추론 |
| T02 | Analogical Transfer Engine | Analogy | 구조 전이·적응 |
| T03 | Theory Builder | Abstraction | 이론화 |
| T04 | Strategic Planner | Composition | 목표 분해·계획 |
| T05 | Social Modeler | Relation | 신념/욕구/의도 추론 |
| T06 | Meta-Learner | Abstraction | 메타학습·신속 적응 |
| T07 | Ethical Reasoner | Logic | 윤리 판단 |
| T08 | Creative Synthesizer | Composition | 창의적 결합 |

## 시작하기

### 요구사항

- Python 3.11+
- PyTorch 2.0+
- NumPy, SciPy
- (선택) CUDA 11.8+ for GPU acceleration

### 설치

```bash
git clone https://github.com/tjwlstj/cognitive-seed-framework.git
cd cognitive-seed-framework
pip install -r requirements.txt
```

### 빠른 시작

#### 방법 1: load_seed() 헬퍼 함수 사용 (권장)

```python
from seeds import load_seed

# 개별 시드 로드 - 다양한 명명 규칙 지원
edge_detector = load_seed("SEED-A01")  # ✅ 작동
edge_detector = load_seed("A01")        # ✅ 작동 (동일한 시드)
edge_detector = load_seed("A01_Edge_Detector")  # ✅ 작동 (동일한 시드)

# 시드 실행
import torch
input_tensor = torch.randn(1, 3, 224, 224)
output = edge_detector(input_tensor)
```

#### 방법 2: 코어 아키텍처 사용 (고급)

```python
from core import SeedRegistry, SeedRouter, CompositionEngine, CacheManager
from seeds import load_seed

# 1. 코어 컴포넌트 초기화
registry = SeedRegistry()
cache = CacheManager()
router = SeedRouter(registry)
engine = CompositionEngine(registry, cache)

# 2. 시드 등록 (별칭 지원)
from core import SeedMetadata

edge_detector = load_seed("A01")
metadata = SeedMetadata(
    name="A01_Edge_Detector",
    level=0,
    version="1.0.0",
    description="Detects edges in images",
    geometry=["E"],
    tags=["vision", "edge"]
)
registry.register(
    "A01_Edge_Detector",
    edge_detector,
    metadata,
    aliases=["A01", "SEED-A01"]  # 별칭 등록
)

# 3. 시드 조회 (별칭으로도 가능)
seed = registry.get("A01")  # ✅ 작동
seed = registry.get("SEED-A01")  # ✅ 작동
seed = registry.get("A01_Edge_Detector")  # ✅ 작동

# 4. 태스크 실행
selected_seeds = ["A01_Edge_Detector"]
result = engine.execute(selected_seeds, input_tensor)
```

#### 방법 3: 재현성 보장

```python
from core import set_seed, enable_reproducibility
from seeds import load_seed

# 재현성 활성화 (Magic Seed 3407 사용)
enable_reproducibility()

# 또는 커스텀 시드 사용
set_seed(42, deterministic=True)

# 이제 모든 실행이 재현 가능
model = load_seed("A01")
output = model(input_tensor)
```

## 재현성 보장

프레임워크는 완전한 재현성을 위한 유틸리티를 제공합니다:

```python
from core import (
    set_seed,
    seed_worker,
    get_reproducible_dataloader_config,
    check_reproducibility,
    ReproducibleContext
)
from torch.utils.data import DataLoader

# 1. 전역 시드 설정
set_seed(42, deterministic=True)

# 2. DataLoader 재현성 (worker seed 초기화)
config = get_reproducible_dataloader_config()
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, **config)

# 3. 재현성 자동 체크
model = load_seed("A01")
is_reproducible = check_reproducibility(model, input_tensor, seed=42, num_runs=5)

# 4. 컨텍스트 매니저 사용
with ReproducibleContext(seed=42):
    output = model(input_tensor)  # 이 블록 내에서 재현성 보장
```

자세한 내용은 `examples/reproducibility_example.py`를 참조하세요.

## 평가 및 벤치마크

각 레벨별 수용 기준:

- **Level 0**: F1 ≥ 0.90, latency < 1ms/32샘플
- **Level 1**: AMI/ARI ≥ 0.85, latency < 10ms
- **Level 2**: Few-shot 지표 ≥ 0.80, latency < 100ms
- **Level 3**: 인간 합의율 ≥ 0.70, < 1s

벤치마크 실행:

```bash
python benchmarks/run_evaluation.py --level all --output results.json
```

## 로드맵

- **Phase 1**: 32 시드 참조 구현 + 단독 벤치마크 ✅ (Level 0 완료)
- **Phase 2**: Level 1 (Molecular) 구현 ✅ (8/8 완료)
- **Phase 3**: Level 2 (Cellular) 구현 (예정)
- **Phase 4**: Level 3 (Tissue) 구현 (예정)
- **Phase 5**: 백본 통합·QAT + 공개 벤치마크 결과
- **Phase 6**: 허브/배포 자동화, 아키텍처 검색
- **Phase 7**: 신경과학 영감 신규 시드, 안전·윤리 프레임 통합

## 구현 현황

### Level 0 (Atomic) - 100% ✅
- A01~A08 전체 구현 완료
- 총 파라미터: ~1.09M

### Level 1 (Molecular) - 100% ✅
- M01~M08 전체 구현 완료
- 총 파라미터: ~6.37M
- 최근 완성: M08 Conflict Resolver (2025-11-17)

### Level 2 (Cellular) - 12.5% 🟡
- C01 Metaphor Engine: ✅ 완료 (2025-11-26)
- C02~C08: 📅 예정

### Level 3 (Tissue) - 0% 📅
- T01~T08 예정

**전체 진행률**: 16/32 (50%) 🎉

## 최근 업데이트 (v1.2.0)

### 추가된 기능

1. **M08 Conflict Resolver 구현**: Level 1 완성 (8/8)
   - A08, M06, M02 통합
   - 충돌 탐지, 우선순위 계산, 공정성 보장
   - 24개 단위 테스트 포함
2. **M05 Concept Crystallizer 등록**: seeds/__init__.py 업데이트
3. **개발 계획 v2**: 토큰 효율적인 분할 개발 계획 수립

### 주요 마일스톤

- ✅ Level 0 (Atomic) 8개 시드 완성
- ✅ Level 1 (Molecular) 8개 시드 완성
- ✅ 보안 검사 통과 (Bandit, pip-audit)
- ✅ 의존성 안정화
- 📅 Level 2 (Cellular) 구현 예정

## 기여

본 프로젝트는 오픈소스 기여를 환영합니다. 기여 가이드라인은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

## 라이선스

Apache License 2.0 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 참고문헌

- 표준 인지 시드 설계 가이드 v1.1 (2025-10-20)
- 작성: 체시(Chesi) · 협업: 제로(Zero)
- "Torch.manual_seed(3407) is all you need" - https://arxiv.org/abs/2109.08203
- PyTorch Reproducibility Guide - https://pytorch.org/docs/stable/notes/randomness.html

## 연락처

- Issues: [GitHub Issues](https://github.com/tjwlstj/cognitive-seed-framework/issues)
- Discussions: [GitHub Discussions](https://github.com/tjwlstj/cognitive-seed-framework/discussions)

---

**Built with curiosity and precision** 🧠✨

