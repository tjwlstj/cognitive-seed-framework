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
- **양자화 지원**: INT8/FP8/FP16 다양한 비트폭으로 효율적 추론

## 아키텍처

### 설계 철학

1. **모듈성 & 재사용성**: 태스크 독립적 핵심 인지 기능을 모듈화
2. **기하학적 적합성**: 데이터 구조에 맞춘 다중 기하학 공간 활용
3. **스케일 강건성**: 연속 스케일 조건부 처리로 입력 변화에 대응
4. **정량 표준**: 명확한 I/O 규격, 벤치마크, 수용 기준
5. **설명가능성**: 각 시드의 기능, 가정, 제약을 투명하게 문서화

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
│   ├── cellular/            # Level 2: 8개 세포 시드
│   └── tissue/              # Level 3: 8개 조직 시드
├── compositions/            # 시드 조합 레시피
├── benchmarks/              # 평가 벤치마크 및 결과
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

```python
from seeds import load_seed, SeedRouter

# 개별 시드 로드
edge_detector = load_seed("SEED-A01")
output = edge_detector(input_tensor)

# 시드 라우터 사용
router = SeedRouter()
active_seeds = router.select(task="segmentation", context=context)
result = router.forward(input_tensor, active_seeds)
```

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

- **Phase 1**: 32 시드 참조 구현 + 단독 벤치마크 (현재)
- **Phase 2**: 백본 통합·QAT + 공개 벤치마크 결과
- **Phase 3**: 허브/배포 자동화, 아키텍처 검색
- **Phase 4**: 신경과학 영감 신규 시드, 안전·윤리 프레임 통합

## 기여

본 프로젝트는 오픈소스 기여를 환영합니다. 기여 가이드라인은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

## 라이선스

Apache License 2.0 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 참고문헌

- 표준 인지 시드 설계 가이드 v1.1 (2025-10-20)
- 작성: 체시(Chesi) · 협업: 제로(Zero)

## 연락처

- Issues: [GitHub Issues](https://github.com/tjwlstj/cognitive-seed-framework/issues)
- Discussions: [GitHub Discussions](https://github.com/tjwlstj/cognitive-seed-framework/discussions)

---

**Built with curiosity and precision** 🧠✨

