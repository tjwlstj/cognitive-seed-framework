# Level 1 (Molecular) Seeds

Level 0 (Atomic) 시드를 조합하여 더 복잡한 인지 기능을 수행하는 중간 단계 모듈입니다.

## 개요

Molecular 시드는 2-3개의 Atomic 시드를 조합하여 구조적 추론, 인과 관계 파악, 패턴 완성, 공간 변환 등의 고급 기능을 제공합니다.

### 특징

- **조합성 (Compositionality)**: 하위 시드의 기능을 조합하여 새로운 능력 창발
- **계층적 추상화**: 원자 수준보다 높은 추상화 레벨에서 동작
- **중간 복잡도**: 파라미터 수 증가, 더 복잡한 연산 수행
- **재사용성**: 상위 레벨(Level 2, 3) 시드의 구성 요소로 활용

### 수용 기준

- **Exactness**: AMI/ARI ≥ 0.85 (구조·인과 태스크)
- **Latency**: < 10ms (CPU 기준)
- **Robustness**: 노이즈/변동에 성능 편차 < 15%
- **Bit Depth**: INT8/FP8 양자화 지원

## 구현 완료 시드 (Phase 1)

### SEED-M01 — Hierarchy Builder

**Category**: Relation  
**Params**: ~500K  
**Composed From**: A05 (Grouping Nucleus) + A08 (Binary Comparator) + A07 (Scale Normalizer)

상하/포함 관계를 파악하여 트리 또는 DAG(Directed Acyclic Graph) 구조를 구축합니다.

#### 주요 기능

- 노드 간 계층 관계 행렬 구축
- 트리 구조 추출 (인접 행렬, 레벨, 루트)
- 계층 정보 인코딩
- BFS 기반 레벨 계산

#### 사용 예제

```python
from seeds.molecular import HierarchyBuilder

# 시드 생성
hierarchy_builder = HierarchyBuilder(input_dim=128, num_clusters=16)

# 노드 특징 입력 [B, N, D]
nodes = torch.randn(4, 20, 128)  # 배치 4, 노드 20개

# Forward pass
hierarchy_features = hierarchy_builder(nodes)

# 트리 구조 추출
tree_structure = hierarchy_builder.get_tree_structure(nodes)
print(f"Adjacency matrix: {tree_structure['adjacency_matrix'].shape}")
print(f"Levels: {tree_structure['levels'].shape}")
print(f"Number of roots: {tree_structure['num_roots']}")
```

---

### SEED-M02 — Causality Detector

**Category**: Temporal/Logic  
**Params**: ~600K  
**Composed From**: A06 (Sequence Tracker) + A03 (Recurrence Spotter) + A08 (Binary Comparator)

시간적 선후 관계와 개입 효과를 기반으로 인과 구조를 추정합니다.

#### 주요 기능

- 시간적 패턴 추적
- 반복 패턴 검출 (주기성)
- 선후 관계 분석
- 인과 그래프 (DAG) 추정
- Granger 인과성 테스트
- 개입 효과 추정

#### 사용 예제

```python
from seeds.molecular import CausalityDetector

# 시드 생성
causality_detector = CausalityDetector(input_dim=128)

# 시계열 데이터 입력 [B, T, D]
timeseries = torch.randn(4, 50, 128)  # 배치 4, 시간 50

# Forward pass
causal_features = causality_detector(timeseries)

# 인과 그래프 추정
causal_graph = causality_detector.estimate_causal_graph(timeseries)
print(f"Causal graph: {causal_graph.shape}")  # [B, D, D]

# 개입 효과 포함
interventions = torch.randn(4, 50, 128)
causal_features_with_intervention = causality_detector(
    timeseries, 
    context={'interventions': interventions}
)
```

---

### SEED-M04 — Spatial Transformer

**Category**: Spatial  
**Params**: ~450K  
**Composed From**: A02 (Symmetry Detector) + A07 (Scale Normalizer) + A01 (Edge Detector)

회전, 스케일, 평행이동 등의 공간 변환을 수행하여 입력을 정규 좌표계로 정렬합니다.

#### 주요 기능

- 아핀 변환 (평행이동, 회전, 스케일, 전단)
- 변환 파라미터 자동 추정
- 정규 좌표계로 정렬
- 등변성 보장 인코딩
- 역변환 지원

#### 사용 예제

```python
from seeds.molecular import SpatialTransformer

# 시드 생성
spatial_transformer = SpatialTransformer(input_dim=128)

# 입력 특징 [B, L, D]
features = torch.randn(4, 32, 128)

# Forward pass (자동 정렬)
transformed = spatial_transformer(features)

# 정규 좌표계로 명시적 정렬
aligned, params = spatial_transformer.align_to_canonical(features)
print(f"Translation: {params['translation']}")
print(f"Rotation: {params['rotation']}")
print(f"Scale: {params['scale']}")

# 변환 파라미터만 추정
transform_params = spatial_transformer.estimate_transformation(features)

# 역변환
original = spatial_transformer.inverse_transform(aligned, transform_params)
```

---

## 조합 패턴

Level 1 시드는 다음 세 가지 조합 패턴을 사용합니다.

### 1. 순차적 조합 (Sequential Composition)

```
Input → Seed_A → Seed_B → Output
```

한 시드의 출력이 다음 시드의 입력으로 전달됩니다.

### 2. 병렬 조합 (Parallel Composition)

```
        ┌─ Seed_A ─┐
Input ──┼─ Seed_B ─┼─→ Fusion → Output
        └─ Seed_C ─┘
```

여러 시드가 동일한 입력을 병렬로 처리하고 결과를 융합합니다.

### 3. 계층적 조합 (Hierarchical Composition)

```
Input → Seed_A → Intermediate
              ↓
        Seed_B → Seed_C → Output
```

상위 시드가 하위 시드를 제어하거나 가이드합니다.

---

## 구현 예정 시드

### Phase 2
- **M03**: Pattern Completer (A03 + A06 + A01)
- **M06**: Context Integrator (A06 + M01 + A05)

### Phase 3
- **M05**: Concept Crystallizer (A05 + M03 + M01)
- **M07**: Analogy Mapper (M01 + A08 + M05)

### Phase 4
- **M08**: Conflict Resolver (A08 + M06 + M02)

---

## 파라미터 통계

| 시드 ID | 시드명 | 목표 파라미터 | 실제 파라미터 | 상태 |
|---|---|---|---|---|
| M01 | Hierarchy Builder | ~500K | - | ✓ 완료 |
| M02 | Causality Detector | ~600K | - | ✓ 완료 |
| M04 | Spatial Transformer | ~450K | - | ✓ 완료 |
| M03 | Pattern Completer | ~550K | - | 예정 |
| M06 | Context Integrator | ~650K | - | 예정 |
| M05 | Concept Crystallizer | ~700K | - | 예정 |
| M07 | Analogy Mapper | ~600K | - | 예정 |
| M08 | Conflict Resolver | ~550K | - | 예정 |

---

## 참고 문헌

- 표준 인지 시드 설계 가이드 v1.1 (2025-10-20)
- LEVEL1_IMPLEMENTATION_GUIDE.md
- 작성: 체시(Chesi) · 협업: 제로(Zero)

---

**구현 완료일**: 2025-10-21 (Phase 1)  
**다음 단계**: Phase 2 구현 (M03, M06)

