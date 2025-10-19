# 인지 시드 코어 아키텍처 설계 가이드 v1.1

**문서 버전**: 1.1  
**작성일**: 2025-10-20  
**작성자**: 누스양 (Manus AI Agent)

---

## 1. 개요 (Overview)

이 문서는 **Cognitive Seed Framework**의 핵심인 **코어 아키텍처(Core Architecture)**의 설계 철학, 구성 요소, 데이터 흐름, 그리고 구현 지침을 기술합니다. 본 코어는 32개의 인지 시드(Cognitive Seeds)를 동적으로 조합하여 복잡한 태스크를 해결하는 것을 목표로 합니다.

### 1.1 설계 목표

- **모듈성 (Modularity)**: 각 시드는 독립적으로 개발, 테스트, 배포될 수 있어야 합니다.
- **조합성 (Compositionality)**: 여러 시드를 조합하여 새로운 능력을 창발할 수 있어야 합니다.
- **동적 라우팅 (Dynamic Routing)**: 입력과 태스크에 따라 최적의 시드 조합을 동적으로 결정해야 합니다.
- **효율성 (Efficiency)**: 불필요한 계산을 최소화하고, 중간 결과를 캐싱하여 추론 속도를 높여야 합니다.
- **확장성 (Scalability)**: 새로운 시드를 쉽게 추가하고, 프레임워크를 확장할 수 있어야 합니다.

### 1.2 참조 연구

본 설계는 다음 핵심 연구들에 깊은 영감을 받았습니다:

- **Dynamic Neural Networks: A Survey** (Han et al., 2021) [1]
- **Neural Module Networks** (Andreas et al., 2015) [2]

## 2. 코어 아키텍처 (Core Architecture)

인지 시드 코어는 5개의 주요 컴포넌트로 구성됩니다.

![Core Architecture Diagram](https://i.imgur.com/example.png)  
*(추후 다이어그램 이미지 생성 후 경로 삽입 예정)*

### 2.1 구성 요소

| 컴포넌트 | 기능 | 참조 연구 | 핵심 기술 |
|---|---|---|---|
| **Seed Registry** | 32개 시드의 등록, 메타데이터 관리, 검색 | - | 딕셔너리, 메타데이터 관리 |
| **Seed Router** | 입력/태스크 분석 후 실행할 시드 조합 결정 | Dynamic Routing [1] | Gating Network, Policy Network |
| **Composition Engine** | 시드 조합을 실행 가능한 계산 그래프로 변환 | NMN [2] | Directed Acyclic Graph (DAG) |
| **Cache Manager** | 시드 실행의 중간/최종 결과 캐싱 | - | LRU Cache, 해시 테이블 |
| **Metrics Collector**| 성능(정확도, 지연시간) 및 실행 통계 수집 | - | 모니터링, 로깅 |

### 2.2 데이터 흐름 (Data Flow)

1.  **입력**: 사용자로부터 태스크 설명과 입력 데이터가 들어옵니다.
2.  **라우팅**: `Seed Router`가 태스크를 분석하여 필요한 시드 목록을 `Seed Registry`에서 조회하고 선택합니다.
3.  **조합**: `Composition Engine`이 선택된 시드들의 의존성을 분석하여 실행 계획(DAG)을 수립합니다.
4.  **실행**: 엔진이 DAG에 따라 시드를 순차적/병렬적으로 실행합니다. `Cache Manager`를 통해 캐시된 결과를 확인하고, 없으면 시드를 실행한 후 결과를 캐시에 저장합니다.
5.  **결과**: 최종 시드의 출력이 사용자에게 반환됩니다.
6.  **모니터링**: `Metrics Collector`가 전 과정의 성능 지표를 기록합니다.

## 3. 컴포넌트 상세 설계

### 3.1 Seed Registry

시드 정보를 관리하는 중앙 저장소입니다.

#### 3.1.1 메타데이터 스키마

각 시드는 다음 메타데이터를 포함하여 등록됩니다:

- `name` (str): 시드의 고유 이름 (예: `A01_Boundary_Detector`)
- `level` (int): 시드 레벨 (0: Atomic, 1: Molecular, 2: Cellular, 3: Tissue)
- `version` (str): 시드 버전 (예: `1.0.0`)
- `description` (str): 기능 설명
- `dependencies` (List[str]): 의존하는 하위 시드 이름 목록
- `geometry` (List[str]): 선호 기하학 (`E`, `H`, `S`)
- `bitwidth` (str): 권장 비트폭 (`INT8`, `FP8`, `FP16`)
- `tags` (List[str]): 검색을 위한 태그 (예: `vision`, `spatial`, `attention`)

#### 3.1.2 API 설계

```python
class SeedRegistry:
    def __init__(self):
        self.seeds = {}

    def register(self, name: str, seed_module: nn.Module, metadata: dict):
        # 시드 등록
        pass

    def get(self, name: str) -> nn.Module:
        # 이름으로 시드 모듈 가져오기
        pass

    def query(self, criteria: dict) -> List[dict]:
        # 조건으로 시드 메타데이터 검색
        pass
```

### 3.2 Seed Router

가장 지능적인 컴포넌트로, 동적 실행의 핵심입니다.

#### 3.2.1 아키텍처

- **Task Encoder**: 태스크 설명(자연어)을 벡터 임베딩으로 변환 (e.g., BERT, Sentence-BERT)
- **Input Analyzer**: 입력 데이터의 특징 추출 (e.g., ResNet, ViT)
- **Gating Network**: Task/Input 임베딩을 입력받아 각 시드의 활성화 확률을 출력하는 MLP 또는 어텐션 기반 네트워크

#### 3.2.2 API 설계

```python
class SeedRouter(nn.Module):
    def __init__(self, registry: SeedRegistry):
        self.registry = registry
        # Task Encoder, Input Analyzer, Gating Network 초기화
        pass

    def forward(self, task_description: str, input_data: Tensor) -> List[str]:
        # 1. 태스크와 입력 분석
        # 2. Gating Network로 시드 선택 확률 계산
        # 3. 확률 기반으로 실행할 시드 이름 목록 반환 (e.g., top-k, thresholding)
        pass
```

### 3.3 Composition Engine

선택된 시드들로 실행 가능한 워크플로우를 구성합니다.

#### 3.3.1 DAG 생성

1.  `Seed Router`가 선택한 시드 목록을 받습니다.
2.  `Seed Registry`에서 각 시드의 `dependencies`를 재귀적으로 조회하여 전체 의존성 트리를 구성합니다.
3.  중복을 제거하고 의존성 방향에 따라 엣지를 연결하여 DAG를 생성합니다.

#### 3.3.2 API 설계

```python
class CompositionEngine:
    def __init__(self, registry: SeedRegistry, cache_manager: CacheManager):
        pass

    def execute(self, selected_seeds: List[str], input_data: Tensor) -> Tensor:
        # 1. 의존성 분석 및 DAG 생성
        # 2. DAG 위상 정렬(topological sort)로 실행 순서 결정
        # 3. 순서에 따라 시드 실행 (캐시 확인 포함)
        # 4. 최종 결과 반환
        pass
```

### 3.4 Cache Manager

반복 계산을 방지하여 효율성을 높입니다.

#### 3.4.1 캐시 키 생성

`캐시 키 = hash(시드 이름 + 시드 버전 + 입력 데이터의 해시)`

- 입력 데이터의 해시는 전체 데이터가 아닌, 핵심 특징이나 파일 경로 등으로 대체하여 효율화할 수 있습니다.

#### 3.4.2 API 설계

```python
from functools import lru_cache

class CacheManager:
    def __init__(self, max_size=1024):
        self.cache = lru_cache(maxsize=max_size)

    def get(self, key: str) -> Optional[Tensor]:
        return self.cache.get(key)

    def set(self, key: str, value: Tensor):
        self.cache[key] = value
```

## 4. 구현 및 통합 가이드

### 4.1 시드 모듈 구현

모든 시드는 `torch.nn.Module`을 상속하고, 표준화된 인터페이스를 따라야 합니다.

```python
import torch.nn as nn

class BaseSeed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

# 예시: A01 경계 탐지 시드
class A01_Boundary_Detector(BaseSeed):
    def __init__(self, config):
        super().__init__(config)
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        return torch.sigmoid(self.conv(image))
```

### 4.2 코어 초기화 및 사용

```python
# 1. 코어 컴포넌트 초기화
registry = SeedRegistry()
cache = CacheManager()
router = SeedRouter(registry)
engine = CompositionEngine(registry, cache)

# 2. 시드 등록
# ... (32개 시드 인스턴스화 및 메타데이터와 함께 등록)

# 3. 추론 실행
task = "이미지에서 고양이의 위치를 찾아주세요."
image = load_image("cat.jpg")

# 라우터가 시드 선택
selected_seeds = router(task, image)

# 엔진이 조합 및 실행
result = engine.execute(selected_seeds, image)
```

## 5. 학습 전략

코어, 특히 `Seed Router`의 학습은 다음 3단계 하이브리드 전략을 권장합니다.

1.  **개별 시드 사전학습**: 각 시드를 해당 기능에 맞는 데이터셋으로 독립적으로 학습합니다.
2.  **라우터 학습**: 사전학습된 시드는 고정한 채, `Seed Router`의 Gating Network를 학습합니다. 라우팅 결정에 대한 보상은 최종 태스크 성능과 계산 비용을 조합하여 설정합니다 (강화학습 또는 Gumbel-Softmax 활용).
3.  **End-to-End 미세조정**: 전체 시스템(시드 + 라우터)을 소량의 학습률로 공동 미세조정합니다.

## 6. 버전 관리 및 향후 계획

- **버전**: v1.1
- **다음 단계**:
    - 각 컴포넌트의 Python 클래스 구현
    - 32개 시드 기본 뼈대(skeleton) 코드 작성
    - Level 0 시드 3개 (A01, A05, A07) 우선 구현 및 테스트
    - 코어 아키텍처의 성능을 검증하기 위한 기본 벤치마크 수립

---

## 7. 참고문헌

[1] Han, Y., Huang, G., Song, S., Yang, L., Wang, H., & Wang, Y. (2021). Dynamic Neural Networks: A Survey. *arXiv preprint arXiv:2102.04906*.

[2] Andreas, J., Rohrbach, M., Darrell, T., & Klein, D. (2015). Neural Module Networks. *arXiv preprint arXiv:1511.02799*.

