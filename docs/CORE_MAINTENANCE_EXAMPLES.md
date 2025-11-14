# 메인 코어 유지보수 예제 모음

**대상 범위**: `core/` 모듈 (SeedRegistry, SeedRouter, CompositionEngine, CacheManager, MetricsCollector)

본 문서는 인지 시드 코어의 **오류 진단**, **상시 점검**, **업데이트**에 바로 활용할 수 있는 예제들을 제공합니다. 모든 예제는 `core` 패키지의 공개 API만 사용하며, 실제 운영 환경에 적용하기 위한 최소 골격을 제공합니다.

## 1. 오류 대응 (Error Handling)

### 1.1 레지스트리 무결성 검사 및 누락 의존성 감지

레지스트리에 등록된 시드 중 의존성이 누락된 항목을 찾아내고, 조합 실행 전에 예외를 발생시켜 안전하게 중단합니다.

```python
from core import SeedRegistry, SeedMetadata, CompositionEngine
import torch

registry = SeedRegistry()
registry.register(
    "A01_EdgeDetector",
    seed_module=lambda x: x,  # 더미 시드
    metadata=SeedMetadata(
        name="A01_EdgeDetector",
        level=0,
        version="1.0.0",
        description="경계 검출",
        dependencies=["A99_Unknown"]  # 존재하지 않는 의존성
    ),
)

engine = CompositionEngine(registry)

try:
    engine.build_graph(["A01_EdgeDetector"])
except KeyError as exc:
    print(f"⚠️ 의존성 오류 감지: {exc}")
```

> **Tip**: 위 흐름을 CI 단계에 추가하면, 신규 시드 등록 시 의존성 누락을 빠르게 발견할 수 있습니다. `CompositionEngine.build_graph()`는 내부적으로 `SeedRegistry.get_dependencies()`를 호출하므로, 의존성이 없거나 잘못된 경우 `KeyError`를 그대로 전달합니다.【F:core/composition.py†L80-L122】【F:core/registry.py†L118-L166】

### 1.2 캐시 메모리 압력 모니터링

캐시가 메모리 제한을 초과해도 자동으로 LRU 제거가 발생하는지 확인하고, 초과 시 경고 로그를 발생시킵니다.

```python
from core import CacheManager
import torch

cache = CacheManager(max_size=2, max_memory_mb=0.001)  # 약 1KB 제한

def make_tensor():
    return torch.ones(256, dtype=torch.float32)  # 약 1KB

for idx in range(4):
    key = f"seed-{idx}"
    cache.set(key, make_tensor())
    stats = cache.get_stats()
    if stats["memory_mb"] > stats["max_memory_mb"]:
        print("🚨 캐시 메모리 초과")
```

`CacheManager.set()`은 항목 추가 전에 용량을 확인하고, 조건을 만족할 때까지 `_evict_oldest()`를 반복 호출합니다. 위와 같이 작은 메모리 한도를 설정하면 제거 로직이 정상 동작하는지 쉽게 검증할 수 있습니다.【F:core/cache.py†L38-L113】

## 2. 점검 시나리오 (Routine Checks)

### 2.1 라우터 선택 결과 점검 리포트

라우터가 출력한 확률 상위 시드를 모니터링하여 비정상적인 확률 패턴(예: 특정 시드의 확률 고정화)을 감지합니다.

```python
from core import SeedRegistry, SeedMetadata, SeedRouter
import torch

registry = SeedRegistry()
for idx in range(4):
    name = f"A0{idx+1}_Dummy"
    registry.register(
        name,
        seed_module=torch.nn.Identity(),
        metadata=SeedMetadata(
            name=name,
            level=0,
            version="1.0.0",
            description="테스트 시드"
        )
    )

router = SeedRouter(registry, hidden_dim=32, vocab_size=128, input_dim=16)

task_tokens = torch.randint(0, 128, (1, 10))
input_features = torch.randn(1, 16)

ranked = router.explain_selection(task_tokens, input_features)
for name, prob in ranked[:3]:
    print(f"{name}: {prob:.3f}")
```

`SeedRouter.explain_selection()`은 확률을 내림차순으로 정렬하여 반환하므로, 특정 시드가 항상 1.0에 고정되는 현상 등을 손쉽게 식별할 수 있습니다. 주기적으로 결과를 로깅하여 이상 탐지 룰을 추가하면 점검 자동화를 구현할 수 있습니다.【F:core/router.py†L113-L195】

### 2.2 메트릭 요약 리포트 출력

정기 점검 시 `MetricsCollector.print_summary()`를 호출하여 실행 통계를 확인합니다.

```python
from core import MetricsCollector

metrics = MetricsCollector()
metrics.start_execution("exec-001", ["A01", "A02"])
metrics.record_seed_execution("A01", 0.012)
metrics.record_seed_execution("A02", 0.045, cache_hit=True)
metrics.end_execution()
metrics.print_summary()
```

요약에는 전체 실행 횟수, 평균 실행 시간, 상위 사용 시드 목록 등이 포함되므로, SLA 초과 또는 병목 시드를 빠르게 파악할 수 있습니다.【F:core/metrics.py†L17-L190】

## 3. 업데이트 시나리오 (Update Workflows)

### 3.1 시드 버전 업그레이드 및 메타데이터 교체

새 버전을 도입할 때는 `unregister()`와 `register()` 조합을 사용하여 원자적 교체를 수행합니다.

```python
from core import SeedRegistry, SeedMetadata
import torch

class OldSeed(torch.nn.Module):
    def forward(self, x):
        return x * 0.9

class NewSeed(torch.nn.Module):
    def forward(self, x):
        return x * 1.1

registry = SeedRegistry()
registry.register(
    "M03_PatternCompleter",
    OldSeed(),
    SeedMetadata(
        name="M03_PatternCompleter",
        level=1,
        version="1.0.0",
        description="패턴 보간"
    )
)

registry.unregister("M03_PatternCompleter")
registry.register(
    "M03_PatternCompleter",
    NewSeed(),
    SeedMetadata(
        name="M03_PatternCompleter",
        level=1,
        version="1.1.0",
        description="패턴 보간 (성능 개선)",
        tags=["update", "stable"]
    )
)
```

`SeedRegistry.register()`는 동일 이름이 이미 존재하면 `ValueError`를 발생시키므로, 교체 전 반드시 `unregister()`로 제거해야 합니다. 버전과 태그를 업데이트하면 추후 감사 로그와 호환성 분석에 유리합니다.【F:core/registry.py†L45-L116】

### 3.2 라우터 재학습 후 가중치 적용

학습 파이프라인에서 얻은 새로운 파라미터를 로딩하여 즉시 배포합니다.

```python
from core import SeedRouter, SeedRegistry, SeedMetadata
import torch

registry = SeedRegistry()
registry.register(
    "A01_EdgeDetector",
    torch.nn.Identity(),
    SeedMetadata(
        name="A01_EdgeDetector",
        level=0,
        version="1.1.0",
        description="경계 검출"
    )
)

router = SeedRouter(registry, hidden_dim=64, vocab_size=256, input_dim=32)
state_dict = torch.load("router_checkpoint.pt", map_location="cpu")
router.load_state_dict(state_dict)
print("✅ 라우터 파라미터 업데이트 완료")
```

`SeedRouter`는 표준 PyTorch `nn.Module`이므로 `state_dict` 기반 가중치 교체가 가능합니다. 업데이트 직후 `explain_selection()`을 호출해 주요 시드 확률 분포를 검증하는 것이 좋습니다.【F:core/router.py†L22-L195】

---

### 운용 팁

- **사전 시뮬레이션**: 운영 반영 전, 위 예제를 로컬 또는 스테이징 환경에서 실행하여 로그와 메트릭을 반드시 확인하세요.
- **자동화**: CI/CD 파이프라인에 오류/점검 스크립트를 편성하면 회귀 버그를 예방할 수 있습니다.
- **버전 관리**: `SeedMetadata.version`과 `SeedRegistry.list_all()`을 활용해 변경 이력을 주기적으로 기록하세요.

이 문서는 메인 코어 컴포넌트만을 다루며, 시드 구현(`seeds/`)이나 예제(`examples/`) 모듈은 포함하지 않습니다.
