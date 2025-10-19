"""
Basic Usage Example

코어 아키텍처의 기본 사용법을 보여주는 예제입니다.
"""

import torch
import torch.nn as nn
from core import (
    SeedRegistry,
    SeedMetadata,
    SeedRouter,
    CompositionEngine,
    CacheManager,
    MetricsCollector
)


# 1. 간단한 시드 정의
class BoundaryDetector(nn.Module):
    """A01: 경계 탐지 시드"""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(x))


class FeatureExtractor(nn.Module):
    """A05: 특징 추출 시드"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x


class ObjectDetector(nn.Module):
    """M01: 객체 탐지 시드 (A01, A05에 의존)"""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)  # 10개 클래스
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        pooled = features.mean(dim=[2, 3])
        return self.fc(pooled)


def main():
    print("=" * 60)
    print("Cognitive Seed Framework - Basic Usage Example")
    print("=" * 60)
    
    # 2. 코어 컴포넌트 초기화
    print("\n[1] Initializing core components...")
    registry = SeedRegistry()
    cache = CacheManager(max_size=100)
    metrics = MetricsCollector()
    engine = CompositionEngine(registry, cache)
    
    # 3. 시드 등록
    print("[2] Registering seeds...")
    
    # A01: Boundary Detector
    boundary_detector = BoundaryDetector()
    metadata_a01 = SeedMetadata(
        name="A01_Boundary_Detector",
        level=0,
        version="1.0.0",
        description="Detects boundaries in images",
        geometry=["E"],
        bitwidth="FP16",
        tags=["vision", "edge", "atomic"]
    )
    registry.register("A01_Boundary_Detector", boundary_detector, metadata_a01)
    
    # A05: Feature Extractor
    feature_extractor = FeatureExtractor()
    metadata_a05 = SeedMetadata(
        name="A05_Feature_Extractor",
        level=0,
        version="1.0.0",
        description="Extracts features from images",
        geometry=["E"],
        bitwidth="FP16",
        tags=["vision", "feature", "atomic"]
    )
    registry.register("A05_Feature_Extractor", feature_extractor, metadata_a05)
    
    # M01: Object Detector
    object_detector = ObjectDetector()
    metadata_m01 = SeedMetadata(
        name="M01_Object_Detector",
        level=1,
        version="1.0.0",
        description="Detects objects in images",
        dependencies=["A05_Feature_Extractor"],
        geometry=["E"],
        bitwidth="FP16",
        tags=["vision", "detection", "molecular"]
    )
    registry.register("M01_Object_Detector", object_detector, metadata_m01)
    
    print(f"   Registered {len(registry)} seeds")
    
    # 4. 레지스트리 조회
    print("\n[3] Querying registry...")
    level0_seeds = registry.query(level=0)
    print(f"   Level 0 seeds: {[s.name for s in level0_seeds]}")
    
    vision_seeds = registry.query(tags=["vision"])
    print(f"   Vision seeds: {[s.name for s in vision_seeds]}")
    
    # 5. 조합 그래프 생성 및 시각화
    print("\n[4] Building composition graph...")
    selected_seeds = ["M01_Object_Detector"]
    graph_viz = engine.visualize_graph(selected_seeds)
    print(graph_viz)
    
    # 6. 실행
    print("[5] Executing composition...")
    
    # 더미 입력 (배치 크기 2, 3채널, 224x224 이미지)
    input_image = torch.randn(2, 3, 224, 224)
    
    metrics.start_execution("example_exec", selected_seeds)
    
    import time
    start_time = time.time()
    
    output = engine.execute(selected_seeds, input_image)
    
    execution_time = time.time() - start_time
    
    metrics.record_seed_execution("M01_Object_Detector", execution_time)
    exec_stats = metrics.end_execution()
    
    print(f"   Output shape: {output.shape}")
    print(f"   Execution time: {execution_time*1000:.2f}ms")
    
    # 7. 캐시 테스트 (같은 입력으로 재실행)
    print("\n[6] Testing cache (re-executing with same input)...")
    
    metrics.start_execution("example_exec_cached", selected_seeds)
    
    start_time = time.time()
    output_cached = engine.execute(selected_seeds, input_image)
    execution_time_cached = time.time() - start_time
    
    metrics.record_seed_execution("M01_Object_Detector", execution_time_cached, cache_hit=True)
    metrics.end_execution()
    
    print(f"   Cached execution time: {execution_time_cached*1000:.2f}ms")
    print(f"   Speedup: {execution_time/execution_time_cached:.2f}x")
    
    # 8. 통계 출력
    print("\n[7] Statistics:")
    print(f"   Cache: {cache}")
    print(f"   Metrics: {metrics}")
    
    metrics.print_summary()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

