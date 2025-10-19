"""
Core Components Unit Tests

코어 컴포넌트의 기능을 테스트합니다.
"""

import unittest
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


class DummySeed(nn.Module):
    """테스트용 더미 시드"""
    
    def __init__(self, name: str, output_dim: int = 64):
        super().__init__()
        self.name = name
        self.fc = nn.Linear(64, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class TestSeedRegistry(unittest.TestCase):
    """SeedRegistry 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.registry = SeedRegistry()
    
    def test_register_and_get(self):
        """시드 등록 및 가져오기 테스트"""
        seed = DummySeed("TestSeed")
        metadata = SeedMetadata(
            name="TestSeed",
            level=0,
            version="1.0.0",
            description="Test seed"
        )
        
        self.registry.register("TestSeed", seed, metadata)
        retrieved_seed = self.registry.get("TestSeed")
        
        self.assertIsInstance(retrieved_seed, DummySeed)
        self.assertEqual(retrieved_seed.name, "TestSeed")
    
    def test_query(self):
        """시드 검색 테스트"""
        # 여러 시드 등록
        for i in range(3):
            seed = DummySeed(f"Seed{i}")
            metadata = SeedMetadata(
                name=f"Seed{i}",
                level=i % 2,
                version="1.0.0",
                description=f"Test seed {i}",
                tags=["test", f"level{i%2}"]
            )
            self.registry.register(f"Seed{i}", seed, metadata)
        
        # 레벨 0 시드 검색
        results = self.registry.query(level=0)
        self.assertEqual(len(results), 2)  # Seed0, Seed2
    
    def test_dependencies(self):
        """의존성 조회 테스트"""
        # 의존성 있는 시드 등록
        seed_a = DummySeed("SeedA")
        metadata_a = SeedMetadata(
            name="SeedA",
            level=0,
            version="1.0.0",
            description="Base seed"
        )
        self.registry.register("SeedA", seed_a, metadata_a)
        
        seed_b = DummySeed("SeedB")
        metadata_b = SeedMetadata(
            name="SeedB",
            level=1,
            version="1.0.0",
            description="Dependent seed",
            dependencies=["SeedA"]
        )
        self.registry.register("SeedB", seed_b, metadata_b)
        
        deps = self.registry.get_dependencies("SeedB")
        self.assertEqual(deps, ["SeedA"])


class TestCacheManager(unittest.TestCase):
    """CacheManager 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.cache = CacheManager(max_size=10)
    
    def test_set_and_get(self):
        """캐시 저장 및 가져오기 테스트"""
        key = "test_key"
        value = torch.randn(4, 64)
        
        self.cache.set(key, value)
        retrieved = self.cache.get(key)
        
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.allclose(value, retrieved))
    
    def test_lru_eviction(self):
        """LRU 제거 테스트"""
        # 최대 크기보다 많은 항목 추가
        for i in range(15):
            self.cache.set(f"key{i}", torch.randn(4, 64))
        
        # 가장 오래된 항목들이 제거되었는지 확인
        self.assertIsNone(self.cache.get("key0"))
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNotNone(self.cache.get("key14"))
    
    def test_stats(self):
        """통계 테스트"""
        self.cache.set("key1", torch.randn(4, 64))
        self.cache.get("key1")  # hit
        self.cache.get("key2")  # miss
        
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)


class TestCompositionEngine(unittest.TestCase):
    """CompositionEngine 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.registry = SeedRegistry()
        self.cache = CacheManager()
        self.engine = CompositionEngine(self.registry, self.cache)
        
        # 테스트 시드 등록
        seed_a = DummySeed("SeedA", output_dim=64)
        metadata_a = SeedMetadata(
            name="SeedA",
            level=0,
            version="1.0.0",
            description="Base seed"
        )
        self.registry.register("SeedA", seed_a, metadata_a)
        
        seed_b = DummySeed("SeedB", output_dim=64)
        metadata_b = SeedMetadata(
            name="SeedB",
            level=1,
            version="1.0.0",
            description="Dependent seed",
            dependencies=["SeedA"]
        )
        self.registry.register("SeedB", seed_b, metadata_b)
    
    def test_build_graph(self):
        """그래프 생성 테스트"""
        graph = self.engine.build_graph(["SeedB"])
        
        # SeedB와 의존성 SeedA가 모두 포함되어야 함
        self.assertIn("SeedA", graph.nodes)
        self.assertIn("SeedB", graph.nodes)
    
    def test_topological_sort(self):
        """위상 정렬 테스트"""
        graph = self.engine.build_graph(["SeedB"])
        execution_order = graph.topological_sort()
        
        # SeedA가 SeedB보다 먼저 실행되어야 함
        self.assertEqual(execution_order.index("SeedA"), 0)
        self.assertEqual(execution_order.index("SeedB"), 1)
    
    def test_execute(self):
        """실행 테스트"""
        input_data = torch.randn(4, 64)
        output = self.engine.execute(["SeedB"], input_data)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (4, 64))


class TestMetricsCollector(unittest.TestCase):
    """MetricsCollector 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.metrics = MetricsCollector()
    
    def test_execution_tracking(self):
        """실행 추적 테스트"""
        self.metrics.start_execution("exec1", ["SeedA", "SeedB"])
        self.metrics.record_seed_execution("SeedA", 0.1)
        self.metrics.record_seed_execution("SeedB", 0.2)
        stats = self.metrics.end_execution()
        
        self.assertEqual(stats["num_seeds"], 2)
        self.assertGreater(stats["total_time"], 0)
    
    def test_seed_stats(self):
        """시드 통계 테스트"""
        self.metrics.start_execution("exec1", ["SeedA"])
        self.metrics.record_seed_execution("SeedA", 0.1)
        self.metrics.end_execution()
        
        self.metrics.start_execution("exec2", ["SeedA"])
        self.metrics.record_seed_execution("SeedA", 0.2, cache_hit=True)
        self.metrics.end_execution()
        
        seed_stats = self.metrics.get_seed_stats("SeedA")
        self.assertEqual(seed_stats["count"], 2)
        self.assertAlmostEqual(seed_stats["avg_time"], 0.15)
        self.assertAlmostEqual(seed_stats["cache_hit_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

