"""
Metrics Collector Module

시드 실행의 성능 지표를 수집하고 모니터링합니다.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import time
import torch
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """성능 지표 수집기"""
    
    def __init__(self):
        """메트릭 수집기 초기화"""
        # 시드별 실행 통계
        self.seed_execution_times: Dict[str, List[float]] = defaultdict(list)
        self.seed_execution_counts: Dict[str, int] = defaultdict(int)
        self.seed_cache_hits: Dict[str, int] = defaultdict(int)
        
        # 전체 실행 통계
        self.total_executions = 0
        self.total_execution_time = 0.0
        
        # 현재 실행 컨텍스트
        self.current_execution: Optional[Dict] = None
        
        logger.info("Metrics Collector initialized")
    
    def start_execution(self, execution_id: str, selected_seeds: List[str]) -> None:
        """
        실행 시작 기록
        
        Args:
            execution_id: 실행 고유 ID
            selected_seeds: 선택된 시드 목록
        """
        self.current_execution = {
            "id": execution_id,
            "seeds": selected_seeds,
            "start_time": time.time(),
            "seed_times": {}
        }
        logger.debug(f"Started execution: {execution_id}")
    
    def record_seed_execution(
        self,
        seed_name: str,
        execution_time: float,
        cache_hit: bool = False
    ) -> None:
        """
        시드 실행 기록
        
        Args:
            seed_name: 시드 이름
            execution_time: 실행 시간 (초)
            cache_hit: 캐시 히트 여부
        """
        self.seed_execution_times[seed_name].append(execution_time)
        self.seed_execution_counts[seed_name] += 1
        
        if cache_hit:
            self.seed_cache_hits[seed_name] += 1
        
        if self.current_execution:
            self.current_execution["seed_times"][seed_name] = execution_time
        
        logger.debug(
            f"Seed {seed_name}: {execution_time*1000:.2f}ms "
            f"(cache_hit={cache_hit})"
        )
    
    def end_execution(self) -> Dict:
        """
        실행 종료 및 통계 반환
        
        Returns:
            실행 통계 딕셔너리
        """
        if not self.current_execution:
            logger.warning("No active execution to end")
            return {}
        
        end_time = time.time()
        total_time = end_time - self.current_execution["start_time"]
        
        self.total_executions += 1
        self.total_execution_time += total_time
        
        stats = {
            "execution_id": self.current_execution["id"],
            "total_time": total_time,
            "num_seeds": len(self.current_execution["seeds"]),
            "seed_times": self.current_execution["seed_times"]
        }
        
        logger.info(
            f"Execution {stats['execution_id']} completed: "
            f"{total_time*1000:.2f}ms ({stats['num_seeds']} seeds)"
        )
        
        self.current_execution = None
        return stats
    
    def get_seed_stats(self, seed_name: str) -> Dict:
        """
        특정 시드의 통계 반환
        
        Args:
            seed_name: 시드 이름
        
        Returns:
            시드 통계 딕셔너리
        """
        times = self.seed_execution_times.get(seed_name, [])
        count = self.seed_execution_counts.get(seed_name, 0)
        cache_hits = self.seed_cache_hits.get(seed_name, 0)
        
        if not times:
            return {
                "seed_name": seed_name,
                "count": 0,
                "avg_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
                "cache_hit_rate": 0.0
            }
        
        return {
            "seed_name": seed_name,
            "count": count,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "cache_hit_rate": cache_hits / count if count > 0 else 0.0
        }
    
    def get_all_stats(self) -> Dict:
        """
        전체 통계 반환
        
        Returns:
            전체 통계 딕셔너리
        """
        seed_stats = {}
        for seed_name in self.seed_execution_counts.keys():
            seed_stats[seed_name] = self.get_seed_stats(seed_name)
        
        return {
            "total_executions": self.total_executions,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": (
                self.total_execution_time / self.total_executions
                if self.total_executions > 0 else 0.0
            ),
            "seed_stats": seed_stats
        }
    
    def get_top_seeds(self, n: int = 10, metric: str = "count") -> List[tuple]:
        """
        상위 N개 시드 반환
        
        Args:
            n: 반환할 시드 개수
            metric: 정렬 기준 ("count", "avg_time", "total_time")
        
        Returns:
            (시드 이름, 값) 튜플의 리스트
        """
        if metric == "count":
            items = list(self.seed_execution_counts.items())
        elif metric == "avg_time":
            items = [
                (name, sum(times) / len(times))
                for name, times in self.seed_execution_times.items()
                if times
            ]
        elif metric == "total_time":
            items = [
                (name, sum(times))
                for name, times in self.seed_execution_times.items()
            ]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]
    
    def reset(self) -> None:
        """모든 통계 초기화"""
        self.seed_execution_times.clear()
        self.seed_execution_counts.clear()
        self.seed_cache_hits.clear()
        self.total_executions = 0
        self.total_execution_time = 0.0
        self.current_execution = None
        logger.info("Metrics reset")
    
    def print_summary(self) -> None:
        """통계 요약 출력"""
        stats = self.get_all_stats()
        
        print("\n" + "=" * 60)
        print("Cognitive Seed Framework - Metrics Summary")
        print("=" * 60)
        print(f"Total Executions: {stats['total_executions']}")
        print(f"Total Time: {stats['total_execution_time']:.3f}s")
        print(f"Avg Time per Execution: {stats['avg_execution_time']*1000:.2f}ms")
        print("\nTop 10 Most Used Seeds:")
        print("-" * 60)
        
        top_seeds = self.get_top_seeds(10, "count")
        for i, (seed_name, count) in enumerate(top_seeds, 1):
            seed_stat = self.get_seed_stats(seed_name)
            print(
                f"{i:2d}. {seed_name:30s} "
                f"Count: {count:4d}  "
                f"Avg: {seed_stat['avg_time']*1000:6.2f}ms  "
                f"Cache Hit: {seed_stat['cache_hit_rate']:5.1%}"
            )
        
        print("=" * 60 + "\n")
    
    def __repr__(self) -> str:
        """메트릭 수집기 문자열 표현"""
        return (
            f"MetricsCollector(executions={self.total_executions}, "
            f"seeds_tracked={len(self.seed_execution_counts)})"
        )

