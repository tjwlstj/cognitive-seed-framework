"""
Composition Engine Module

선택된 시드들을 조합하여 실행 가능한 계산 그래프(DAG)를 생성하고 실행합니다.
"""

from typing import List, Dict, Any, Optional, Set
from collections import deque
import torch
import logging

from .registry import SeedRegistry
from .cache import CacheManager

logger = logging.getLogger(__name__)


class CompositionGraph:
    """시드 조합을 나타내는 DAG (Directed Acyclic Graph)"""
    
    def __init__(self):
        """그래프 초기화"""
        self.nodes: Set[str] = set()
        self.edges: Dict[str, List[str]] = {}  # node -> [dependencies]
        self.in_degree: Dict[str, int] = {}
    
    def add_node(self, node: str) -> None:
        """노드 추가"""
        if node not in self.nodes:
            self.nodes.add(node)
            self.edges[node] = []
            self.in_degree[node] = 0
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """
        엣지 추가 (from_node는 to_node의 의존성)
        
        Args:
            from_node: 의존성 시드
            to_node: 대상 시드
        """
        self.add_node(from_node)
        self.add_node(to_node)
        
        if from_node not in self.edges[to_node]:
            self.edges[to_node].append(from_node)
            self.in_degree[to_node] += 1
    
    def topological_sort(self) -> List[str]:
        """
        위상 정렬로 실행 순서 결정
        
        Returns:
            실행 순서대로 정렬된 시드 이름 리스트
        
        Raises:
            ValueError: 순환 의존성이 있는 경우
        """
        # Kahn's Algorithm
        in_degree_copy = self.in_degree.copy()
        queue = deque([node for node in self.nodes if in_degree_copy[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # 이 노드에 의존하는 노드들의 in_degree 감소
            for dependent in self.nodes:
                if node in self.edges[dependent]:
                    in_degree_copy[dependent] -= 1
                    if in_degree_copy[dependent] == 0:
                        queue.append(dependent)
        
        if len(result) != len(self.nodes):
            raise ValueError("Circular dependency detected in seed composition")
        
        return result
    
    def get_dependencies(self, node: str) -> List[str]:
        """노드의 직접 의존성 반환"""
        return self.edges.get(node, [])
    
    def __repr__(self) -> str:
        return f"CompositionGraph(nodes={len(self.nodes)}, edges={sum(len(v) for v in self.edges.values())})"


class CompositionEngine:
    """조합 엔진 메인 클래스"""
    
    def __init__(
        self,
        registry: SeedRegistry,
        cache_manager: Optional[CacheManager] = None
    ):
        """
        Args:
            registry: 시드 레지스트리
            cache_manager: 캐시 관리자 (선택적)
        """
        self.registry = registry
        self.cache_manager = cache_manager
        logger.info("Composition Engine initialized")
    
    def build_graph(self, selected_seeds: List[str]) -> CompositionGraph:
        """
        선택된 시드들로부터 실행 그래프 생성
        
        Args:
            selected_seeds: 실행할 시드 이름 리스트
        
        Returns:
            CompositionGraph 객체
        """
        graph = CompositionGraph()
        
        # 모든 시드와 의존성을 그래프에 추가
        visited = set()
        
        def add_seed_and_dependencies(seed_name: str):
            if seed_name in visited:
                return
            visited.add(seed_name)
            
            # 노드 추가
            graph.add_node(seed_name)
            
            # 의존성 추가
            dependencies = self.registry.get_dependencies(seed_name, recursive=False)
            for dep in dependencies:
                add_seed_and_dependencies(dep)
                graph.add_edge(dep, seed_name)
        
        for seed_name in selected_seeds:
            add_seed_and_dependencies(seed_name)
        
        logger.debug(f"Built {graph}")
        return graph
    
    def execute(
        self,
        selected_seeds: List[str],
        input_data: torch.Tensor,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        시드 조합 실행
        
        Args:
            selected_seeds: 실행할 시드 이름 리스트
            input_data: 입력 데이터
            context: 추가 컨텍스트 정보
        
        Returns:
            최종 출력
        """
        if context is None:
            context = {}
        
        # 1. 실행 그래프 생성
        graph = self.build_graph(selected_seeds)
        
        # 2. 위상 정렬로 실행 순서 결정
        execution_order = graph.topological_sort()
        logger.info(f"Execution order: {execution_order}")
        
        # 3. 중간 결과 저장소
        intermediate_results: Dict[str, torch.Tensor] = {}
        
        # 4. 순서대로 시드 실행
        for seed_name in execution_order:
            # 캐시 확인
            cache_key = self._compute_cache_key(seed_name, input_data, intermediate_results, graph)
            
            if self.cache_manager and cache_key in self.cache_manager:
                output = self.cache_manager.get(cache_key)
                logger.debug(f"Cache hit for {seed_name}")
            else:
                # 시드 실행
                seed_module = self.registry.get(seed_name)
                
                # 의존성 결과 수집
                dependencies = graph.get_dependencies(seed_name)
                if dependencies:
                    # 의존성이 있으면 그 결과를 입력으로 사용
                    dep_outputs = [intermediate_results[dep] for dep in dependencies]
                    if len(dep_outputs) == 1:
                        seed_input = dep_outputs[0]
                    else:
                        # 여러 의존성 결과를 결합 (간단히 concat)
                        seed_input = torch.cat(dep_outputs, dim=-1)
                else:
                    # 의존성이 없으면 원본 입력 사용
                    seed_input = input_data
                
                # 시드 forward
                output = seed_module(seed_input)
                logger.debug(f"Executed {seed_name}")
                
                # 캐시 저장
                if self.cache_manager:
                    self.cache_manager.set(cache_key, output)
            
            # 중간 결과 저장
            intermediate_results[seed_name] = output
        
        # 5. 최종 결과 반환 (마지막 시드의 출력)
        final_seed = selected_seeds[-1] if selected_seeds else execution_order[-1]
        final_output = intermediate_results[final_seed]
        
        logger.info(f"Composition execution completed")
        return final_output
    
    def _compute_cache_key(
        self,
        seed_name: str,
        input_data: torch.Tensor,
        intermediate_results: Dict[str, torch.Tensor],
        graph: CompositionGraph
    ) -> str:
        """
        캐시 키 계산
        
        Args:
            seed_name: 시드 이름
            input_data: 입력 데이터
            intermediate_results: 중간 결과
            graph: 실행 그래프
        
        Returns:
            캐시 키 문자열
        """
        metadata = self.registry.get_metadata(seed_name)
        
        # 시드 이름 + 버전
        key_parts = [seed_name, metadata.version]
        
        # 의존성 결과 해시
        dependencies = graph.get_dependencies(seed_name)
        if dependencies:
            for dep in dependencies:
                if dep in intermediate_results:
                    result_hash = hash(intermediate_results[dep].detach().cpu().numpy().tobytes())
                    key_parts.append(str(result_hash))
        else:
            # 원본 입력 해시
            input_hash = hash(input_data.detach().cpu().numpy().tobytes())
            key_parts.append(str(input_hash))
        
        cache_key = ":".join(key_parts)
        return cache_key
    
    def visualize_graph(self, selected_seeds: List[str]) -> str:
        """
        실행 그래프를 텍스트로 시각화
        
        Args:
            selected_seeds: 시드 이름 리스트
        
        Returns:
            그래프의 텍스트 표현
        """
        graph = self.build_graph(selected_seeds)
        execution_order = graph.topological_sort()
        
        lines = ["Composition Graph:"]
        lines.append("=" * 50)
        
        for i, seed_name in enumerate(execution_order, 1):
            dependencies = graph.get_dependencies(seed_name)
            metadata = self.registry.get_metadata(seed_name)
            
            lines.append(f"{i}. {seed_name} (Level {metadata.level})")
            if dependencies:
                lines.append(f"   Dependencies: {', '.join(dependencies)}")
            else:
                lines.append("   Dependencies: None (uses input)")
            lines.append("")
        
        return "\n".join(lines)

