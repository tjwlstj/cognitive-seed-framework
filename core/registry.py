"""
Seed Registry Module

32개의 인지 시드를 등록, 관리, 검색하는 중앙 저장소입니다.
"""

from typing import Dict, List, Optional, Any
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SeedMetadata:
    """시드 메타데이터 클래스"""
    
    def __init__(
        self,
        name: str,
        level: int,
        version: str,
        description: str,
        dependencies: List[str] = None,
        geometry: List[str] = None,
        bitwidth: str = "FP16",
        tags: List[str] = None
    ):
        """
        Args:
            name: 시드 고유 이름 (예: A01_Boundary_Detector)
            level: 시드 레벨 (0: Atomic, 1: Molecular, 2: Cellular, 3: Tissue)
            version: 시드 버전 (예: 1.0.0)
            description: 기능 설명
            dependencies: 의존하는 하위 시드 이름 목록
            geometry: 선호 기하학 (E, H, S)
            bitwidth: 권장 비트폭 (INT8, FP8, FP16, BF16)
            tags: 검색용 태그
        """
        self.name = name
        self.level = level
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        self.geometry = geometry or ["E"]  # 기본값: Euclidean
        self.bitwidth = bitwidth
        self.tags = tags or []
    
    def to_dict(self) -> Dict[str, Any]:
        """메타데이터를 딕셔너리로 변환"""
        return {
            "name": self.name,
            "level": self.level,
            "version": self.version,
            "description": self.description,
            "dependencies": self.dependencies,
            "geometry": self.geometry,
            "bitwidth": self.bitwidth,
            "tags": self.tags
        }


class SeedRegistry:
    """시드 레지스트리 클래스"""
    
    def __init__(self):
        """레지스트리 초기화"""
        self.seeds: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical name mapping
        logger.info("Seed Registry initialized")
    
    def register(
        self,
        name: str,
        seed_module: nn.Module,
        metadata: SeedMetadata,
        aliases: List[str] = None
    ) -> None:
        """
        시드를 레지스트리에 등록
        
        Args:
            name: 시드 이름 (canonical name)
            seed_module: 시드 모듈 (torch.nn.Module)
            metadata: 시드 메타데이터
            aliases: 별칭 리스트 (예: ["A01", "SEED-A01"])
        
        Raises:
            ValueError: 이미 등록된 시드인 경우
        """
        if name in self.seeds:
            raise ValueError(f"Seed '{name}' is already registered")
        
        self.seeds[name] = {
            "module": seed_module,
            "metadata": metadata
        }
        
        # 별칭 등록
        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    logger.warning(f"Alias '{alias}' already exists, overwriting")
                self._aliases[alias] = name
        
        logger.info(
            f"Registered seed: {name} (Level {metadata.level}, "
            f"Version {metadata.version})"
        )
    
    def unregister(self, name: str) -> None:
        """
        시드를 레지스트리에서 제거
        
        Args:
            name: 시드 이름
        
        Raises:
            KeyError: 등록되지 않은 시드인 경우
        """
        if name not in self.seeds:
            raise KeyError(f"Seed '{name}' is not registered")
        
        del self.seeds[name]
        logger.info(f"Unregistered seed: {name}")
    
    def _resolve_name(self, name: str) -> str:
        """별칭을 canonical name으로 변환"""
        return self._aliases.get(name, name)
    
    def get(self, name: str) -> nn.Module:
        """
        이름으로 시드 모듈 가져오기
        
        별칭도 지원합니다 (예: "A01", "SEED-A01", "A01_Edge_Detector" 모두 동일)
        
        Args:
            name: 시드 이름 또는 별칭
        
        Returns:
            시드 모듈
        
        Raises:
            KeyError: 등록되지 않은 시드인 경우
        """
        canonical_name = self._resolve_name(name)
        if canonical_name not in self.seeds:
            raise KeyError(f"Seed '{name}' is not registered")
        
        return self.seeds[canonical_name]["module"]
    
    def get_metadata(self, name: str) -> SeedMetadata:
        """
        이름으로 시드 메타데이터 가져오기
        
        Args:
            name: 시드 이름 또는 별칭
        
        Returns:
            시드 메타데이터
        
        Raises:
            KeyError: 등록되지 않은 시드인 경우
        """
        canonical_name = self._resolve_name(name)
        if canonical_name not in self.seeds:
            raise KeyError(f"Seed '{name}' is not registered")
        
        return self.seeds[canonical_name]["metadata"]
    
    def query(
        self,
        level: Optional[int] = None,
        tags: Optional[List[str]] = None,
        geometry: Optional[str] = None,
        bitwidth: Optional[str] = None
    ) -> List[SeedMetadata]:
        """
        조건에 맞는 시드 메타데이터 검색
        
        Args:
            level: 시드 레벨 필터
            tags: 태그 필터 (OR 조건)
            geometry: 기하학 필터
            bitwidth: 비트폭 필터
        
        Returns:
            조건에 맞는 시드 메타데이터 리스트
        """
        results = []
        
        for name, seed_data in self.seeds.items():
            metadata = seed_data["metadata"]
            
            # 레벨 필터
            if level is not None and metadata.level != level:
                continue
            
            # 태그 필터 (OR 조건)
            if tags is not None:
                if not any(tag in metadata.tags for tag in tags):
                    continue
            
            # 기하학 필터
            if geometry is not None and geometry not in metadata.geometry:
                continue
            
            # 비트폭 필터
            if bitwidth is not None and metadata.bitwidth != bitwidth:
                continue
            
            results.append(metadata)
        
        logger.debug(f"Query returned {len(results)} seeds")
        return results
    
    def list_all(self) -> List[str]:
        """
        등록된 모든 시드 이름 목록 반환
        
        Returns:
            시드 이름 리스트
        """
        return list(self.seeds.keys())
    
    def get_dependencies(self, name: str, recursive: bool = False) -> List[str]:
        """
        시드의 의존성 목록 가져오기
        
        Args:
            name: 시드 이름
            recursive: 재귀적으로 모든 하위 의존성 포함 여부
        
        Returns:
            의존하는 시드 이름 리스트
        
        Raises:
            KeyError: 등록되지 않은 시드인 경우
        """
        if name not in self.seeds:
            raise KeyError(f"Seed '{name}' is not registered")
        
        metadata = self.seeds[name]["metadata"]
        dependencies = metadata.dependencies.copy()
        
        if recursive:
            # 재귀적으로 모든 하위 의존성 수집
            all_deps = set(dependencies)
            for dep in dependencies:
                if dep in self.seeds:
                    sub_deps = self.get_dependencies(dep, recursive=True)
                    all_deps.update(sub_deps)
            dependencies = list(all_deps)
        
        return dependencies
    
    def __len__(self) -> int:
        """등록된 시드 개수 반환"""
        return len(self.seeds)
    
    def __contains__(self, name: str) -> bool:
        """시드 등록 여부 확인"""
        return name in self.seeds
    
    def __repr__(self) -> str:
        """레지스트리 문자열 표현"""
        return f"SeedRegistry(seeds={len(self.seeds)})"

