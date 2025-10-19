"""
Cache Manager Module

시드 실행 결과를 캐싱하여 반복 계산을 방지합니다.
"""

from typing import Optional, Any
from collections import OrderedDict
import torch
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """LRU 캐시 기반 캐시 관리자"""
    
    def __init__(self, max_size: int = 1024, max_memory_mb: float = 512.0):
        """
        Args:
            max_size: 최대 캐시 항목 수
            max_memory_mb: 최대 메모리 사용량 (MB)
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.current_memory = 0
        
        # 통계
        self.hits = 0
        self.misses = 0
        
        logger.info(f"Cache Manager initialized (max_size={max_size}, max_memory={max_memory_mb}MB)")
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        캐시에서 값 가져오기
        
        Args:
            key: 캐시 키
        
        Returns:
            캐시된 값 (없으면 None)
        """
        if key in self.cache:
            # LRU: 최근 사용 항목을 끝으로 이동
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        else:
            self.misses += 1
            logger.debug(f"Cache miss: {key}")
            return None
    
    def set(self, key: str, value: torch.Tensor) -> None:
        """
        캐시에 값 저장
        
        Args:
            key: 캐시 키
            value: 저장할 값
        """
        # 이미 있으면 업데이트
        if key in self.cache:
            old_value = self.cache[key]
            old_size = self._get_tensor_size(old_value)
            self.current_memory -= old_size
            del self.cache[key]
        
        # 새 값 크기 계산
        new_size = self._get_tensor_size(value)
        
        # 메모리 또는 크기 제한 초과 시 오래된 항목 제거
        while (
            (len(self.cache) >= self.max_size or 
             self.current_memory + new_size > self.max_memory_bytes)
            and len(self.cache) > 0
        ):
            self._evict_oldest()
        
        # 새 값 추가
        self.cache[key] = value.detach().clone()  # 복사본 저장
        self.current_memory += new_size
        
        logger.debug(f"Cache set: {key} (size={new_size/1024:.2f}KB)")
    
    def _evict_oldest(self) -> None:
        """가장 오래된 항목 제거 (LRU)"""
        if len(self.cache) == 0:
            return
        
        oldest_key, oldest_value = self.cache.popitem(last=False)
        size = self._get_tensor_size(oldest_value)
        self.current_memory -= size
        
        logger.debug(f"Evicted: {oldest_key} (size={size/1024:.2f}KB)")
    
    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """텐서의 메모리 크기 계산 (바이트)"""
        return tensor.element_size() * tensor.nelement()
    
    def clear(self) -> None:
        """캐시 전체 삭제"""
        self.cache.clear()
        self.current_memory = 0
        logger.info("Cache cleared")
    
    def remove(self, key: str) -> bool:
        """
        특정 키 제거
        
        Args:
            key: 캐시 키
        
        Returns:
            제거 성공 여부
        """
        if key in self.cache:
            value = self.cache[key]
            size = self._get_tensor_size(value)
            self.current_memory -= size
            del self.cache[key]
            logger.debug(f"Removed: {key}")
            return True
        return False
    
    def get_stats(self) -> dict:
        """
        캐시 통계 반환
        
        Returns:
            통계 딕셔너리
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_mb": self.current_memory / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }
    
    def __contains__(self, key: str) -> bool:
        """캐시 키 존재 여부 확인"""
        return key in self.cache
    
    def __len__(self) -> int:
        """캐시 항목 수 반환"""
        return len(self.cache)
    
    def __repr__(self) -> str:
        """캐시 문자열 표현"""
        stats = self.get_stats()
        return (
            f"CacheManager(size={stats['size']}/{stats['max_size']}, "
            f"memory={stats['memory_mb']:.2f}/{stats['max_memory_mb']:.2f}MB, "
            f"hit_rate={stats['hit_rate']:.2%})"
        )

