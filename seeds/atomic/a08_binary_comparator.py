"""
SEED-A08 — Binary Comparator

대소/동등 비교를 수행하는 원자 시드입니다.
"""

from typing import Optional, Dict, Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig


class BinaryComparator(BaseSeed):
    """
    SEED-A08: Binary Comparator
    
    Category: Logic
    Bit: INT8
    Params: ~96
    Purpose: 대소/동등 비교 원자 연산
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 48):
        config = SeedConfig(
            seed_id="SEED-A08",
            name="Binary Comparator",
            level=0,
            category="Logic",
            bit_depth="INT8",
            params=96,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E"]  # 비교 연산은 유클리드 공간에서 수행
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        
        # 비교 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 비교 연산 네트워크
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3가지 비교 결과: <, =, >
            nn.Softmax(dim=-1)
        )
        
        # 임계값 학습
        self.threshold = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 형태의 입력 텐서
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보
        Returns:
            [B, L, D] 형태의 비교 정보가 인코딩된 텐서
        """
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 특징 추출
        features = self.feature_extractor(x)  # [B, L, hidden_dim]
        
        # 인접 요소 간 비교
        batch_size, seq_len, _ = features.shape
        
        if seq_len > 1:
            # 현재와 다음 요소 페어링
            current = features[:, :-1, :]  # [B, L-1, hidden_dim]
            next_elem = features[:, 1:, :]  # [B, L-1, hidden_dim]
            
            # 페어 결합
            pairs = torch.cat([current, next_elem], dim=-1)  # [B, L-1, hidden_dim*2]
            
            # 비교 수행
            comparison = self.comparator(pairs)  # [B, L-1, 3]
            
            # 첫 번째 요소는 자기 자신과 비교 (동등)
            first_comparison = torch.zeros(batch_size, 1, 3, device=x.device)
            first_comparison[:, :, 1] = 1.0  # 동등
            
            comparison = torch.cat([first_comparison, comparison], dim=1)  # [B, L, 3]
        else:
            # 시퀀스 길이가 1이면 자기 자신과 비교
            comparison = torch.zeros(batch_size, seq_len, 3, device=x.device)
            comparison[:, :, 1] = 1.0  # 동등
        
        # 비교 정보를 원래 차원으로 투영
        comparison_info = torch.matmul(comparison, torch.randn(3, x.size(-1), device=x.device))
        
        # 원본과 비교 정보 결합
        output = x + comparison_info
        
        return output
    
    def compare(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        두 텐서를 비교합니다.
        
        Args:
            a: [B, D] 형태의 첫 번째 텐서
            b: [B, D] 형태의 두 번째 텐서
        Returns:
            [B, 3] 형태의 비교 결과 (<, =, >)
        """
        # 특징 추출
        features_a = self.feature_extractor(a)  # [B, hidden_dim]
        features_b = self.feature_extractor(b)  # [B, hidden_dim]
        
        # 페어 결합
        pairs = torch.cat([features_a, features_b], dim=-1)  # [B, hidden_dim*2]
        
        # 비교 수행
        comparison = self.comparator(pairs)  # [B, 3]
        
        return comparison
    
    def is_less_than(self, a: torch.Tensor, b: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        """
        a < b 인지 판정합니다.
        
        Args:
            a: [B, D] 형태의 첫 번째 텐서
            b: [B, D] 형태의 두 번째 텐서
            threshold: 판정 임계값
        Returns:
            [B] 형태의 불린 텐서
        """
        if threshold is None:
            threshold = 0.5
        
        comparison = self.compare(a, b)
        less_than = comparison[:, 0] > threshold
        
        return less_than
    
    def is_equal(self, a: torch.Tensor, b: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        """
        a == b 인지 판정합니다.
        
        Args:
            a: [B, D] 형태의 첫 번째 텐서
            b: [B, D] 형태의 두 번째 텐서
            threshold: 판정 임계값
        Returns:
            [B] 형태의 불린 텐서
        """
        if threshold is None:
            threshold = 0.5
        
        comparison = self.compare(a, b)
        equal = comparison[:, 1] > threshold
        
        return equal
    
    def is_greater_than(self, a: torch.Tensor, b: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        """
        a > b 인지 판정합니다.
        
        Args:
            a: [B, D] 형태의 첫 번째 텐서
            b: [B, D] 형태의 두 번째 텐서
            threshold: 판정 임계값
        Returns:
            [B] 형태의 불린 텐서
        """
        if threshold is None:
            threshold = 0.5
        
        comparison = self.compare(a, b)
        greater_than = comparison[:, 2] > threshold
        
        return greater_than
    
    def get_comparison_type(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        비교 결과 유형을 반환합니다.
        
        Args:
            a: [B, D] 형태의 첫 번째 텐서
            b: [B, D] 형태의 두 번째 텐서
        Returns:
            [B] 형태의 정수 텐서 (0: <, 1: =, 2: >)
        """
        comparison = self.compare(a, b)
        comparison_type = torch.argmax(comparison, dim=-1)
        
        return comparison_type


def create_binary_comparator(input_dim: int = 128, hidden_dim: int = 48) -> BinaryComparator:
    """Binary Comparator 시드 생성 함수"""
    return BinaryComparator(input_dim=input_dim, hidden_dim=hidden_dim)

