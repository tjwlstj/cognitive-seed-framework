"""
SEED-A01 — Edge Detector

급격한 변화/경계를 검출하는 원자 시드입니다.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig


class EdgeDetector(BaseSeed):
    """
    SEED-A01: Edge Detector
    
    Category: Pattern
    Bit: INT8
    Params: ~128
    Purpose: 급격한 변화/경계 검출
    I/O: [B,*,D] → [B,*,D]
    Invariance: 부분적 평행이동
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        config = SeedConfig(
            seed_id="SEED-A01",
            name="Edge Detector",
            level=0,
            category="Pattern",
            bit_depth="INT8",
            params=128,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E"]  # 경계 검출은 주로 유클리드 공간에서 수행
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        
        # 경계 검출을 위한 컨볼루션 레이어 (Sobel-like)
        self.edge_conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
        )
        
        # 경계 강도 추정
        self.edge_strength = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 형태의 입력 텐서 (L: 시퀀스 길이, D: 특징 차원)
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보
        Returns:
            [B, L, D] 형태의 경계 검출 결과
        """
        batch_size, seq_len, dim = x.shape
        
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 경계 검출 (1D 컨볼루션)
        x_conv = x.transpose(1, 2)  # [B, D, L]
        edges = self.edge_conv(x_conv)  # [B, D, L]
        edges = edges.transpose(1, 2)  # [B, L, D]
        
        # 경계 강도 추정
        strength = self.edge_strength(edges)
        
        # 원본과 경계의 가중 결합
        output = x + strength * edges
        
        return output
    
    def detect_edges(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        경계를 이진 마스크로 반환합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
            threshold: 경계 판정 임계값
        Returns:
            [B, L] 형태의 이진 마스크
        """
        output = self.forward(x)
        edge_magnitude = torch.norm(output - x, dim=-1)  # [B, L]
        edge_mask = (edge_magnitude > threshold).float()
        return edge_mask


def create_edge_detector(input_dim: int = 128, hidden_dim: int = 64) -> EdgeDetector:
    """Edge Detector 시드 생성 함수"""
    return EdgeDetector(input_dim=input_dim, hidden_dim=hidden_dim)

