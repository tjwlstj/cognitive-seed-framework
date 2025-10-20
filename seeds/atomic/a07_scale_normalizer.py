"""
SEED-A07 — Scale Normalizer

스케일/단위를 정규화하는 원자 시드입니다.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig


class ScaleNormalizer(BaseSeed):
    """
    SEED-A07: Scale Normalizer
    
    Category: Abstraction
    Bit: INT8
    Params: ~128
    Purpose: 스케일/단위 정규화
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        config = SeedConfig(
            seed_id="SEED-A07",
            name="Scale Normalizer",
            level=0,
            category="Abstraction",
            bit_depth="INT8",
            params=128,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E"]  # 정규화는 유클리드 공간에서 수행
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        
        # 스케일 추정기
        self.scale_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 양수 스케일 보장
        )
        
        # 정규화 변환
        self.normalizer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 학습 가능한 스케일 파라미터
        self.target_scale = nn.Parameter(torch.ones(1))
        self.min_scale = nn.Parameter(torch.tensor(0.01))
        self.max_scale = nn.Parameter(torch.tensor(100.0))
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 형태의 입력 텐서
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보
        Returns:
            [B, L, D] 형태의 정규화된 텐서
        """
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 입력 스케일 추정
        estimated_scale = self.scale_estimator(x)  # [B, L, 1]
        
        # 스케일 클리핑
        estimated_scale = torch.clamp(estimated_scale, self.min_scale, self.max_scale)
        
        # 정규화 (목표 스케일로 변환)
        scale_factor = self.target_scale / (estimated_scale + 1e-8)
        x_scaled = x * scale_factor
        
        # 정규화 변환 적용
        normalized = self.normalizer(x_scaled)
        
        return normalized
    
    def estimate_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력의 스케일을 추정합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            [B, L] 형태의 추정 스케일
        """
        estimated_scale = self.scale_estimator(x).squeeze(-1)  # [B, L]
        return estimated_scale
    
    def normalize_to_scale(self, x: torch.Tensor, target_scale: float = 1.0) -> torch.Tensor:
        """
        특정 스케일로 정규화합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
            target_scale: 목표 스케일
        Returns:
            [B, L, D] 형태의 정규화된 텐서
        """
        # 입력 스케일 추정
        estimated_scale = self.scale_estimator(x)  # [B, L, 1]
        
        # 목표 스케일로 변환
        scale_factor = target_scale / (estimated_scale + 1e-8)
        x_scaled = x * scale_factor
        
        # 정규화 변환 적용
        normalized = self.normalizer(x_scaled)
        
        return normalized
    
    def compute_variance_stability(self, x: torch.Tensor) -> torch.Tensor:
        """
        분산 안정성을 계산합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            [B] 형태의 분산 안정성 점수
        """
        # 정규화 전 분산
        input_var = torch.var(x, dim=[1, 2])
        
        # 정규화 후 분산
        normalized = self.forward(x)
        output_var = torch.var(normalized, dim=[1, 2])
        
        # 분산 안정성 (1에 가까울수록 안정)
        stability = 1.0 / (1.0 + torch.abs(output_var - 1.0))
        
        return stability
    
    def check_overflow_underflow(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        오버플로우/언더플로우 위험을 체크합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            위험 통계를 담은 딕셔너리
        """
        max_val = torch.max(torch.abs(x))
        min_val = torch.min(torch.abs(x[x != 0])) if torch.any(x != 0) else torch.tensor(0.0)
        
        # FP16 기준 (약 65504)
        overflow_risk = (max_val > 1e4).float()
        # FP16 기준 (약 6e-5)
        underflow_risk = (min_val < 1e-4).float() if min_val > 0 else torch.tensor(0.0)
        
        return {
            "max_value": max_val,
            "min_value": min_val,
            "overflow_risk": overflow_risk,
            "underflow_risk": underflow_risk
        }


def create_scale_normalizer(input_dim: int = 128, hidden_dim: int = 64) -> ScaleNormalizer:
    """Scale Normalizer 시드 생성 함수"""
    return ScaleNormalizer(input_dim=input_dim, hidden_dim=hidden_dim)

