"""
SEED-A02 — Symmetry Detector

반사/회전/병진 대칭성을 추정하는 원자 시드입니다.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig


class SymmetryDetector(BaseSeed):
    """
    SEED-A02: Symmetry Detector
    
    Category: Spatial
    Bit: INT8
    Params: ~256
    Purpose: 반사/회전/병진 대칭성 추정
    Invariance: 회전(부분), 스케일(정규화 후)
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        config = SeedConfig(
            seed_id="SEED-A02",
            name="Symmetry Detector",
            level=0,
            category="Spatial",
            bit_depth="INT8",
            params=256,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E", "S"]  # 대칭은 유클리드와 구면 공간에서 유용
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        
        # 대칭 특징 추출
        self.symmetry_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 대칭 유형 분류 (반사/회전/병진)
        self.symmetry_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3가지 대칭 유형
            nn.Softmax(dim=-1)
        )
        
        # 대칭 축/정도 추정
        self.symmetry_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 형태의 입력 텐서
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보
        Returns:
            [B, L, D] 형태의 대칭 정보가 인코딩된 텐서
        """
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 대칭 특징 추출
        symmetry_features = self.symmetry_encoder(x)  # [B, L, hidden_dim]
        
        # 대칭 정보 추정
        symmetry_info = self.symmetry_estimator(symmetry_features)  # [B, L, D]
        
        # 원본과 대칭 정보 결합
        output = x + symmetry_info
        
        return output
    
    def detect_symmetry_type(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        대칭 유형과 확률을 반환합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            symmetry_types: [B, L, 3] 형태의 대칭 유형 확률 (반사/회전/병진)
            symmetry_axis: [B, L, D] 형태의 대칭 축 정보
        """
        # 대칭 특징 추출
        symmetry_features = self.symmetry_encoder(x)
        
        # 대칭 유형 분류
        symmetry_types = self.symmetry_classifier(symmetry_features)  # [B, L, 3]
        
        # 대칭 축 추정
        symmetry_axis = self.symmetry_estimator(symmetry_features)  # [B, L, D]
        
        return symmetry_types, symmetry_axis
    
    def compute_symmetry_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        대칭성 점수를 계산합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            [B, L] 형태의 대칭성 점수 (0~1)
        """
        symmetry_types, _ = self.detect_symmetry_type(x)
        # 가장 높은 대칭 유형의 확률을 점수로 사용
        symmetry_score = torch.max(symmetry_types, dim=-1)[0]
        return symmetry_score


def create_symmetry_detector(input_dim: int = 128, hidden_dim: int = 128) -> SymmetryDetector:
    """Symmetry Detector 시드 생성 함수"""
    return SymmetryDetector(input_dim=input_dim, hidden_dim=hidden_dim)

