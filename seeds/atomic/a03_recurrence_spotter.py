"""
SEED-A03 — Recurrence Spotter

반복/주기/모티프를 검출하는 원자 시드입니다.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig


class RecurrenceSpotter(BaseSeed):
    """
    SEED-A03: Recurrence Spotter
    
    Category: Temporal
    Bit: INT8
    Params: ~192
    Purpose: 반복/주기/모티프 검출
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 96):
        config = SeedConfig(
            seed_id="SEED-A03",
            name="Recurrence Spotter",
            level=0,
            category="Temporal",
            bit_depth="INT8",
            params=192,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E", "S"]  # 주기성은 구면 공간에서 자연스럽게 표현
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        
        # 시간적 패턴 인코더 (LSTM 기반)
        self.temporal_encoder = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )
        
        # 반복 패턴 검출기
        self.recurrence_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 주기 추정기
        self.period_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # 양수 주기 보장
        )
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 형태의 입력 텐서
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보
        Returns:
            [B, L, D] 형태의 반복 패턴 정보가 인코딩된 텐서
        """
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 시간적 패턴 인코딩
        temporal_features, _ = self.temporal_encoder(x)  # [B, L, hidden_dim*2]
        
        # 반복 패턴 검출
        recurrence_info = self.recurrence_detector(temporal_features)  # [B, L, D]
        
        # 원본과 반복 정보 결합
        output = x + recurrence_info
        
        return output
    
    def detect_period(self, x: torch.Tensor) -> torch.Tensor:
        """
        주기를 추정합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            [B, L] 형태의 추정 주기
        """
        # 시간적 패턴 인코딩
        temporal_features, _ = self.temporal_encoder(x)  # [B, L, hidden_dim*2]
        
        # 주기 추정
        period = self.period_estimator(temporal_features).squeeze(-1)  # [B, L]
        
        return period
    
    def compute_recurrence_score(self, x: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """
        반복성 점수를 계산합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
            window_size: 비교 윈도우 크기
        Returns:
            [B, L] 형태의 반복성 점수 (0~1)
        """
        batch_size, seq_len, dim = x.shape
        
        # 각 위치에서 이전 위치들과의 유사도 계산
        scores = torch.zeros(batch_size, seq_len, device=x.device)
        
        for i in range(window_size, seq_len):
            # 현재 위치와 이전 윈도우 내 위치들의 코사인 유사도
            current = x[:, i:i+1, :]  # [B, 1, D]
            window = x[:, i-window_size:i, :]  # [B, window_size, D]
            
            # 코사인 유사도
            similarity = F.cosine_similarity(
                current.expand(-1, window_size, -1), 
                window, 
                dim=-1
            )  # [B, window_size]
            
            # 최대 유사도를 반복성 점수로 사용
            scores[:, i] = torch.max(similarity, dim=-1)[0]
        
        return scores


def create_recurrence_spotter(input_dim: int = 128, hidden_dim: int = 96) -> RecurrenceSpotter:
    """Recurrence Spotter 시드 생성 함수"""
    return RecurrenceSpotter(input_dim=input_dim, hidden_dim=hidden_dim)

