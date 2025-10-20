"""
SEED-A06 — Sequence Tracker

순서를 추적하고 다음 상태를 예측하는 원자 시드입니다.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig


class SequenceTracker(BaseSeed):
    """
    SEED-A06: Sequence Tracker
    
    Category: Temporal
    Bit: INT8
    Params: ~320
    Purpose: 순서 추적·다음 상태 예측
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 160):
        config = SeedConfig(
            seed_id="SEED-A06",
            name="Sequence Tracker",
            level=0,
            category="Temporal",
            bit_depth="INT8",
            params=320,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E"]  # 시퀀스 추적은 유클리드 공간에서 수행
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        
        # 시퀀스 인코더 (GRU 기반)
        self.sequence_encoder = nn.GRU(
            input_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 다음 상태 예측기
        self.next_state_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 순서 임베딩
        self.position_encoder = nn.Sequential(
            nn.Linear(1, 64),
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
            [B, L, D] 형태의 시퀀스 정보가 인코딩된 텐서
        """
        batch_size, seq_len, dim = x.shape
        
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 위치 인코딩 추가
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        positions = positions.unsqueeze(0).unsqueeze(-1) / seq_len  # [1, L, 1]
        position_encoding = self.position_encoder(positions)  # [1, L, D]
        x_with_pos = x + position_encoding
        
        # 시퀀스 인코딩
        sequence_features, _ = self.sequence_encoder(x_with_pos)  # [B, L, hidden_dim]
        
        # 다음 상태 예측
        next_state = self.next_state_predictor(sequence_features)  # [B, L, D]
        
        # 원본과 예측 정보 결합
        output = x + next_state
        
        return output
    
    def predict_next(self, x: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """
        다음 상태를 예측합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
            num_steps: 예측할 스텝 수
        Returns:
            [B, num_steps, D] 형태의 예측 텐서
        """
        batch_size, seq_len, dim = x.shape
        
        # 위치 인코딩 추가
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        positions = positions.unsqueeze(0).unsqueeze(-1) / seq_len
        position_encoding = self.position_encoder(positions)
        x_with_pos = x + position_encoding
        
        # 시퀀스 인코딩
        _, hidden = self.sequence_encoder(x_with_pos)  # hidden: [2, B, hidden_dim]
        
        predictions = []
        current_hidden = hidden
        
        # 마지막 입력으로 시작
        current_input = x[:, -1:, :]  # [B, 1, D]
        
        for step in range(num_steps):
            # 다음 위치 인코딩
            next_pos = (seq_len + step) / (seq_len + num_steps)
            next_pos_tensor = torch.tensor([[next_pos]], dtype=torch.float32, device=x.device)
            next_pos_encoding = self.position_encoder(next_pos_tensor)  # [1, 1, D]
            
            current_input_with_pos = current_input + next_pos_encoding
            
            # 다음 상태 예측
            _, current_hidden = self.sequence_encoder(current_input_with_pos, current_hidden)
            next_state = self.next_state_predictor(current_hidden[-1:].transpose(0, 1))  # [B, 1, D]
            
            predictions.append(next_state)
            current_input = next_state
        
        predictions = torch.cat(predictions, dim=1)  # [B, num_steps, D]
        
        return predictions
    
    def compute_tracking_accuracy(self, x: torch.Tensor) -> torch.Tensor:
        """
        시퀀스 추적 정확도를 계산합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            [B] 형태의 추적 정확도
        """
        if x.size(1) < 2:
            return torch.zeros(x.size(0), device=x.device)
        
        # 각 위치에서 다음 상태 예측
        predictions = self.predict_next(x[:, :-1, :], num_steps=1)  # [B, 1, D]
        targets = x[:, -1:, :]  # [B, 1, D]
        
        # MSE 기반 정확도
        mse = F.mse_loss(predictions, targets, reduction='none').mean(dim=[1, 2])
        accuracy = 1.0 / (1.0 + mse)
        
        return accuracy


def create_sequence_tracker(input_dim: int = 128, hidden_dim: int = 160) -> SequenceTracker:
    """Sequence Tracker 시드 생성 함수"""
    return SequenceTracker(input_dim=input_dim, hidden_dim=hidden_dim)

