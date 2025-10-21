"""
SEED-M03: Pattern Completer

결손된 패턴을 보간(interpolation)하거나 외삽(extrapolation)하여 완성합니다.

Category: Pattern
Composed From: A03 (Recurrence Spotter) + A06 (Sequence Tracker) + A01 (Edge Detector)
Target Params: ~550K
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from seeds.base import BaseSeed, SeedConfig
from seeds.atomic import RecurrenceSpotter, SequenceTracker, EdgeDetector


@dataclass
class PatternCompleterConfig(SeedConfig):
    """Pattern Completer 설정"""
    seed_id: str = "SEED-M03"
    name: str = "Pattern Completer"
    level: int = 1
    category: str = "Pattern"
    bit_depth: str = "FP8"
    params: int = 550000
    input_dim: int = 128
    output_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1


class PatternCompleter(BaseSeed):
    """
    SEED-M03: Pattern Completer
    
    결손된 패턴을 보간/외삽하여 완성합니다.
    
    주요 기능:
    - 결손 위치 자동 감지
    - 반복 패턴 기반 보간
    - 시퀀스 추세 기반 외삽
    - Transformer 기반 맥락 활용
    """
    
    def __init__(self, input_dim: int = 128, num_heads: int = 8, 
                 num_layers: int = 4, dropout: float = 0.1):
        config = PatternCompleterConfig(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        
        # Atomic seeds
        self.recurrence_spotter = RecurrenceSpotter(input_dim)  # A03
        self.sequence_tracker = SequenceTracker(input_dim)      # A06
        self.edge_detector = EdgeDetector(input_dim)            # A01
        
        # Pattern completion network (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.completion_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Mask predictor
        self.mask_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Completion generator
        self.completion_generator = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, input_dim)
        )
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(5000, input_dim))
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Positional encoding 생성
        
        Args:
            max_len: 최대 시퀀스 길이
            d_model: 모델 차원
        Returns:
            pos_encoding: [max_len, d_model]
        """
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def get_positional_encoding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        시퀀스 길이에 맞는 positional encoding 반환
        
        Args:
            seq_len: 시퀀스 길이
            device: 디바이스
        Returns:
            pos_encoding: [seq_len, d_model]
        """
        return self.pos_encoding[:seq_len].to(device)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                context: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, L, D] - 입력 시퀀스 (일부 결손 가능)
            mask: [B, L] - 결손 위치 (1=관측, 0=결손), None이면 자동 감지
            context: 추가 맥락 정보
        Returns:
            completed: [B, L, D] - 완성된 시퀀스
        """
        B, L, D = x.shape
        
        # 1. 결손 위치 자동 감지 (mask가 없는 경우)
        if mask is None:
            mask = self.detect_missing_positions(x)  # [B, L]
        
        # 2. 반복 패턴 분석
        recurrence_info = self.recurrence_spotter(x * mask.unsqueeze(-1))
        
        # 3. 시퀀스 추세 파악
        sequence_info = self.sequence_tracker(x * mask.unsqueeze(-1))
        
        # 4. 경계 검출 (패턴 전환점)
        edge_info = self.edge_detector(x * mask.unsqueeze(-1))
        
        # 5. 특징 융합
        combined = recurrence_info + sequence_info + edge_info  # [B, L, D]
        
        # 6. Transformer 기반 패턴 완성
        # Positional encoding 추가
        pos_encoding = self.get_positional_encoding(L, x.device)  # [L, D]
        combined_with_pos = combined + pos_encoding.unsqueeze(0)  # [B, L, D]
        
        # Self-attention으로 전체 맥락 활용
        # Attention mask 생성 (결손 부분도 attend 가능)
        encoded = self.completion_encoder(combined_with_pos)  # [B, L, D]
        
        # 7. 결손 부분 생성
        completed = self.generate_missing_parts(x, encoded, mask)
        
        return completed
    
    def detect_missing_positions(self, x: torch.Tensor) -> torch.Tensor:
        """
        결손 위치 자동 감지
        
        Args:
            x: [B, L, D]
        Returns:
            mask: [B, L] - 1=관측, 0=결손
        """
        # 이상치 검출 기반: 크기가 너무 작으면 결손으로 간주
        magnitude = torch.norm(x, dim=-1)  # [B, L]
        threshold = magnitude.mean(dim=1, keepdim=True) * 0.1
        mask = (magnitude > threshold).float()
        
        return mask
    
    def generate_missing_parts(self, x: torch.Tensor, encoded: torch.Tensor, 
                               mask: torch.Tensor) -> torch.Tensor:
        """
        결손 부분 생성
        
        Args:
            x: [B, L, D] - 원본 입력
            encoded: [B, L, D] - 인코딩된 특징
            mask: [B, L] - 관측 마스크
        Returns:
            completed: [B, L, D] - 완성된 시퀀스
        """
        # 관측된 부분과 인코딩 정보 결합
        combined = torch.cat([x, encoded], dim=-1)  # [B, L, 2D]
        generated = self.completion_generator(combined)  # [B, L, D]
        
        # 마스크 적용: 관측된 부분은 유지, 결손 부분은 생성
        completed = x * mask.unsqueeze(-1) + generated * (1 - mask.unsqueeze(-1))
        
        return completed
    
    def interpolate(self, x: torch.Tensor, missing_indices: List[int]) -> torch.Tensor:
        """
        특정 위치 보간
        
        Args:
            x: [B, L, D]
            missing_indices: 결손 위치 리스트
        Returns:
            interpolated: [B, L, D]
        """
        B, L, D = x.shape
        
        # 마스크 생성
        mask = torch.ones(B, L, device=x.device)
        mask[:, missing_indices] = 0
        
        return self.forward(x, mask)
    
    def extrapolate(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        미래 패턴 외삽
        
        Args:
            x: [B, L, D]
            num_steps: 예측할 미래 스텝 수
        Returns:
            extrapolated: [B, L+num_steps, D]
        """
        B, L, D = x.shape
        
        # 시퀀스 추적기를 사용한 예측
        predictions = self.sequence_tracker.predict_next(x, num_steps=num_steps)  # [B, num_steps, D]
        
        # 반복 패턴 정보 활용하여 보정
        period_info = self.recurrence_spotter.detect_period(x)
        
        # 주기성을 고려한 보정 (간단한 버전)
        # 실제로는 더 정교한 주기성 기반 보정 필요
        period_adjusted = predictions  # 현재는 그대로 사용
        
        # 원본과 예측 결합
        extrapolated = torch.cat([x, period_adjusted], dim=1)  # [B, L+num_steps, D]
        
        return extrapolated
    
    def adjust_by_periodicity(self, predictions: torch.Tensor, 
                             period_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        주기성 정보를 활용한 예측 보정
        
        Args:
            predictions: [B, num_steps, D]
            period_info: 주기 정보
        Returns:
            adjusted: [B, num_steps, D]
        """
        # 간단한 구현: 주기 정보를 가중치로 활용
        # 실제로는 더 정교한 주기성 기반 보정 필요
        return predictions
    
    def compute_completion_quality(self, original: torch.Tensor, 
                                   completed: torch.Tensor,
                                   mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        완성 품질 평가
        
        Args:
            original: [B, L, D] - 원본 (일부 결손)
            completed: [B, L, D] - 완성된 시퀀스
            mask: [B, L] - 관측 마스크
        Returns:
            metrics: 품질 메트릭
        """
        # 결손 부분만 추출
        missing_mask = 1 - mask  # [B, L]
        
        # MSE (결손 부분)
        mse = torch.mean(
            ((original - completed) ** 2) * missing_mask.unsqueeze(-1)
        )
        
        # 구조 유사도 (간단한 버전)
        # 실제로는 SSIM 등 사용
        structural_sim = torch.cosine_similarity(
            original.reshape(-1, original.size(-1)),
            completed.reshape(-1, completed.size(-1)),
            dim=-1
        ).mean()
        
        return {
            'mse': mse,
            'structural_similarity': structural_sim,
            'completion_rate': missing_mask.mean()
        }

