"""
SEED-M06: Context Integrator

다층적 맥락을 통합하여 중의성을 해소하는 Molecular 레벨 시드입니다.

구성 시드:
- A06: Sequence Tracker (시간적 맥락)
- M01: Hierarchy Builder (계층적 맥락)
- A05: Grouping Nucleus (그룹 맥락)

주요 기능:
- Multi-scale context encoding (local/global)
- Hierarchical context integration
- Multi-head attention fusion
- Context-based disambiguation

Author: Manus AI (누스양)
Date: 2025-11-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from seeds.base import BaseSeed, SeedConfig
from seeds.atomic.a06_sequence_tracker import SequenceTracker
from seeds.atomic.a05_grouping_nucleus import GroupingNucleus
from seeds.molecular.m01_hierarchy_builder import HierarchyBuilder


@dataclass
class ContextIntegratorConfig(SeedConfig):
    """Context Integrator 설정"""
    seed_id: str = "SEED-M06"
    name: str = "Context Integrator"
    level: int = 1
    category: str = "Composition"
    bit_depth: str = "FP8"
    params: int = 650000
    input_dim: int = 128
    output_dim: int = 128
    
    # M06 특화 설정
    num_heads: int = 8
    num_encoder_layers: int = 2
    context_window: int = 5
    dropout: float = 0.1


class ContextIntegrator(BaseSeed):
    """
    SEED-M06: Context Integrator
    
    다층적 맥락을 통합하여 중의성을 해소합니다.
    
    주요 기능:
    - Multi-scale context encoding (local/global)
    - Hierarchical context integration
    - Multi-head attention fusion
    - Context-based disambiguation
    
    Examples:
        >>> integrator = ContextIntegrator(input_dim=128)
        >>> x = torch.randn(4, 50, 128)
        >>> output = integrator(x)
        >>> output.shape
        torch.Size([4, 50, 128])
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 2,
        context_window: int = 5,
        dropout: float = 0.1
    ):
        config = ContextIntegratorConfig(
            input_dim=input_dim,
            output_dim=input_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            context_window=context_window,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        
        # 컴포넌트 초기화
        self._init_atomic_seeds()
        self._init_context_encoders()
        self._init_fusion_module()
        self._init_disambiguator()
    
    def _init_atomic_seeds(self):
        """Atomic/Molecular seeds 초기화"""
        
        # A06: Sequence Tracker (시간적 맥락)
        self.sequence_tracker = SequenceTracker(self.config.input_dim)
        
        # M01: Hierarchy Builder (계층적 맥락)
        self.hierarchy_builder = HierarchyBuilder(self.config.input_dim)
        
        # A05: Grouping Nucleus (그룹 맥락)
        self.grouping_nucleus = GroupingNucleus(self.config.input_dim)
    
    def _init_context_encoders(self):
        """Local/Global context encoders 초기화"""
        
        # Local context encoder (Transformer)
        local_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.input_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.input_dim * 4,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.local_context_encoder = nn.TransformerEncoder(
            local_encoder_layer,
            num_layers=self.config.num_encoder_layers
        )
        
        # Global context encoder (Transformer)
        global_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.input_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.input_dim * 4,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.global_context_encoder = nn.TransformerEncoder(
            global_encoder_layer,
            num_layers=self.config.num_encoder_layers
        )
    
    def encode_local_context(
        self,
        x: torch.Tensor,
        window_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        슬라이딩 윈도우 기반 국소 맥락 인코딩
        
        Args:
            x: [B, L, D] - 입력 시퀀스
            window_size: 윈도우 크기 (None이면 config 값 사용)
        
        Returns:
            local_context: [B, L, D] - 국소 맥락
        """
        if window_size is None:
            window_size = self.config.context_window
        
        B, L, D = x.shape
        
        # 패딩 (양쪽에 window_size // 2씩)
        pad_size = window_size // 2
        padded = F.pad(x, (0, 0, pad_size, pad_size))  # [B, L+2*pad_size, D]
        
        # 슬라이딩 윈도우 추출
        local_contexts = []
        for i in range(L):
            window = padded[:, i:i+window_size, :]  # [B, window_size, D]
            local_contexts.append(window)
        
        local_contexts = torch.stack(local_contexts, dim=1)  # [B, L, window_size, D]
        
        # Transformer 인코딩 (각 윈도우 독립적으로)
        local_contexts_flat = local_contexts.view(B * L, window_size, D)
        encoded_flat = self.local_context_encoder(local_contexts_flat)  # [B*L, window_size, D]
        
        # 중앙 토큰만 추출 (윈도우의 중심)
        center_idx = window_size // 2
        local_context = encoded_flat[:, center_idx, :]  # [B*L, D]
        local_context = local_context.view(B, L, D)  # [B, L, D]
        
        return local_context
    
    def encode_global_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        전체 시퀀스 기반 전역 맥락 인코딩
        
        Args:
            x: [B, L, D] - 입력 시퀀스
        
        Returns:
            global_context: [B, L, D] - 전역 맥락
        """
        # Transformer 인코딩 (전체 시퀀스)
        global_context = self.global_context_encoder(x)  # [B, L, D]
        return global_context
    
    def _init_fusion_module(self):
        """Context fusion module 초기화"""
        
        # Multi-head attention for fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=self.config.input_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True
        )
        
        # Fusion weights (학습 가능한 가중치)
        self.fusion_weights = nn.Parameter(torch.ones(5))  # 5개 맥락 소스
        
        # Projection layer
        self.fusion_projection = nn.Linear(
            self.config.input_dim * 5,  # 5개 맥락 concat
            self.config.input_dim
        )
        
        self.fusion_norm = nn.LayerNorm(self.config.input_dim)
        self.fusion_dropout = nn.Dropout(self.config.dropout)
    
    def fuse_contexts(
        self,
        local_context: torch.Tensor,
        global_context: torch.Tensor,
        temporal_context: torch.Tensor,
        hierarchical_context: torch.Tensor,
        group_context: torch.Tensor
    ) -> torch.Tensor:
        """
        다층적 맥락 융합
        
        Args:
            local_context: [B, L, D]
            global_context: [B, L, D]
            temporal_context: [B, L, D]
            hierarchical_context: [B, L, D]
            group_context: [B, L, D]
        
        Returns:
            fused_context: [B, L, D]
        """
        B, L, D = local_context.shape
        
        # 방법 1: Cross-attention fusion
        # Query: local_context, Key/Value: global_context
        attn_output, attn_weights = self.fusion_attention(
            local_context,
            global_context,
            global_context
        )  # [B, L, D]
        
        # 방법 2: Weighted sum
        # Softmax normalize weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Stack all contexts
        all_contexts = torch.stack([
            local_context,
            global_context,
            temporal_context,
            hierarchical_context,
            group_context
        ], dim=0)  # [5, B, L, D]
        
        # Weighted sum
        weighted_sum = torch.sum(
            all_contexts * weights.view(5, 1, 1, 1),
            dim=0
        )  # [B, L, D]
        
        # 두 방법 평균
        fused = (attn_output + weighted_sum) / 2
        
        # Residual connection + Norm
        fused = self.fusion_norm(fused + local_context)
        fused = self.fusion_dropout(fused)
        
        return fused
    
    def _init_disambiguator(self):
        """Disambiguator (중의성 해소) 초기화"""
        
        # 3-layer MLP
        self.disambiguator = nn.Sequential(
            nn.Linear(self.config.input_dim * 3, self.config.input_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.input_dim * 2, self.config.input_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.input_dim, self.config.input_dim)
        )
        
        self.disambiguator_norm = nn.LayerNorm(self.config.input_dim)
    
    def disambiguate(
        self,
        x: torch.Tensor,
        fused_context: torch.Tensor
    ) -> torch.Tensor:
        """
        맥락 기반 중의성 해소
        
        Args:
            x: [B, L, D] - 원본 입력
            fused_context: [B, L, D] - 융합된 맥락
        
        Returns:
            disambiguated: [B, L, D] - 중의성 해소된 출력
        """
        # 상호작용 특징 (element-wise product)
        interaction = x * fused_context  # [B, L, D]
        
        # Concat: 원본 + 맥락 + 상호작용
        combined = torch.cat([x, fused_context, interaction], dim=-1)  # [B, L, 3*D]
        
        # MLP
        disambiguated = self.disambiguator(combined)  # [B, L, D]
        
        # Residual connection + Norm
        disambiguated = self.disambiguator_norm(disambiguated + x)
        
        return disambiguated
    
    def forward(
        self,
        x: torch.Tensor,
        context_window: Optional[int] = None,
        return_metadata: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Forward pass
        
        Args:
            x: [B, L, D] - 입력 시퀀스
            context_window: 윈도우 크기 (None이면 config 값 사용)
            return_metadata: 메타데이터 반환 여부
        
        Returns:
            output: [B, L, D] - 중의성 해소된 출력
            metadata (optional): 중간 결과 딕셔너리
        """
        B, L, D = x.shape
        
        # 1. Multi-scale context encoding
        local_context = self.encode_local_context(x, context_window)  # [B, L, D]
        global_context = self.encode_global_context(x)  # [B, L, D]
        
        # 2. Hierarchical context integration
        # A06: Temporal context
        temporal_output = self.sequence_tracker(x)  # [B, L, D]
        temporal_context = temporal_output
        
        # M01: Hierarchical context
        hierarchical_output = self.hierarchy_builder(x)  # [B, L, D]
        hierarchical_context = hierarchical_output
        
        # A05: Group context
        group_output = self.grouping_nucleus(x)  # [B, L, D]
        group_context = group_output
        
        # 3. Context fusion
        fused_context = self.fuse_contexts(
            local_context,
            global_context,
            temporal_context,
            hierarchical_context,
            group_context
        )  # [B, L, D]
        
        # 4. Disambiguation
        output = self.disambiguate(x, fused_context)  # [B, L, D]
        
        if return_metadata:
            metadata = {
                'local_context': local_context,
                'global_context': global_context,
                'temporal_context': temporal_context,
                'hierarchical_context': hierarchical_context,
                'group_context': group_context,
                'fused_context': fused_context,
                'fusion_weights': F.softmax(self.fusion_weights, dim=0).detach()
            }
            return output, metadata
        
        return output
    
    def get_context_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """
        각 맥락의 중요도 계산
        
        Args:
            x: [B, L, D] - 입력 시퀀스
        
        Returns:
            importance: 맥락별 중요도 딕셔너리
        """
        with torch.no_grad():
            _, metadata = self.forward(x, return_metadata=True)
            weights = metadata['fusion_weights'].cpu().numpy()
            
            importance = {
                'local': float(weights[0]),
                'global': float(weights[1]),
                'temporal': float(weights[2]),
                'hierarchical': float(weights[3]),
                'group': float(weights[4])
            }
        
        return importance


# 편의 함수
def create_context_integrator(input_dim: int = 128, **kwargs) -> ContextIntegrator:
    """
    Context Integrator 생성 헬퍼 함수
    
    Args:
        input_dim: 입력 차원
        **kwargs: 추가 설정
    
    Returns:
        integrator: ContextIntegrator 인스턴스
    """
    return ContextIntegrator(input_dim=input_dim, **kwargs)
