"""
SEED-C06: Attention Director

주의 가중 배분 및 중요도 평가를 수행하는 Cellular 레벨 시드입니다.

구성 시드:
- M06: Context Integrator (맥락 통합)
- M01: Hierarchy Builder (계층 구조)
- A05: Grouping Nucleus (그룹화)

주요 기능:
- Multi-level attention computation (그룹, 계층, 맥락 기반)
- Dynamic importance scoring
- Context-aware attention weighting
- Hierarchical attention aggregation

Author: Manus AI
Date: 2025-12-09
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from seeds.base import BaseSeed, SeedConfig
from seeds.atomic.a05_grouping_nucleus import GroupingNucleus
from seeds.molecular.m01_hierarchy_builder import HierarchyBuilder
from seeds.molecular.m06_context_integrator import ContextIntegrator


@dataclass
class AttentionDirectorConfig(SeedConfig):
    """Attention Director 설정"""
    seed_id: str = "SEED-C06"
    name: str = "Attention Director"
    level: int = 2
    category: str = "Composition"
    bit_depth: str = "FP8"
    params: int = 1500000
    input_dim: int = 128
    output_dim: int = 128
    
    # C06 특화 설정
    num_heads: int = 8
    num_attention_layers: int = 2
    num_clusters: int = 16
    dropout: float = 0.1
    temperature: float = 1.0  # Attention temperature


class AttentionDirector(BaseSeed):
    """
    SEED-C06: Attention Director
    
    주의 가중 배분 및 중요도 평가를 수행합니다.
    
    주요 기능:
    - Multi-level attention computation (그룹, 계층, 맥락 기반)
    - Dynamic importance scoring
    - Context-aware attention weighting
    - Hierarchical attention aggregation
    
    Examples:
        >>> director = AttentionDirector(input_dim=128)
        >>> x = torch.randn(4, 50, 128)
        >>> output = director(x)
        >>> output['attended_output'].shape
        torch.Size([4, 50, 128])
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 8,
        num_attention_layers: int = 2,
        num_clusters: int = 16,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        config = AttentionDirectorConfig(
            input_dim=input_dim,
            output_dim=input_dim,
            num_heads=num_heads,
            num_attention_layers=num_attention_layers,
            num_clusters=num_clusters,
            dropout=dropout,
            temperature=temperature
        )
        super().__init__(config)
        
        self.config = config
        
        # 컴포넌트 초기화
        self._init_component_seeds()
        self._init_attention_modules()
        self._init_importance_scorer()
        self._init_aggregator()
    
    def _init_component_seeds(self):
        """구성 시드 초기화"""
        
        # A05: Grouping Nucleus - 입력 그룹화
        self.grouping_nucleus = GroupingNucleus(
            input_dim=self.config.input_dim,
            num_clusters=self.config.num_clusters
        )
        
        # M01: Hierarchy Builder - 계층 구조 구축
        self.hierarchy_builder = HierarchyBuilder(
            input_dim=self.config.input_dim,
            num_clusters=self.config.num_clusters
        )
        
        # M06: Context Integrator - 맥락 통합
        self.context_integrator = ContextIntegrator(
            input_dim=self.config.input_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout
        )
    
    def _init_attention_modules(self):
        """Attention 모듈 초기화"""
        
        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.input_dim),
            nn.LayerNorm(self.config.input_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
        
        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.config.input_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
                batch_first=True
            )
            for _ in range(self.config.num_attention_layers)
        ])
        
        # Layer normalization for each attention layer
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(self.config.input_dim)
            for _ in range(self.config.num_attention_layers)
        ])
        
        # Query, Key, Value projections for hierarchical attention
        self.hierarchical_query = nn.Linear(self.config.input_dim, self.config.input_dim)
        self.hierarchical_key = nn.Linear(self.config.input_dim, self.config.input_dim)
        self.hierarchical_value = nn.Linear(self.config.input_dim, self.config.input_dim)
        
        # Group-based attention projection
        self.group_attention_proj = nn.Linear(
            self.config.input_dim * 2,  # input + group features
            self.config.input_dim
        )
    
    def _init_importance_scorer(self):
        """중요도 평가 모듈 초기화"""
        
        # Importance scoring network
        self.importance_scorer = nn.Sequential(
            nn.Linear(self.config.input_dim * 3, 256),  # input + context + hierarchy
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0~1 범위의 중요도 점수
        )
        
        # Context-aware importance modulator
        self.context_modulator = nn.Sequential(
            nn.Linear(self.config.input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Hierarchical importance weights
        self.hierarchy_importance_weights = nn.Parameter(
            torch.ones(self.config.num_clusters)
        )
    
    def _init_aggregator(self):
        """최종 출력 집계 모듈 초기화"""
        
        # Multi-source attention aggregation
        self.aggregation_weights = nn.Parameter(torch.ones(4))  # 4개 소스
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.config.input_dim * 4, self.config.input_dim * 2),
            nn.LayerNorm(self.config.input_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.input_dim * 2, self.config.output_dim)
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(self.config.output_dim)
    
    def compute_group_attention(
        self,
        x: torch.Tensor,
        group_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        그룹 기반 주의 계산
        
        Args:
            x: [B, L, D] - 입력 텐서
            group_features: [B, L, D] - 그룹 특징
        
        Returns:
            group_attended: [B, L, D] - 그룹 주의 적용된 출력
            group_weights: [B, L] - 그룹 주의 가중치
        """
        B, L, D = x.shape
        
        # 입력과 그룹 특징 결합
        combined = torch.cat([x, group_features], dim=-1)  # [B, L, 2D]
        
        # 그룹 주의 계산
        group_attended = self.group_attention_proj(combined)  # [B, L, D]
        
        # 그룹 가중치 계산 (각 요소의 그룹 내 중요도)
        group_weights = torch.norm(group_features, dim=-1)  # [B, L]
        group_weights = F.softmax(group_weights / self.config.temperature, dim=-1)
        
        return group_attended, group_weights
    
    def compute_hierarchical_attention(
        self,
        x: torch.Tensor,
        hierarchy_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        계층 기반 주의 계산
        
        Args:
            x: [B, L, D] - 입력 텐서
            hierarchy_features: [B, L, D] - 계층 특징
        
        Returns:
            hierarchy_attended: [B, L, D] - 계층 주의 적용된 출력
            hierarchy_weights: [B, L] - 계층 주의 가중치
        """
        B, L, D = x.shape
        
        # Query, Key, Value 계산
        Q = self.hierarchical_query(x)  # [B, L, D]
        K = self.hierarchical_key(hierarchy_features)  # [B, L, D]
        V = self.hierarchical_value(hierarchy_features)  # [B, L, D]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # [B, L, L]
        attn_weights = F.softmax(scores / self.config.temperature, dim=-1)  # [B, L, L]
        
        # Attention 적용
        hierarchy_attended = torch.matmul(attn_weights, V)  # [B, L, D]
        
        # 각 위치의 평균 주의 가중치 (중요도 지표)
        hierarchy_weights = attn_weights.mean(dim=-1)  # [B, L]
        
        return hierarchy_attended, hierarchy_weights
    
    def compute_context_attention(
        self,
        x: torch.Tensor,
        context_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        맥락 기반 주의 계산
        
        Args:
            x: [B, L, D] - 입력 텐서
            context_features: [B, L, D] - 맥락 특징
        
        Returns:
            context_attended: [B, L, D] - 맥락 주의 적용된 출력
            context_weights: [B, L] - 맥락 주의 가중치
        """
        B, L, D = x.shape
        
        # Multi-head self-attention with context
        context_attended = x
        for attn_layer, norm_layer in zip(self.attention_layers, self.attention_norms):
            # Self-attention
            attn_output, attn_weights = attn_layer(
                context_attended,
                context_features,
                context_features
            )
            
            # Residual connection + normalization
            context_attended = norm_layer(context_attended + attn_output)
        
        # 맥락 가중치 계산 (맥락과의 유사도)
        context_similarity = F.cosine_similarity(
            x.view(B * L, D),
            context_features.view(B * L, D),
            dim=-1
        ).view(B, L)
        
        context_weights = F.softmax(context_similarity / self.config.temperature, dim=-1)
        
        return context_attended, context_weights
    
    def compute_importance_scores(
        self,
        x: torch.Tensor,
        context_features: torch.Tensor,
        hierarchy_features: torch.Tensor
    ) -> torch.Tensor:
        """
        종합 중요도 점수 계산
        
        Args:
            x: [B, L, D] - 입력 텐서
            context_features: [B, L, D] - 맥락 특징
            hierarchy_features: [B, L, D] - 계층 특징
        
        Returns:
            importance_scores: [B, L] - 중요도 점수
        """
        B, L, D = x.shape
        
        # 특징 결합
        combined = torch.cat([x, context_features, hierarchy_features], dim=-1)  # [B, L, 3D]
        
        # 중요도 점수 계산
        importance_scores = self.importance_scorer(combined).squeeze(-1)  # [B, L]
        
        # 맥락 기반 조정
        context_combined = torch.cat([x, context_features], dim=-1)  # [B, L, 2D]
        context_modulation = self.context_modulator(context_combined).squeeze(-1)  # [B, L]
        
        # 최종 중요도 점수 (곱셈 조정)
        importance_scores = importance_scores * context_modulation
        
        return importance_scores
    
    def aggregate_attention(
        self,
        x: torch.Tensor,
        group_attended: torch.Tensor,
        hierarchy_attended: torch.Tensor,
        context_attended: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        다중 소스 주의 집계
        
        Args:
            x: [B, L, D] - 원본 입력
            group_attended: [B, L, D] - 그룹 주의 출력
            hierarchy_attended: [B, L, D] - 계층 주의 출력
            context_attended: [B, L, D] - 맥락 주의 출력
            importance_scores: [B, L] - 중요도 점수
        
        Returns:
            aggregated: [B, L, D] - 집계된 출력
        """
        B, L, D = x.shape
        
        # 가중치 정규화
        weights = F.softmax(self.aggregation_weights, dim=0)
        
        # 가중 합
        weighted_sum = (
            weights[0] * x +
            weights[1] * group_attended +
            weights[2] * hierarchy_attended +
            weights[3] * context_attended
        )  # [B, L, D]
        
        # 모든 소스 결합
        all_sources = torch.cat([
            x,
            group_attended,
            hierarchy_attended,
            context_attended
        ], dim=-1)  # [B, L, 4D]
        
        # 최종 투영
        aggregated = self.output_projection(all_sources)  # [B, L, D]
        
        # 중요도 점수 적용
        importance_scores = importance_scores.unsqueeze(-1)  # [B, L, 1]
        aggregated = aggregated * importance_scores
        
        # 정규화
        aggregated = self.output_norm(aggregated)
        
        return aggregated
    
    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, L, D] - 입력 시퀀스
            scale: [B, 1] - 스케일 매개변수 (선택)
            context: 추가 맥락 정보 (선택)
                - 'query': [B, D] - 질의 벡터
                - 'context': [B, C, D] - 맥락 시퀀스
        
        Returns:
            Dict containing:
                - 'attended_output': [B, L, D] - 주의 적용된 출력
                - 'attention_weights': [B, L] - 종합 주의 가중치
                - 'importance_scores': [B, L] - 중요도 점수
                - 'group_weights': [B, L] - 그룹 주의 가중치
                - 'hierarchy_weights': [B, L] - 계층 주의 가중치
                - 'context_weights': [B, L] - 맥락 주의 가중치
        """
        B, L, D = x.shape
        
        # 1. 입력 인코딩
        x_encoded = self.input_encoder(x)  # [B, L, D]
        
        # 2. 구성 시드 적용
        # A05: 그룹화
        group_features = self.grouping_nucleus(x_encoded, scale, context)  # [B, L, D]
        
        # M01: 계층 구조
        hierarchy_features = self.hierarchy_builder(x_encoded, scale, context)  # [B, L, D]
        
        # M06: 맥락 통합
        context_features = self.context_integrator(x_encoded, scale, context)  # [B, L, D]
        
        # 3. 다층 주의 계산
        # 그룹 기반 주의
        group_attended, group_weights = self.compute_group_attention(
            x_encoded, group_features
        )
        
        # 계층 기반 주의
        hierarchy_attended, hierarchy_weights = self.compute_hierarchical_attention(
            x_encoded, hierarchy_features
        )
        
        # 맥락 기반 주의
        context_attended, context_weights = self.compute_context_attention(
            x_encoded, context_features
        )
        
        # 4. 중요도 점수 계산
        importance_scores = self.compute_importance_scores(
            x_encoded, context_features, hierarchy_features
        )
        
        # 5. 주의 집계
        attended_output = self.aggregate_attention(
            x_encoded,
            group_attended,
            hierarchy_attended,
            context_attended,
            importance_scores
        )
        
        # 6. 종합 주의 가중치 계산 (모든 주의의 평균)
        attention_weights = (
            group_weights + hierarchy_weights + context_weights
        ) / 3.0
        
        return {
            'attended_output': attended_output,
            'attention_weights': attention_weights,
            'importance_scores': importance_scores,
            'group_weights': group_weights,
            'hierarchy_weights': hierarchy_weights,
            'context_weights': context_weights,
            'group_features': group_features,
            'hierarchy_features': hierarchy_features,
            'context_features': context_features
        }
    
    def get_attention_map(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        주의 맵 추출 (시각화용)
        
        Args:
            x: [B, L, D] - 입력 시퀀스
            scale: [B, 1] - 스케일 매개변수
            context: 추가 맥락 정보
        
        Returns:
            attention_map: [B, L, L] - 주의 맵
        """
        output = self.forward(x, scale, context)
        
        # 계층 주의 가중치를 주의 맵으로 사용
        # (실제로는 compute_hierarchical_attention에서 계산된 attn_weights)
        # 여기서는 간단히 importance_scores를 대각 행렬로 변환
        importance = output['importance_scores']  # [B, L]
        attention_map = torch.diag_embed(importance)  # [B, L, L]
        
        return attention_map
