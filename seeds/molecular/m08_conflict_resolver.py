"""
SEED-M08: Conflict Resolver

제약 충돌을 해소하고 타협 솔루션을 생성하는 Molecular 레벨 시드입니다.

구성 시드:
- A08: Binary Comparator (충돌 탐지)
- M06: Context Integrator (맥락 분석)
- M02: Causality Detector (인과 추론)

주요 기능:
- 다중 제약 조건 처리
- 충돌 심각도 평가
- 맥락 기반 우선순위 결정
- 인과 기반 해결 경로 탐색
- 공정성 보장 타협 솔루션 생성

Author: Manus AI
Date: 2025-11-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from seeds.base import BaseSeed, SeedConfig
from seeds.atomic.a08_binary_comparator import BinaryComparator
from seeds.molecular.m06_context_integrator import ContextIntegrator
from seeds.molecular.m02_causality_detector import CausalityDetector


@dataclass
class ConflictResolverConfig(SeedConfig):
    """Conflict Resolver 설정"""
    seed_id: str = "SEED-M08"
    name: str = "Conflict Resolver"
    level: int = 1
    category: str = "Logic"
    bit_depth: str = "FP8"
    params: int = 800000
    input_dim: int = 128
    output_dim: int = 128
    
    # M08 특화 설정
    num_constraints_max: int = 10
    resolution_layers: int = 3
    fairness_weight: float = 0.5
    dropout: float = 0.1


class ConflictResolver(BaseSeed):
    """
    SEED-M08: Conflict Resolver
    
    제약 충돌을 해소하고 타협 솔루션을 생성합니다.
    
    주요 기능:
    - 다중 제약 조건 처리 및 인코딩
    - 제약 간 충돌 탐지 및 심각도 평가
    - 맥락 기반 우선순위 결정
    - 인과 추론을 통한 해결 경로 탐색
    - 공정성 보장 타협 솔루션 생성
    
    Examples:
        >>> resolver = ConflictResolver(input_dim=128)
        >>> constraints = torch.randn(4, 5, 128)  # 4 batches, 5 constraints
        >>> context = torch.randn(4, 10, 128)     # 4 batches, 10 context tokens
        >>> resolution, conflict_score, fairness_score = resolver(constraints, context)
        >>> resolution.shape
        torch.Size([4, 128])
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_constraints_max: int = 10,
        resolution_layers: int = 3,
        fairness_weight: float = 0.5,
        dropout: float = 0.1
    ):
        config = ConflictResolverConfig(
            input_dim=input_dim,
            output_dim=input_dim,
            num_constraints_max=num_constraints_max,
            resolution_layers=resolution_layers,
            fairness_weight=fairness_weight,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        
        # 컴포넌트 초기화
        self._init_atomic_molecular_seeds()
        self._init_constraint_encoder()
        self._init_conflict_detector()
        self._init_resolution_generator()
        self._init_fairness_module()
    
    def _init_atomic_molecular_seeds(self):
        """Atomic/Molecular seeds 초기화"""
        # A08: Binary Comparator - 충돌 탐지
        self.comparator = BinaryComparator(
            input_dim=self.config.input_dim,
            hidden_dim=64
        )
        
        # M06: Context Integrator - 맥락 분석
        self.context_integrator = ContextIntegrator(
            input_dim=self.config.input_dim,
            num_heads=8,
            num_encoder_layers=2
        )
        
        # M02: Causality Detector - 인과 추론
        self.causality_detector = CausalityDetector(
            input_dim=self.config.input_dim
        )
    
    def _init_constraint_encoder(self):
        """제약 조건 인코더 초기화"""
        self.constraint_encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, self.config.input_dim)
        )
        
        # 제약 간 관계 분석을 위한 self-attention
        self.constraint_attention = nn.MultiheadAttention(
            embed_dim=self.config.input_dim,
            num_heads=8,
            dropout=self.config.dropout,
            batch_first=True
        )
    
    def _init_conflict_detector(self):
        """충돌 탐지 모듈 초기화"""
        # 페어와이즈 충돌 점수 계산
        self.conflict_scorer = nn.Sequential(
            nn.Linear(self.config.input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 충돌 확률 [0, 1]
        )
        
        # 충돌 심각도 평가
        self.severity_estimator = nn.Sequential(
            nn.Linear(self.config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 심각도 [0, 1]
        )
    
    def _init_resolution_generator(self):
        """해결책 생성 모듈 초기화"""
        # 다층 해결책 생성 네트워크
        layers = []
        current_dim = self.config.input_dim
        
        for i in range(self.config.resolution_layers):
            layers.extend([
                nn.Linear(current_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(self.config.dropout)
            ])
            current_dim = 256
        
        layers.append(nn.Linear(256, self.config.output_dim))
        
        self.resolution_generator = nn.Sequential(*layers)
        
        # 우선순위 가중치 학습
        self.priority_net = nn.Sequential(
            nn.Linear(self.config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)  # 제약 간 우선순위
        )
    
    def _init_fairness_module(self):
        """공정성 보장 모듈 초기화"""
        # 공정성 점수 계산
        self.fairness_scorer = nn.Sequential(
            nn.Linear(self.config.input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 공정성 점수 [0, 1]
        )
        
        # 공정성 조정 네트워크
        self.fairness_adjuster = nn.Sequential(
            nn.Linear(self.config.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.input_dim)
        )
    
    def forward(
        self,
        constraints: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            constraints: [B, N, D] - N개의 제약 조건
            context: [B, L, D] - 맥락 정보 (선택)
            scale: [B, 1] - 스케일 매개변수 (선택)
        
        Returns:
            resolution: [B, D] - 해결책
            conflict_score: [B] - 충돌 심각도 점수
            fairness_score: [B] - 공정성 점수
        """
        batch_size, num_constraints, dim = constraints.shape
        
        # 1. 제약 조건 인코딩
        encoded_constraints = self._encode_constraints(constraints)
        
        # 2. 맥락 분석 (M06 활용)
        if context is not None:
            context_features = self.context_integrator(context, scale=scale)
            # 맥락을 제약에 통합
            context_summary = context_features.mean(dim=1, keepdim=True)  # [B, 1, D]
            encoded_constraints = encoded_constraints + context_summary
        
        # 3. 충돌 탐지 및 평가
        conflict_matrix, conflict_score = self._detect_conflicts(encoded_constraints)
        
        # 4. 인과 추론 (M02 활용)
        causal_features = self.causality_detector(encoded_constraints, scale=scale)
        
        # 5. 우선순위 결정
        priorities = self._compute_priorities(encoded_constraints, causal_features)
        
        # 6. 해결책 생성
        resolution = self._generate_resolution(
            encoded_constraints,
            causal_features,
            priorities,
            conflict_matrix
        )
        
        # 7. 공정성 평가 및 조정
        fairness_score = self._evaluate_fairness(resolution, encoded_constraints)
        resolution = self._adjust_for_fairness(resolution, fairness_score)
        
        return resolution, conflict_score, fairness_score
    
    def _encode_constraints(self, constraints: torch.Tensor) -> torch.Tensor:
        """
        제약 조건을 인코딩하고 관계를 분석합니다.
        
        Args:
            constraints: [B, N, D]
        Returns:
            encoded: [B, N, D]
        """
        # 개별 제약 인코딩
        encoded = self.constraint_encoder(constraints)  # [B, N, D]
        
        # 제약 간 관계 분석 (self-attention)
        attn_output, _ = self.constraint_attention(
            encoded, encoded, encoded
        )  # [B, N, D]
        
        # 잔차 연결
        encoded = encoded + attn_output
        
        return encoded
    
    def _detect_conflicts(
        self,
        constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        제약 간 충돌을 탐지하고 심각도를 평가합니다.
        
        Args:
            constraints: [B, N, D]
        Returns:
            conflict_matrix: [B, N, N] - 페어와이즈 충돌 점수
            conflict_score: [B] - 전체 충돌 심각도
        """
        batch_size, num_constraints, dim = constraints.shape
        
        # 페어와이즈 충돌 탐지 (A08 활용)
        conflict_matrix = torch.zeros(
            batch_size, num_constraints, num_constraints,
            device=constraints.device
        )
        
        for i in range(num_constraints):
            for j in range(i + 1, num_constraints):
                constraint_i = constraints[:, i, :]  # [B, D]
                constraint_j = constraints[:, j, :]  # [B, D]
                
                # 비교를 통한 충돌 탐지
                comparison = self.comparator.compare(constraint_i, constraint_j)  # [B, 3]
                
                # 충돌 점수 계산 (동등하지 않을수록 충돌 가능성 높음)
                conflict_prob = 1.0 - comparison[:, 1]  # 1 - P(equal)
                
                # 페어 결합하여 충돌 점수 계산
                pair = torch.cat([constraint_i, constraint_j], dim=-1)  # [B, 2D]
                conflict_score_ij = self.conflict_scorer(pair).squeeze(-1)  # [B]
                
                # 최종 충돌 점수
                final_conflict = conflict_prob * conflict_score_ij
                
                conflict_matrix[:, i, j] = final_conflict
                conflict_matrix[:, j, i] = final_conflict
        
        # 전체 충돌 심각도 계산
        severity_scores = self.severity_estimator(constraints).squeeze(-1)  # [B, N]
        conflict_score = (conflict_matrix.sum(dim=(1, 2)) * severity_scores.mean(dim=1)) / (num_constraints ** 2)
        
        return conflict_matrix, conflict_score
    
    def _compute_priorities(
        self,
        constraints: torch.Tensor,
        causal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        제약의 우선순위를 계산합니다.
        
        Args:
            constraints: [B, N, D]
            causal_features: [B, N, D]
        Returns:
            priorities: [B, N, 1] - 각 제약의 우선순위 가중치
        """
        # 인과 정보를 고려한 우선순위 계산
        combined = constraints + causal_features
        priorities = self.priority_net(combined)  # [B, N, 1]
        
        return priorities
    
    def _generate_resolution(
        self,
        constraints: torch.Tensor,
        causal_features: torch.Tensor,
        priorities: torch.Tensor,
        conflict_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        타협 솔루션을 생성합니다.
        
        Args:
            constraints: [B, N, D]
            causal_features: [B, N, D]
            priorities: [B, N, 1]
            conflict_matrix: [B, N, N]
        Returns:
            resolution: [B, D]
        """
        # 우선순위 가중 평균
        weighted_constraints = constraints * priorities  # [B, N, D]
        
        # 충돌을 고려한 가중치 조정
        conflict_weights = 1.0 - conflict_matrix.mean(dim=2, keepdim=True)  # [B, N, 1]
        weighted_constraints = weighted_constraints * conflict_weights
        
        # 제약 통합
        integrated = weighted_constraints.sum(dim=1)  # [B, D]
        
        # 인과 정보 통합
        causal_summary = (causal_features * priorities).sum(dim=1)  # [B, D]
        
        # 최종 해결책 생성
        combined = integrated + 0.5 * causal_summary
        resolution = self.resolution_generator(combined)  # [B, D]
        
        return resolution
    
    def _evaluate_fairness(
        self,
        resolution: torch.Tensor,
        constraints: torch.Tensor
    ) -> torch.Tensor:
        """
        해결책의 공정성을 평가합니다.
        
        Args:
            resolution: [B, D]
            constraints: [B, N, D]
        Returns:
            fairness_score: [B]
        """
        batch_size, num_constraints, dim = constraints.shape
        
        # 각 제약에 대한 만족도 계산
        resolution_expanded = resolution.unsqueeze(1).expand(-1, num_constraints, -1)  # [B, N, D]
        
        # 페어와이즈 공정성 점수
        fairness_scores = []
        for i in range(num_constraints):
            constraint_i = constraints[:, i, :]  # [B, D]
            pair = torch.cat([resolution, constraint_i], dim=-1)  # [B, 2D]
            score = self.fairness_scorer(pair).squeeze(-1)  # [B]
            fairness_scores.append(score)
        
        fairness_scores = torch.stack(fairness_scores, dim=1)  # [B, N]
        
        # 평균 공정성 점수
        fairness_score = fairness_scores.mean(dim=1)  # [B]
        
        return fairness_score
    
    def _adjust_for_fairness(
        self,
        resolution: torch.Tensor,
        fairness_score: torch.Tensor
    ) -> torch.Tensor:
        """
        공정성을 고려하여 해결책을 조정합니다.
        
        Args:
            resolution: [B, D]
            fairness_score: [B]
        Returns:
            adjusted_resolution: [B, D]
        """
        # 공정성 점수가 낮을수록 더 많이 조정
        adjustment_weight = (1.0 - fairness_score).unsqueeze(-1) * self.config.fairness_weight
        
        # 조정 벡터 생성
        adjustment = self.fairness_adjuster(resolution)
        
        # 가중 조정 적용
        adjusted_resolution = resolution + adjustment_weight * adjustment
        
        return adjusted_resolution
    
    def resolve_conflicts(
        self,
        constraints: List[torch.Tensor],
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        고수준 API: 제약 충돌을 해소하고 상세 정보를 반환합니다.
        
        Args:
            constraints: List of constraint tensors
            context: Optional context tensor
        Returns:
            Dictionary containing:
                - resolution: 해결책
                - conflict_score: 충돌 심각도
                - fairness_score: 공정성 점수
                - priorities: 제약 우선순위
        """
        # 제약을 배치 텐서로 변환
        constraints_tensor = torch.stack(constraints, dim=1)  # [B, N, D]
        
        # 해결책 생성
        resolution, conflict_score, fairness_score = self.forward(
            constraints_tensor, context
        )
        
        # 우선순위 계산
        encoded_constraints = self._encode_constraints(constraints_tensor)
        causal_features = self.causality_detector(encoded_constraints)
        priorities = self._compute_priorities(encoded_constraints, causal_features)
        
        return {
            'resolution': resolution,
            'conflict_score': conflict_score,
            'fairness_score': fairness_score,
            'priorities': priorities.squeeze(-1)
        }


def create_conflict_resolver(
    input_dim: int = 128,
    num_constraints_max: int = 10,
    resolution_layers: int = 3,
    fairness_weight: float = 0.5,
    dropout: float = 0.1
) -> ConflictResolver:
    """Conflict Resolver 시드 생성 함수"""
    return ConflictResolver(
        input_dim=input_dim,
        num_constraints_max=num_constraints_max,
        resolution_layers=resolution_layers,
        fairness_weight=fairness_weight,
        dropout=dropout
    )
