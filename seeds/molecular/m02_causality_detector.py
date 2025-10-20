"""
SEED-M02 — Causality Detector

시간적 선후 관계와 개입 효과를 기반으로 인과 구조를 추정하는 분자 시드입니다.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig
from ..atomic.a06_sequence_tracker import SequenceTracker
from ..atomic.a03_recurrence_spotter import RecurrenceSpotter
from ..atomic.a08_binary_comparator import BinaryComparator


class CausalityDetector(BaseSeed):
    """
    SEED-M02: Causality Detector
    
    Category: Temporal/Logic
    Bit: FP8
    Params: ~600K
    Purpose: 시간적 패턴과 개입 효과를 분석하여 인과 관계 추정
    I/O: [B,T,D] → [B,T,D]
    Composed From: A06 + A03 + A08
    """
    
    def __init__(self, input_dim: int = 128):
        config = SeedConfig(
            seed_id="SEED-M02",
            name="Causality Detector",
            level=1,
            category="Temporal/Logic",
            bit_depth="FP8",
            params=600000,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E"]  # Euclidean (시계열 분석)
        )
        super().__init__(config)
        
        # Atomic seeds
        self.sequence_tracker = SequenceTracker(input_dim)      # A06
        self.recurrence_spotter = RecurrenceSpotter(input_dim)  # A03
        self.comparator = BinaryComparator(input_dim)           # A08
        
        # Causal inference network
        self.causal_encoder = nn.Sequential(
            nn.Linear(input_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim)
        )
        
        # Causal graph predictor (DAG)
        self.graph_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 인과 관계 확률
        )
        
        # Intervention effect estimator
        self.intervention_net = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
        # Temporal precedence analyzer
        self.precedence_net = nn.GRU(
            input_dim, input_dim, num_layers=2, batch_first=True, bidirectional=True
        )
        self.precedence_proj = nn.Linear(input_dim * 2, input_dim)
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] - 시계열 데이터
            scale: [B, 1] 형태의 스케일 매개변수
            context: {'interventions': [B, T, D]} - 개입 정보 (선택)
        Returns:
            causal_features: [B, T, D] - 인과 정보가 인코딩된 특징
        """
        batch_size, seq_len, dim = x.shape
        
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x_proj = self.mgp(x)
        else:
            x_proj = x
        
        # 1. 시간적 패턴 추적 (A06)
        temporal_features = self.sequence_tracker(x_proj)
        
        # 2. 반복 패턴 검출 (A03) - 주기성
        recurrence_features = self.recurrence_spotter(x_proj)
        
        # 3. 선후 관계 분석
        precedence_features = self.analyze_precedence(x_proj)
        
        # 4. 특징 융합
        combined = torch.cat([
            temporal_features,
            recurrence_features,
            precedence_features
        ], dim=-1)
        
        causal_features = self.causal_encoder(combined)
        
        # 5. 개입 효과 추정 (있는 경우)
        if context and 'interventions' in context:
            interventions = context['interventions']
            intervention_effect = self.estimate_intervention_effect(
                causal_features, interventions
            )
            causal_features = causal_features + 0.3 * intervention_effect
        
        return causal_features
    
    def analyze_precedence(self, x: torch.Tensor) -> torch.Tensor:
        """
        시간적 선후 관계 분석
        
        Args:
            x: [B, T, D]
        Returns:
            precedence_features: [B, T, D]
        """
        # 양방향 GRU로 과거와 미래 맥락 파악
        precedence_out, _ = self.precedence_net(x)  # [B, T, 2D]
        precedence_features = self.precedence_proj(precedence_out)  # [B, T, D]
        
        return precedence_features
    
    def estimate_intervention_effect(self, features: torch.Tensor, 
                                    interventions: torch.Tensor) -> torch.Tensor:
        """
        개입 효과 추정
        
        Args:
            features: [B, T, D]
            interventions: [B, T, D]
        Returns:
            intervention_effect: [B, T, D]
        """
        combined = torch.cat([features, interventions], dim=-1)
        intervention_effect = self.intervention_net(combined)
        
        return intervention_effect
    
    def estimate_causal_graph(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        인과 그래프 (DAG) 추정
        
        Args:
            x: [B, T, D]
            threshold: 인과 관계 판정 임계값
        Returns:
            causal_graph: [B, T, T] - 시간 단계 간 인과 관계 행렬
        """
        B, T, D = x.shape
        
        # 각 시간 단계 간 인과 관계 추정
        causal_graph = torch.zeros(B, T, T, device=x.device)
        
        # 시간 단계 i와 j 간의 인과 관계
        for i in range(min(T, 10)):  # 계산 효율을 위해 일부만 샘플링
            for j in range(min(T, 10)):
                if i != j:
                    # i와 j 시간 단계의 특징 결합
                    pair = torch.cat([x[:, i, :], x[:, j, :]], dim=-1)  # [B, 2D]
                    causality_score = self.graph_predictor(pair)  # [B, 1]
                    causal_graph[:, i, j] = causality_score.squeeze(-1)
        
        # 대각선 제거 (자기 자신과의 인과 관계 없음)
        mask = torch.eye(T, device=x.device).unsqueeze(0).expand(B, T, T)
        causal_graph = causal_graph * (1 - mask)
        
        # DAG 제약 적용 (비순환성)
        causal_graph = self.enforce_dag_constraint(causal_graph, threshold)
        
        return causal_graph
    
    def enforce_dag_constraint(self, graph: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        DAG 제약 적용 (비순환성 보장)
        
        Args:
            graph: [B, N, N] - 인과 관계 행렬 (N은 T 또는 D)
            threshold: 임계값
        Returns:
            dag: [B, N, N] - DAG로 변환된 인과 관계 행렬
        """
        B, N, _ = graph.shape
        
        # 임계값 적용
        binary_graph = (graph > threshold).float()
        
        # 간단한 DAG 변환: 상삼각 행렬로 제한 (위상 정렬 가정)
        # 실제로는 더 정교한 알고리즘 필요 (예: NOTEARS)
        triu_mask = torch.triu(torch.ones(N, N, device=graph.device), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).expand(B, N, N)
        
        dag = graph * triu_mask
        
        return dag
    
    def granger_causality_test(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Granger 인과성 테스트 (신경망 기반)
        
        x가 y의 원인인지 테스트
        
        Args:
            x: [B, T, 1]
            y: [B, T, 1]
        Returns:
            causality_score: [B]
        """
        B, T, _ = x.shape
        
        # x와 y를 결합
        combined = torch.cat([x, y], dim=-1)  # [B, T, 2]
        
        # 시퀀스 트래커로 다음 시점 예측
        # y만 사용한 예측
        y_features = self.sequence_tracker(y)
        y_pred_without_x = y_features[:, -1:, :]  # [B, 1, D]
        
        # x와 y를 함께 사용한 예측
        combined_features = self.sequence_tracker(combined)
        y_pred_with_x = combined_features[:, -1:, :]  # [B, 1, D]
        
        # 실제 다음 시점 y 값 (간단히 마지막 값 사용)
        y_true = y[:, -1:, :]
        
        # 예측 개선 정도를 인과성 점수로 사용
        mse_without_x = F.mse_loss(y_pred_without_x, y_true, reduction='none').mean(dim=[1, 2])
        mse_with_x = F.mse_loss(y_pred_with_x, y_true, reduction='none').mean(dim=[1, 2])
        
        improvement = mse_without_x - mse_with_x
        causality_score = torch.sigmoid(improvement * 10)  # 스케일 조정
        
        return causality_score


def create_causality_detector(input_dim: int = 128) -> CausalityDetector:
    """Causality Detector 시드 생성 함수"""
    return CausalityDetector(input_dim=input_dim)

