"""
SEED-C05: Narrative Constructor

서사 구조화를 통해 이야기를 생성하고 인과 관계를 연결하는 Cellular 레벨 시드입니다.

Category: Composition
Composed From: M06 (Context Integrator) + M03 (Pattern Completer) + A06 (Sequence Tracker)
Target Params: ~1.0M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from seeds.base import BaseSeed, SeedConfig


@dataclass
class NarrativeConstructorConfig(SeedConfig):
    """Narrative Constructor 설정"""
    seed_id: str = "SEED-C05"
    name: str = "Narrative Constructor"
    level: int = 2
    category: str = "Composition"
    bit_depth: str = "FP8"
    params: int = 1000000
    input_dim: int = 128
    output_dim: int = 128
    hidden_dim: int = 192
    num_narrative_stages: int = 4  # 기승전결
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1
    causality_weight: float = 0.4


class NarrativeConstructor(BaseSeed):
    """
    SEED-C05: Narrative Constructor
    
    서사 구조화를 통해 이야기를 생성하고 인과 관계를 연결합니다.
    
    주요 기능:
    - 서사 구조 생성 (기승전결: 도입-전개-위기-결말)
    - 인과 관계 연결 (M06 기반 맥락 통합)
    - 시간적 일관성 유지 (A06 기반 시퀀스 추적)
    - 서사 완성도 평가 (M03 기반 패턴 완성)
    - 플롯 포인트 식별
    
    입력:
    - events: 사건 시퀀스 [B, L, D]
    - context: 맥락 정보 (선택적)
    
    출력:
    - narrative: 구조화된 서사 [B, L, D]
    - structure: 서사 구조 정보
    - coherence_score: 일관성 점수 [B]
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 192,
        num_narrative_stages: int = 4,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        causality_weight: float = 0.4
    ):
        config = NarrativeConstructorConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_narrative_stages=num_narrative_stages,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            causality_weight=causality_weight
        )
        super().__init__(config)
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_narrative_stages = num_narrative_stages
        self.num_heads = num_heads
        self.causality_weight = causality_weight
        
        # 1. Context Integrator (M06 아이디어: 맥락 통합)
        self.context_encoder = self._build_context_encoder()
        
        # 2. Sequence Tracker (A06 아이디어: 시간적 순서)
        self.temporal_encoder = self._build_temporal_encoder()
        
        # 3. Pattern Completer (M03 아이디어: 서사 완성)
        self.pattern_completer = self._build_pattern_completer()
        
        # 4. Narrative Stage Classifier (기승전결 분류)
        self.stage_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_narrative_stages),
            nn.Softmax(dim=-1)
        )
        
        # 5. Causality Network (인과 관계 모델링)
        self.causality_network = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 6. Narrative Structure Builder
        self.structure_builder = nn.ModuleList([
            self._build_stage_encoder() for _ in range(num_narrative_stages)
        ])
        
        # 7. Plot Point Detector
        self.plot_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 8. Coherence Evaluator
        self.coherence_evaluator = nn.Sequential(
            nn.Linear(hidden_dim * num_narrative_stages, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 9. Narrative Fusion
        self.narrative_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 10. Positional Encoding
        self.register_buffer(
            'pos_encoding',
            self._create_positional_encoding(5000, hidden_dim)
        )
    
    def _build_context_encoder(self) -> nn.Module:
        """맥락 인코더 (M06 아이디어)"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.config.dropout,
            batch_first=True
        )
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.TransformerEncoder(encoder_layer, num_layers=2)
        )
    
    def _build_temporal_encoder(self) -> nn.Module:
        """시간적 인코더 (A06 아이디어)"""
        return nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.config.dropout
        )
    
    def _build_pattern_completer(self) -> nn.Module:
        """패턴 완성 네트워크 (M03 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_stage_encoder(self) -> nn.Module:
        """개별 서사 단계 인코더"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _create_positional_encoding(
        self,
        max_len: int,
        d_model: int
    ) -> torch.Tensor:
        """위치 인코딩 생성"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict, torch.Tensor]:
        """
        Args:
            x: [B, L, D] - 입력 사건 시퀀스
            context: [B, C, D] - 맥락 정보 (선택적)
            scale: [B, 1] - 스케일 매개변수
        
        Returns:
            narrative: [B, L, D] - 구조화된 서사
            structure: 서사 구조 정보 딕셔너리
            coherence_score: [B] - 일관성 점수
        """
        batch_size, seq_len, dim = x.shape
        
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 1. 맥락 통합 (M06 아이디어)
        context_features = self.context_encoder(x)  # [B, L, hidden_dim]
        
        # 위치 인코딩 추가
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0)  # [1, L, hidden_dim]
        context_features = context_features + pos_enc
        
        # 2. 시간적 순서 추적 (A06 아이디어)
        temporal_features, _ = self.temporal_encoder(context_features)  # [B, L, hidden_dim]
        
        # 3. 인과 관계 모델링
        causal_features, causal_weights = self.causality_network(
            temporal_features,
            temporal_features,
            temporal_features
        )  # [B, L, hidden_dim], [B, L, L]
        
        # 4. 서사 단계 분류 (기승전결)
        stage_probs = self.stage_classifier(causal_features)  # [B, L, num_stages]
        stage_assignments = torch.argmax(stage_probs, dim=-1)  # [B, L]
        
        # 5. 각 단계별 특징 추출
        stage_features = []
        for stage_idx, stage_encoder in enumerate(self.structure_builder):
            # 해당 단계에 속하는 토큰들의 가중치
            stage_mask = (stage_assignments == stage_idx).float().unsqueeze(-1)  # [B, L, 1]
            
            # 단계별 특징 추출
            stage_feat = stage_encoder(causal_features)  # [B, L, hidden_dim]
            weighted_feat = stage_feat * stage_mask
            
            # 단계별 대표 특징 (평균 풀링)
            stage_rep = weighted_feat.sum(dim=1) / (stage_mask.sum(dim=1) + 1e-8)  # [B, hidden_dim]
            stage_features.append(stage_rep)
        
        stage_features_tensor = torch.stack(stage_features, dim=1)  # [B, num_stages, hidden_dim]
        
        # 6. 플롯 포인트 탐지
        plot_points = self.plot_detector(causal_features).squeeze(-1)  # [B, L]
        
        # 7. 패턴 완성 (M03 아이디어)
        completed_features = self.pattern_completer(causal_features)  # [B, L, hidden_dim]
        
        # 8. 서사 융합
        combined = torch.cat([causal_features, completed_features], dim=-1)  # [B, L, hidden_dim*2]
        narrative = self.narrative_fusion(combined)  # [B, L, input_dim]
        
        # 9. 일관성 평가
        stage_concat = stage_features_tensor.view(batch_size, -1)  # [B, num_stages*hidden_dim]
        coherence_score = self.coherence_evaluator(stage_concat).squeeze(-1)  # [B]
        
        # 서사 구조 정보
        structure = {
            'stage_probabilities': stage_probs,  # [B, L, num_stages]
            'stage_assignments': stage_assignments,  # [B, L]
            'stage_features': stage_features_tensor,  # [B, num_stages, hidden_dim]
            'causal_weights': causal_weights,  # [B, L, L]
            'plot_points': plot_points,  # [B, L]
            'coherence_score': coherence_score  # [B]
        }
        
        return narrative, structure, coherence_score
    
    def analyze_narrative_structure(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        서사 구조 상세 분석
        
        Args:
            x: [B, L, D] - 입력 사건 시퀀스
        
        Returns:
            analysis: 분석 결과 딕셔너리
        """
        narrative, structure, coherence_score = self.forward(x)
        
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 각 단계의 길이 계산
        stage_lengths = []
        for stage_idx in range(self.num_narrative_stages):
            stage_mask = (structure['stage_assignments'] == stage_idx).float()
            stage_len = stage_mask.sum(dim=1)  # [B]
            stage_lengths.append(stage_len)
        
        stage_lengths_tensor = torch.stack(stage_lengths, dim=1)  # [B, num_stages]
        
        # 플롯 포인트 위치
        plot_point_indices = []
        for b in range(batch_size):
            indices = torch.where(structure['plot_points'][b] > 0.5)[0]
            plot_point_indices.append(indices)
        
        # 인과 관계 강도
        causal_strength = structure['causal_weights'].mean(dim=-1)  # [B, L]
        
        analysis = {
            'narrative': narrative,
            'stage_assignments': structure['stage_assignments'],
            'stage_lengths': stage_lengths_tensor,
            'plot_points': structure['plot_points'],
            'plot_point_indices': plot_point_indices,
            'causal_strength': causal_strength,
            'coherence_score': coherence_score,
            'stage_transitions': self._detect_stage_transitions(structure['stage_assignments'])
        }
        
        return analysis
    
    def _detect_stage_transitions(
        self,
        stage_assignments: torch.Tensor
    ) -> List[List[int]]:
        """
        서사 단계 전환점 탐지
        
        Args:
            stage_assignments: [B, L] - 단계 할당
        
        Returns:
            transitions: 배치별 전환점 인덱스 리스트
        """
        batch_size, seq_len = stage_assignments.shape
        transitions = []
        
        for b in range(batch_size):
            batch_transitions = []
            for i in range(1, seq_len):
                if stage_assignments[b, i] != stage_assignments[b, i-1]:
                    batch_transitions.append(i)
            transitions.append(batch_transitions)
        
        return transitions
    
    def generate_narrative(
        self,
        events: torch.Tensor,
        target_structure: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        목표 구조에 맞춰 서사 생성
        
        Args:
            events: [B, L, D] - 입력 사건들
            target_structure: 목표 단계 비율 (예: [0.25, 0.25, 0.25, 0.25])
        
        Returns:
            narrative: [B, L, D] - 생성된 서사
            info: 생성 정보
        """
        if target_structure is None:
            target_structure = [1.0 / self.num_narrative_stages] * self.num_narrative_stages
        
        narrative, structure, coherence_score = self.forward(events)
        
        # 현재 구조 계산
        batch_size, seq_len = events.shape[0], events.shape[1]
        current_structure = []
        for stage_idx in range(self.num_narrative_stages):
            stage_mask = (structure['stage_assignments'] == stage_idx).float()
            stage_ratio = stage_mask.sum(dim=1) / seq_len  # [B]
            current_structure.append(stage_ratio)
        
        current_structure_tensor = torch.stack(current_structure, dim=1)  # [B, num_stages]
        
        # 목표 구조와의 차이
        target_tensor = torch.tensor(
            target_structure,
            device=events.device
        ).unsqueeze(0).expand(batch_size, -1)  # [B, num_stages]
        
        structure_diff = torch.abs(current_structure_tensor - target_tensor).mean(dim=1)  # [B]
        
        info = {
            'narrative': narrative,
            'structure': structure,
            'coherence_score': coherence_score,
            'current_structure': current_structure_tensor,
            'target_structure': target_tensor,
            'structure_difference': structure_diff
        }
        
        return narrative, info
    
    def evaluate_coherence(
        self,
        narrative: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        서사 일관성 평가
        
        Args:
            narrative: [B, L, D] - 서사
        
        Returns:
            evaluation: 평가 결과
        """
        _, structure, coherence_score = self.forward(narrative)
        
        # 인과 관계 밀도
        causal_density = structure['causal_weights'].mean(dim=[1, 2])  # [B]
        
        # 플롯 포인트 밀도
        plot_density = structure['plot_points'].mean(dim=1)  # [B]
        
        # 단계 균형도
        stage_balance = structure['stage_probabilities'].std(dim=1).mean(dim=1)  # [B]
        
        evaluation = {
            'coherence_score': coherence_score,
            'causal_density': causal_density,
            'plot_density': plot_density,
            'stage_balance': stage_balance,
            'overall_quality': (coherence_score + causal_density + plot_density) / 3.0
        }
        
        return evaluation


def create_narrative_constructor(
    input_dim: int = 128,
    hidden_dim: int = 192,
    num_narrative_stages: int = 4
) -> NarrativeConstructor:
    """Narrative Constructor 시드 생성 함수"""
    return NarrativeConstructor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_narrative_stages=num_narrative_stages
    )
