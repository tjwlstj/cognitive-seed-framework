"""
SEED-T01: Abductive Reasoner

최선 설명 추론(Abductive Reasoning)을 수행하는 Tissue 레벨 시드입니다.
주어진 관찰 결과에 대해 가장 그럴듯한 설명을 추론합니다.

구성 시드:
- M02: Causality Detector (인과 관계 추정)
- M08: Conflict Resolver (제약 충돌 해소)
- M05: Concept Crystallizer (개념 추상화)
- C02: Counterfactual Reasoner (반사실 추론)

주요 기능:
- 관찰 데이터로부터 가설 생성
- 인과 구조 기반 설명 추론
- 반사실 추론을 통한 가설 검증
- 최선의 설명 선택 및 평가

Author: Manus AI (누스양)
Date: 2026-01-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from seeds.base import BaseSeed, SeedConfig
from seeds.molecular.m02_causality_detector import CausalityDetector
from seeds.molecular.m08_conflict_resolver import ConflictResolver
from seeds.molecular.m05_concept_crystallizer import ConceptCrystallizer
from seeds.cellular.c02_counterfactual_reasoner import CounterfactualReasoner


@dataclass
class AbductiveReasonerConfig(SeedConfig):
    """Abductive Reasoner 설정"""
    seed_id: str = "SEED-T01"
    name: str = "Abductive Reasoner"
    level: int = 3
    category: str = "Logic"
    bit_depth: str = "FP8"
    params: int = 3000000  # ~3.0M
    input_dim: int = 128
    output_dim: int = 128
    
    # T01 특화 설정
    num_hypotheses: int = 8  # 생성할 가설 수
    max_explanation_length: int = 10  # 최대 설명 길이
    plausibility_threshold: float = 0.6  # 그럴듯함 임계값
    consistency_weight: float = 0.3
    parsimony_weight: float = 0.2  # 간결성 가중치
    dropout: float = 0.1


class T01AbductiveReasoner(BaseSeed):
    """
    SEED-T01: Abductive Reasoner
    
    Category: Logic
    Level: 3 (Tissue)
    Bit: FP8
    Params: ~3.0M
    Purpose: 관찰로부터 최선 설명 추론 (Abductive Reasoning)
    I/O: [B, T, D] → [B, T, D]
    Composed From: M02 + M08 + M05 + C02
    
    Abductive Reasoning:
    - 관찰: "잔디가 젖어있다"
    - 가설 1: "비가 왔다" (가능성 높음)
    - 가설 2: "스프링클러가 작동했다" (가능성 중간)
    - 가설 3: "이슬이 맺혔다" (가능성 낮음)
    - 최선 설명: "비가 왔다" (가장 그럴듯한 설명)
    """
    
    def __init__(self, config: Optional[AbductiveReasonerConfig] = None):
        if config is None:
            config = AbductiveReasonerConfig()
        super().__init__(config)
        
        self.config = config
        dim = config.input_dim
        
        # ===== Composed Seeds =====
        # M02: 인과 관계 추정
        self.causality_detector = CausalityDetector(input_dim=dim)
        
        # M08: 제약 충돌 해소
        self.conflict_resolver = ConflictResolver(input_dim=dim)
        
        # M05: 개념 추상화
        self.concept_crystallizer = ConceptCrystallizer(input_dim=dim)
        
        # C02: 반사실 추론
        self.counterfactual_reasoner = CounterfactualReasoner()
        
        # ===== Observation Encoder =====
        # 관찰 데이터를 인코딩하여 추론에 적합한 표현으로 변환
        self.observation_encoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 2, dim)
        )
        
        # ===== Hypothesis Generator =====
        # 관찰로부터 가능한 가설들을 생성
        self.hypothesis_generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            ) for _ in range(config.num_hypotheses)
        ])
        
        # ===== Explanation Scorer =====
        # 각 가설의 설명력을 평가
        self.explanation_scorer = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),  # observation + hypothesis + causal
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),  # 설명력 스코어
            nn.Sigmoid()
        )
        
        # ===== Plausibility Evaluator =====
        # 가설의 그럴듯함(plausibility) 평가
        self.plausibility_evaluator = nn.Sequential(
            nn.Linear(dim * 2, dim),  # hypothesis + counterfactual
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # ===== Parsimony Evaluator =====
        # 설명의 간결성(parsimony) 평가 - Occam's Razor
        self.parsimony_evaluator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # ===== Best Explanation Selector =====
        # 최선의 설명을 선택하는 어텐션 메커니즘
        self.explanation_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # ===== Output Projection =====
        self.output_projection = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim * 2, dim)
        )
        
        # ===== Residual Connection =====
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hypotheses: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: 입력 텐서 [B, T, D] - 관찰 데이터
            mask: 마스크 텐서 [B, T] (optional)
            return_hypotheses: 가설 정보 반환 여부
            
        Returns:
            output: 최선 설명 텐서 [B, T, D]
            hypotheses_info: 가설 정보 (return_hypotheses=True일 때)
        """
        B, T, D = x.shape
        
        # ===== Step 1: Observation Encoding =====
        # 관찰 데이터를 인코딩
        observation = self.observation_encoder(x)  # [B, T, D]
        
        # ===== Step 2: Causal Structure Detection =====
        # 인과 구조 파악 (M02)
        causal_structure = self.causality_detector(observation)  # [B, T, D]
        
        # ===== Step 3: Concept Abstraction =====
        # 개념 추상화 (M05)
        abstract_concepts = self.concept_crystallizer(observation)  # [B, T, D]
        
        # ===== Step 4: Hypothesis Generation =====
        # 여러 가설 생성
        hypotheses = []
        for hypothesis_net in self.hypothesis_generator:
            h = hypothesis_net(observation)  # [B, T, D]
            hypotheses.append(h)
        hypotheses = torch.stack(hypotheses, dim=2)  # [B, T, num_hypotheses, D]
        
        # ===== Step 5: Counterfactual Reasoning =====
        # 반사실 추론으로 가설 검증 (C02)
        # 각 가설에 대해 반사실 시나리오 생성
        counterfactuals = []
        for i in range(self.config.num_hypotheses):
            h = hypotheses[:, :, i, :]  # [B, T, D]
            cf = self.counterfactual_reasoner(h)  # [B, T, D]
            counterfactuals.append(cf)
        counterfactuals = torch.stack(counterfactuals, dim=2)  # [B, T, num_hypotheses, D]
        
        # ===== Step 6: Explanation Scoring =====
        # 각 가설의 설명력 평가
        explanation_scores = []
        for i in range(self.config.num_hypotheses):
            h = hypotheses[:, :, i, :]  # [B, T, D]
            # observation + hypothesis + causal_structure 결합
            combined = torch.cat([observation, h, causal_structure], dim=-1)  # [B, T, 3*D]
            score = self.explanation_scorer(combined)  # [B, T, 1]
            explanation_scores.append(score)
        explanation_scores = torch.stack(explanation_scores, dim=2)  # [B, T, num_hypotheses, 1]
        
        # ===== Step 7: Plausibility Evaluation =====
        # 가설의 그럴듯함 평가
        plausibility_scores = []
        for i in range(self.config.num_hypotheses):
            h = hypotheses[:, :, i, :]  # [B, T, D]
            cf = counterfactuals[:, :, i, :]  # [B, T, D]
            combined = torch.cat([h, cf], dim=-1)  # [B, T, 2*D]
            score = self.plausibility_evaluator(combined)  # [B, T, 1]
            plausibility_scores.append(score)
        plausibility_scores = torch.stack(plausibility_scores, dim=2)  # [B, T, num_hypotheses, 1]
        
        # ===== Step 8: Parsimony Evaluation =====
        # 설명의 간결성 평가 (Occam's Razor)
        parsimony_scores = []
        for i in range(self.config.num_hypotheses):
            h = hypotheses[:, :, i, :]  # [B, T, D]
            score = self.parsimony_evaluator(h)  # [B, T, 1]
            parsimony_scores.append(score)
        parsimony_scores = torch.stack(parsimony_scores, dim=2)  # [B, T, num_hypotheses, 1]
        
        # ===== Step 9: Combined Scoring =====
        # 설명력, 그럴듯함, 간결성을 종합하여 최종 스코어 계산
        combined_scores = (
            explanation_scores * (1 - self.config.consistency_weight - self.config.parsimony_weight) +
            plausibility_scores * self.config.consistency_weight +
            parsimony_scores * self.config.parsimony_weight
        )  # [B, T, num_hypotheses, 1]
        
        # ===== Step 10: Conflict Resolution =====
        # 가설 간 충돌 해소 (M08)
        # 모든 가설을 하나의 시퀀스로 결합
        hypotheses_flat = hypotheses.reshape(B, T * self.config.num_hypotheses, D)
        resolved_hypotheses = self.conflict_resolver(hypotheses_flat)  # [B, T*num_hypotheses, D]
        resolved_hypotheses = resolved_hypotheses.reshape(B, T, self.config.num_hypotheses, D)
        
        # ===== Step 11: Best Explanation Selection =====
        # 어텐션 메커니즘으로 최선의 설명 선택
        # Query: observation, Key/Value: resolved_hypotheses
        hypotheses_for_attention = resolved_hypotheses.reshape(B * T, self.config.num_hypotheses, D)
        observation_for_attention = observation.reshape(B * T, 1, D)
        
        best_explanation, attention_weights = self.explanation_attention(
            query=observation_for_attention,
            key=hypotheses_for_attention,
            value=hypotheses_for_attention
        )  # [B*T, 1, D], [B*T, 1, num_hypotheses]
        
        best_explanation = best_explanation.reshape(B, T, D)  # [B, T, D]
        attention_weights = attention_weights.reshape(B, T, self.config.num_hypotheses)  # [B, T, num_hypotheses]
        
        # ===== Step 12: Output Projection =====
        output = self.output_projection(best_explanation)  # [B, T, D]
        
        # ===== Step 13: Residual Connection =====
        output = output + self.residual_weight * x
        
        # ===== Return =====
        if return_hypotheses:
            hypotheses_info = {
                'hypotheses': hypotheses,  # [B, T, num_hypotheses, D]
                'counterfactuals': counterfactuals,  # [B, T, num_hypotheses, D]
                'explanation_scores': explanation_scores,  # [B, T, num_hypotheses, 1]
                'plausibility_scores': plausibility_scores,  # [B, T, num_hypotheses, 1]
                'parsimony_scores': parsimony_scores,  # [B, T, num_hypotheses, 1]
                'combined_scores': combined_scores,  # [B, T, num_hypotheses, 1]
                'attention_weights': attention_weights,  # [B, T, num_hypotheses]
                'causal_structure': causal_structure,  # [B, T, D]
                'abstract_concepts': abstract_concepts  # [B, T, D]
            }
            return output, hypotheses_info
        
        return output
    
    def get_best_hypothesis_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        최선의 가설 인덱스 반환
        
        Args:
            x: 입력 텐서 [B, T, D]
            
        Returns:
            best_indices: 최선 가설 인덱스 [B, T]
        """
        _, hypotheses_info = self.forward(x, return_hypotheses=True)
        attention_weights = hypotheses_info['attention_weights']  # [B, T, num_hypotheses]
        best_indices = torch.argmax(attention_weights, dim=-1)  # [B, T]
        return best_indices
    
    def get_explanation_quality(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        설명 품질 지표 반환
        
        Args:
            x: 입력 텐서 [B, T, D]
            
        Returns:
            quality_metrics: 품질 지표 딕셔너리
        """
        _, hypotheses_info = self.forward(x, return_hypotheses=True)
        
        # 최선 가설의 스코어 추출
        attention_weights = hypotheses_info['attention_weights']  # [B, T, num_hypotheses]
        best_indices = torch.argmax(attention_weights, dim=-1)  # [B, T]
        
        B, T = best_indices.shape
        
        # 각 스코어에서 최선 가설의 값 추출
        explanation_scores = hypotheses_info['explanation_scores'].squeeze(-1)  # [B, T, num_hypotheses]
        plausibility_scores = hypotheses_info['plausibility_scores'].squeeze(-1)
        parsimony_scores = hypotheses_info['parsimony_scores'].squeeze(-1)
        combined_scores = hypotheses_info['combined_scores'].squeeze(-1)
        
        best_explanation_score = torch.gather(
            explanation_scores, 2, best_indices.unsqueeze(-1)
        ).squeeze(-1)  # [B, T]
        
        best_plausibility_score = torch.gather(
            plausibility_scores, 2, best_indices.unsqueeze(-1)
        ).squeeze(-1)
        
        best_parsimony_score = torch.gather(
            parsimony_scores, 2, best_indices.unsqueeze(-1)
        ).squeeze(-1)
        
        best_combined_score = torch.gather(
            combined_scores, 2, best_indices.unsqueeze(-1)
        ).squeeze(-1)
        
        return {
            'explanation_score': best_explanation_score,  # [B, T]
            'plausibility_score': best_plausibility_score,  # [B, T]
            'parsimony_score': best_parsimony_score,  # [B, T]
            'combined_score': best_combined_score,  # [B, T]
            'confidence': attention_weights.max(dim=-1)[0]  # [B, T] - 최대 어텐션 가중치
        }


# ===== Factory Function =====
def create_t01_abductive_reasoner(
    input_dim: int = 128,
    num_hypotheses: int = 8,
    **kwargs
) -> T01AbductiveReasoner:
    """
    T01 Abductive Reasoner 생성 팩토리 함수
    
    Args:
        input_dim: 입력 차원
        num_hypotheses: 생성할 가설 수
        **kwargs: 추가 설정
        
    Returns:
        T01AbductiveReasoner 인스턴스
    """
    config = AbductiveReasonerConfig(
        input_dim=input_dim,
        output_dim=input_dim,
        num_hypotheses=num_hypotheses,
        **kwargs
    )
    return T01AbductiveReasoner(config)
