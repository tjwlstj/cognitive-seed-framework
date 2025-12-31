"""
SEED-T01: Abductive Reasoner

귀추적 추론(Abductive Reasoning)을 통해 관찰된 현상에 대한 최선의 설명을 생성하는 Tissue 레벨 시드입니다.

구성 시드:
- M02: Causality Detector (인과 구조 파악)
- C02: Counterfactual Reasoner (반사실 추론)
- C03: Schema Learner (스키마 학습)
- C08: Novelty Assessor (참신성 평가)

주요 기능:
- 관찰 데이터로부터 가능한 설명 후보 생성
- 인과 구조 기반 설명 타당성 평가
- 스키마 기반 설명 일관성 검증
- 최선의 설명(Best Explanation) 선택
- 설명의 참신성 및 신뢰도 평가

Author: Manus AI
Date: 2025-12-31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from seeds.base import BaseSeed, SeedConfig


@dataclass
class AbductiveReasonerConfig(SeedConfig):
    """Abductive Reasoner 설정"""
    seed_id: str = "SEED-T01"
    name: str = "Abductive Reasoner"
    level: int = 3
    category: str = "Logic"
    bit_depth: str = "FP16"
    params: int = 12000000  # ~12M
    input_dim: int = 256
    output_dim: int = 256
    
    # T01 특화 설정
    num_hypotheses: int = 10  # 생성할 가설 후보 수
    max_explanation_length: int = 20  # 최대 설명 길이
    causality_weight: float = 0.4  # 인과성 가중치
    consistency_weight: float = 0.3  # 일관성 가중치
    novelty_weight: float = 0.2  # 참신성 가중치
    simplicity_weight: float = 0.1  # 단순성 가중치
    dropout: float = 0.1


class AbductiveReasoner(BaseSeed):
    """
    SEED-T01: Abductive Reasoner
    
    귀추적 추론을 통해 관찰된 현상에 대한 최선의 설명을 생성합니다.
    
    귀추적 추론(Abductive Reasoning)은 관찰된 현상으로부터 그것을 가장 잘 설명하는
    가설을 추론하는 과정입니다. 연역(Deduction)이나 귀납(Induction)과 달리,
    귀추는 불완전한 정보로부터 최선의 설명을 찾는 창의적 추론 과정입니다.
    
    주요 기능:
    - 관찰 데이터 인코딩 및 패턴 추출
    - 가능한 설명 후보(가설) 생성
    - 인과 구조 기반 타당성 평가
    - 스키마 기반 일관성 검증
    - 참신성 및 단순성 평가
    - 최선의 설명 선택 및 신뢰도 계산
    
    Examples:
        >>> reasoner = AbductiveReasoner(input_dim=256)
        >>> observations = torch.randn(4, 15, 256)  # 4 batches, 15 observations
        >>> explanation, confidence = reasoner(observations)
        >>> explanation.shape
        torch.Size([4, 20, 256])
        >>> confidence.shape
        torch.Size([4, 1])
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_hypotheses: int = 10,
        max_explanation_length: int = 20,
        causality_weight: float = 0.4,
        consistency_weight: float = 0.3,
        novelty_weight: float = 0.2,
        simplicity_weight: float = 0.1,
        dropout: float = 0.1
    ):
        config = AbductiveReasonerConfig(
            input_dim=input_dim,
            output_dim=input_dim,
            num_hypotheses=num_hypotheses,
            max_explanation_length=max_explanation_length,
            causality_weight=causality_weight,
            consistency_weight=consistency_weight,
            novelty_weight=novelty_weight,
            simplicity_weight=simplicity_weight,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        
        # 컴포넌트 초기화
        self._init_observation_encoder()
        self._init_hypothesis_generator()
        self._init_causality_evaluator()
        self._init_consistency_checker()
        self._init_novelty_assessor()
        self._init_explanation_selector()
    
    def _init_observation_encoder(self):
        """관찰 데이터 인코더 초기화"""
        dim = self.config.input_dim
        
        # MGP를 통한 다중 기하학 인코딩
        if self.config.use_mgp:
            self.obs_mgp = self.mgp
        
        # Transformer 기반 관찰 인코더
        self.obs_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=dim * 4,
                dropout=self.config.dropout,
                batch_first=True
            ),
            num_layers=3
        )
        
        # 관찰 요약 (pooling)
        self.obs_pooling = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
    
    def _init_hypothesis_generator(self):
        """가설 생성기 초기화"""
        dim = self.config.input_dim
        num_hyp = self.config.num_hypotheses
        max_len = self.config.max_explanation_length
        
        # 가설 시드 생성
        self.hypothesis_seeds = nn.Parameter(
            torch.randn(num_hyp, dim) * 0.02
        )
        
        # 가설 확장 네트워크 (관찰 → 가설)
        self.hypothesis_expander = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=dim * 4,
                dropout=self.config.dropout,
                batch_first=True
            ),
            num_layers=4
        )
        
        # 위치 인코딩
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_len, dim) * 0.02
        )
    
    def _init_causality_evaluator(self):
        """인과성 평가기 초기화 (M02 기반)"""
        dim = self.config.input_dim
        
        # 인과 구조 추출 네트워크
        self.causal_extractor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim, dim)
        )
        
        # 인과 타당성 평가
        self.causal_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def _init_consistency_checker(self):
        """일관성 검사기 초기화 (C03 기반)"""
        dim = self.config.input_dim
        
        # 스키마 추출 네트워크
        self.schema_extractor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim, dim)
        )
        
        # 일관성 평가
        self.consistency_scorer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def _init_novelty_assessor(self):
        """참신성 평가기 초기화 (C08 기반)"""
        dim = self.config.input_dim
        
        # 참조 분포 (학습된 일반적 설명 패턴)
        self.reference_distribution = nn.Parameter(
            torch.randn(100, dim) * 0.02
        )
        
        # 참신성 계산 네트워크
        self.novelty_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _init_explanation_selector(self):
        """설명 선택기 초기화"""
        dim = self.config.input_dim
        
        # 종합 평가 네트워크
        self.final_scorer = nn.Sequential(
            nn.Linear(4, 32),  # 4개 점수 (causality, consistency, novelty, simplicity)
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 설명 정제 네트워크
        self.explanation_refiner = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(dim, dim)
        )
    
    def encode_observations(
        self,
        observations: torch.Tensor,
        scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        관찰 데이터를 인코딩합니다.
        
        Args:
            observations: [B, N, D] 형태의 관찰 데이터
            scale: [B, 1] 형태의 스케일 매개변수
        
        Returns:
            encoded: [B, N, D] 인코딩된 관찰
            summary: [B, D] 관찰 요약
        """
        B, N, D = observations.shape
        
        # MGP 적용 (다중 기하학 투영)
        if self.config.use_mgp:
            encoded = self.obs_mgp(observations)
        else:
            encoded = observations
        
        # CSE 적용 (스케일 등변)
        if self.config.use_cse:
            encoded = self.cse(encoded, scale)
        
        # Transformer 인코딩
        encoded = self.obs_encoder(encoded)
        
        # 관찰 요약 (mean pooling + projection)
        summary = encoded.mean(dim=1)  # [B, D]
        summary = self.obs_pooling(summary)
        
        return encoded, summary
    
    def generate_hypotheses(
        self,
        obs_encoded: torch.Tensor,
        obs_summary: torch.Tensor
    ) -> torch.Tensor:
        """
        가능한 설명 가설들을 생성합니다.
        
        Args:
            obs_encoded: [B, N, D] 인코딩된 관찰
            obs_summary: [B, D] 관찰 요약
        
        Returns:
            hypotheses: [B, num_hyp, max_len, D] 생성된 가설들
        """
        B, N, D = obs_encoded.shape
        num_hyp = self.config.num_hypotheses
        max_len = self.config.max_explanation_length
        
        # 가설 시드를 배치 크기만큼 복제
        hyp_seeds = self.hypothesis_seeds.unsqueeze(0).expand(B, -1, -1)  # [B, num_hyp, D]
        
        # 위치 인코딩 추가
        pos_enc = self.pos_encoding.expand(B, -1, -1)  # [B, max_len, D]
        
        # 각 가설에 대해 설명 생성
        hypotheses = []
        for i in range(num_hyp):
            # 현재 가설 시드
            hyp_seed = hyp_seeds[:, i:i+1, :].expand(-1, max_len, -1)  # [B, max_len, D]
            
            # 위치 인코딩 추가
            hyp_input = hyp_seed + pos_enc
            
            # Transformer 디코더로 가설 확장
            hyp_expanded = self.hypothesis_expander(
                tgt=hyp_input,
                memory=obs_encoded
            )  # [B, max_len, D]
            
            hypotheses.append(hyp_expanded)
        
        # [B, num_hyp, max_len, D]
        hypotheses = torch.stack(hypotheses, dim=1)
        
        return hypotheses
    
    def evaluate_causality(
        self,
        hypotheses: torch.Tensor,
        obs_summary: torch.Tensor
    ) -> torch.Tensor:
        """
        가설의 인과적 타당성을 평가합니다.
        
        Args:
            hypotheses: [B, num_hyp, max_len, D] 가설들
            obs_summary: [B, D] 관찰 요약
        
        Returns:
            causality_scores: [B, num_hyp] 인과성 점수
        """
        B, num_hyp, max_len, D = hypotheses.shape
        
        # 가설 요약 (mean pooling)
        hyp_summary = hypotheses.mean(dim=2)  # [B, num_hyp, D]
        
        # 인과 구조 추출
        causal_features = self.causal_extractor(hyp_summary)  # [B, num_hyp, D]
        
        # 관찰과의 인과 관계 평가
        obs_expanded = obs_summary.unsqueeze(1).expand(-1, num_hyp, -1)  # [B, num_hyp, D]
        combined = torch.cat([causal_features, obs_expanded], dim=-1)  # [B, num_hyp, D*2]
        
        causality_scores = self.causal_scorer(combined).squeeze(-1)  # [B, num_hyp]
        
        return causality_scores
    
    def evaluate_consistency(
        self,
        hypotheses: torch.Tensor,
        obs_summary: torch.Tensor
    ) -> torch.Tensor:
        """
        가설의 스키마 일관성을 평가합니다.
        
        Args:
            hypotheses: [B, num_hyp, max_len, D] 가설들
            obs_summary: [B, D] 관찰 요약
        
        Returns:
            consistency_scores: [B, num_hyp] 일관성 점수
        """
        B, num_hyp, max_len, D = hypotheses.shape
        
        # 가설 요약
        hyp_summary = hypotheses.mean(dim=2)  # [B, num_hyp, D]
        
        # 스키마 추출
        schema_features = self.schema_extractor(hyp_summary)  # [B, num_hyp, D]
        
        # 관찰과의 일관성 평가
        obs_expanded = obs_summary.unsqueeze(1).expand(-1, num_hyp, -1)  # [B, num_hyp, D]
        combined = torch.cat([schema_features, obs_expanded], dim=-1)  # [B, num_hyp, D*2]
        
        consistency_scores = self.consistency_scorer(combined).squeeze(-1)  # [B, num_hyp]
        
        return consistency_scores
    
    def evaluate_novelty(
        self,
        hypotheses: torch.Tensor
    ) -> torch.Tensor:
        """
        가설의 참신성을 평가합니다.
        
        Args:
            hypotheses: [B, num_hyp, max_len, D] 가설들
        
        Returns:
            novelty_scores: [B, num_hyp] 참신성 점수
        """
        B, num_hyp, max_len, D = hypotheses.shape
        
        # 가설 요약
        hyp_summary = hypotheses.mean(dim=2)  # [B, num_hyp, D]
        
        # 참조 분포와의 거리 계산 (참신성)
        # 참조 분포: 학습된 일반적 설명 패턴
        ref_dist = self.reference_distribution.unsqueeze(0).unsqueeze(0)  # [1, 1, 100, D]
        hyp_expanded = hyp_summary.unsqueeze(2)  # [B, num_hyp, 1, D]
        
        # 코사인 유사도 계산
        similarity = F.cosine_similarity(
            hyp_expanded,
            ref_dist.expand(B, num_hyp, -1, -1),
            dim=-1
        )  # [B, num_hyp, 100]
        
        # 최대 유사도 (가장 유사한 참조 패턴과의 유사도)
        max_similarity = similarity.max(dim=-1)[0]  # [B, num_hyp]
        
        # 참신성 = 1 - 유사도 (유사도가 낮을수록 참신함)
        novelty_scores = 1.0 - max_similarity
        
        # 추가 네트워크를 통한 참신성 정제
        novelty_features = hyp_summary  # [B, num_hyp, D]
        novelty_refined = self.novelty_scorer(novelty_features).squeeze(-1)  # [B, num_hyp]
        
        # 두 점수의 평균
        novelty_scores = (novelty_scores + novelty_refined) / 2.0
        
        return novelty_scores
    
    def evaluate_simplicity(
        self,
        hypotheses: torch.Tensor
    ) -> torch.Tensor:
        """
        가설의 단순성을 평가합니다 (Occam's Razor).
        
        Args:
            hypotheses: [B, num_hyp, max_len, D] 가설들
        
        Returns:
            simplicity_scores: [B, num_hyp] 단순성 점수
        """
        B, num_hyp, max_len, D = hypotheses.shape
        
        # 가설의 복잡도 계산 (변동성 기반)
        # 변동성이 낮을수록 단순함
        variance = hypotheses.var(dim=2).mean(dim=-1)  # [B, num_hyp]
        
        # 정규화 (0~1 범위)
        variance_normalized = torch.sigmoid(-variance)
        
        # 가설 길이 기반 단순성 (짧을수록 단순함)
        # 실제 사용되는 길이 계산 (norm이 작은 토큰은 사용되지 않은 것으로 간주)
        norms = torch.norm(hypotheses, dim=-1)  # [B, num_hyp, max_len]
        active_length = (norms > 0.1).sum(dim=-1).float()  # [B, num_hyp]
        length_penalty = torch.sigmoid(-active_length / max_len)
        
        # 두 점수의 평균
        simplicity_scores = (variance_normalized + length_penalty) / 2.0
        
        return simplicity_scores
    
    def select_best_explanation(
        self,
        hypotheses: torch.Tensor,
        causality_scores: torch.Tensor,
        consistency_scores: torch.Tensor,
        novelty_scores: torch.Tensor,
        simplicity_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        최선의 설명을 선택합니다.
        
        Args:
            hypotheses: [B, num_hyp, max_len, D] 가설들
            causality_scores: [B, num_hyp] 인과성 점수
            consistency_scores: [B, num_hyp] 일관성 점수
            novelty_scores: [B, num_hyp] 참신성 점수
            simplicity_scores: [B, num_hyp] 단순성 점수
        
        Returns:
            best_explanation: [B, max_len, D] 최선의 설명
            confidence: [B, 1] 신뢰도
            all_scores: [B, num_hyp] 모든 가설의 종합 점수
        """
        B, num_hyp, max_len, D = hypotheses.shape
        
        # 4개 점수를 결합
        scores_combined = torch.stack([
            causality_scores,
            consistency_scores,
            novelty_scores,
            simplicity_scores
        ], dim=-1)  # [B, num_hyp, 4]
        
        # 가중 평균 (설정된 가중치 사용)
        weights = torch.tensor([
            self.config.causality_weight,
            self.config.consistency_weight,
            self.config.novelty_weight,
            self.config.simplicity_weight
        ], device=hypotheses.device)
        
        # 최종 점수 계산
        final_scores = self.final_scorer(scores_combined).squeeze(-1)  # [B, num_hyp]
        
        # 가중 평균도 함께 계산
        weighted_scores = (scores_combined * weights).sum(dim=-1)  # [B, num_hyp]
        
        # 두 점수의 평균
        all_scores = (final_scores + weighted_scores) / 2.0  # [B, num_hyp]
        
        # 최고 점수 가설 선택
        best_idx = all_scores.argmax(dim=1)  # [B]
        
        # 최선의 설명 추출
        best_explanation = hypotheses[torch.arange(B), best_idx]  # [B, max_len, D]
        
        # 신뢰도 계산 (최고 점수)
        confidence = all_scores[torch.arange(B), best_idx].unsqueeze(-1)  # [B, 1]
        
        # 설명 정제
        best_explanation = self.explanation_refiner(best_explanation)
        
        return best_explanation, confidence, all_scores
    
    def forward(
        self,
        observations: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        귀추적 추론을 수행하여 최선의 설명을 생성합니다.
        
        Args:
            observations: [B, N, D] 관찰 데이터
            scale: [B, 1] 스케일 매개변수
            context: 추가 맥락 정보
        
        Returns:
            explanation: [B, max_len, D] 최선의 설명
            confidence: [B, 1] 설명의 신뢰도
        """
        # 1. 관찰 인코딩
        obs_encoded, obs_summary = self.encode_observations(observations, scale)
        
        # 2. 가설 생성
        hypotheses = self.generate_hypotheses(obs_encoded, obs_summary)
        
        # 3. 가설 평가
        causality_scores = self.evaluate_causality(hypotheses, obs_summary)
        consistency_scores = self.evaluate_consistency(hypotheses, obs_summary)
        novelty_scores = self.evaluate_novelty(hypotheses)
        simplicity_scores = self.evaluate_simplicity(hypotheses)
        
        # 4. 최선의 설명 선택
        explanation, confidence, all_scores = self.select_best_explanation(
            hypotheses,
            causality_scores,
            consistency_scores,
            novelty_scores,
            simplicity_scores
        )
        
        return explanation, confidence
    
    def get_all_hypotheses(
        self,
        observations: torch.Tensor,
        scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        모든 가설과 평가 점수를 반환합니다 (디버깅 및 분석용).
        
        Args:
            observations: [B, N, D] 관찰 데이터
            scale: [B, 1] 스케일 매개변수
        
        Returns:
            hypotheses: [B, num_hyp, max_len, D] 모든 가설
            scores: 각 가설의 평가 점수 딕셔너리
        """
        # 1. 관찰 인코딩
        obs_encoded, obs_summary = self.encode_observations(observations, scale)
        
        # 2. 가설 생성
        hypotheses = self.generate_hypotheses(obs_encoded, obs_summary)
        
        # 3. 가설 평가
        causality_scores = self.evaluate_causality(hypotheses, obs_summary)
        consistency_scores = self.evaluate_consistency(hypotheses, obs_summary)
        novelty_scores = self.evaluate_novelty(hypotheses)
        simplicity_scores = self.evaluate_simplicity(hypotheses)
        
        # 4. 종합 점수 계산
        scores_combined = torch.stack([
            causality_scores,
            consistency_scores,
            novelty_scores,
            simplicity_scores
        ], dim=-1)  # [B, num_hyp, 4]
        
        weights = torch.tensor([
            self.config.causality_weight,
            self.config.consistency_weight,
            self.config.novelty_weight,
            self.config.simplicity_weight
        ], device=hypotheses.device)
        
        final_scores = (scores_combined * weights).sum(dim=-1)  # [B, num_hyp]
        
        scores = {
            "causality": causality_scores,
            "consistency": consistency_scores,
            "novelty": novelty_scores,
            "simplicity": simplicity_scores,
            "final": final_scores
        }
        
        return hypotheses, scores


# 편의 함수
def create_abductive_reasoner(
    input_dim: int = 256,
    **kwargs
) -> AbductiveReasoner:
    """
    Abductive Reasoner 인스턴스를 생성합니다.
    
    Args:
        input_dim: 입력 차원
        **kwargs: 추가 설정
    
    Returns:
        AbductiveReasoner 인스턴스
    """
    return AbductiveReasoner(input_dim=input_dim, **kwargs)
