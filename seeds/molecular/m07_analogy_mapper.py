"""
SEED-M07: Analogy Mapper

구조적 유사성을 매핑하여 도메인 간 유추 추론을 수행합니다.

Category: Analogy
Composed From: M01 (Hierarchy Builder) + A08 (Binary Comparator) + M05 (Concept Crystallizer)
Target Params: ~750K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from seeds.base import BaseSeed, SeedConfig


@dataclass
class AnalogyMapperConfig(SeedConfig):
    """Analogy Mapper 설정"""
    seed_id: str = "SEED-M07"
    name: str = "Analogy Mapper"
    level: int = 1
    category: str = "Analogy"
    bit_depth: str = "FP8"
    params: int = 750000
    input_dim: int = 128
    output_dim: int = 128
    hidden_dim: int = 192
    num_mapping_layers: int = 2
    dropout: float = 0.1
    similarity_threshold: float = 0.5


class AnalogyMapper(BaseSeed):
    """
    SEED-M07: Analogy Mapper
    
    구조적 유사성을 매핑하여 도메인 간 유추 추론을 수행합니다.
    
    주요 기능:
    - 계층적 구조 매칭 (M01 기반)
    - 개념 수준 유추 (M05 기반)
    - 유사도 평가 (A08 기반)
    - 구조 전이 (Structure Transfer)
    
    입력:
    - source_structure: 소스 구조 [B, N, D]
    - target_structure: 타겟 구조 [B, M, D]
    
    출력:
    - mapping: 매핑 결과 [B, N, D]
    - similarity_score: 유사도 점수 [B]
    - confidence: 신뢰도 [B]
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 192,
        num_mapping_layers: int = 2,
        dropout: float = 0.1,
        similarity_threshold: float = 0.5
    ):
        config = AnalogyMapperConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_mapping_layers=num_mapping_layers,
            dropout=dropout,
            similarity_threshold=similarity_threshold
        )
        super().__init__(config)
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mapping_layers = num_mapping_layers
        self.similarity_threshold = similarity_threshold
        
        # 1. Structure Encoder (M01 아이디어: 계층적 구조 인코딩)
        self.structure_encoder = nn.ModuleDict({
            'source': self._build_structure_encoder(),
            'target': self._build_structure_encoder()
        })
        
        # 2. Concept Matcher (M05 아이디어: 프로토타입 기반 매칭)
        self.concept_matcher = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Similarity Scorer (A08 아이디어: 비교 연산)
        self.similarity_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 4. Mapping Generator (구조 전이)
        mapping_layers = []
        for i in range(num_mapping_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            mapping_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        mapping_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mapping_generator = nn.Sequential(*mapping_layers)
        
        # 5. Attention mechanism for alignment (lightweight)
        self.alignment_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )
        
        # 6. Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def _build_structure_encoder(self) -> nn.Module:
        """계층적 구조 인코더 구축 (M01 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _compute_structural_similarity(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        구조적 유사도 계산 (A08 아이디어)
        
        Args:
            source_features: [B, N, D]
            target_features: [B, M, D]
        
        Returns:
            similarity_matrix: [B, N, M]
        """
        # Pairwise similarity computation
        batch_size, n_source, dim = source_features.shape
        _, n_target, _ = target_features.shape
        
        # Expand for pairwise comparison
        source_expanded = source_features.unsqueeze(2)  # [B, N, 1, D]
        target_expanded = target_features.unsqueeze(1)  # [B, 1, M, D]
        
        # Concatenate pairs
        pairs = torch.cat([
            source_expanded.expand(-1, -1, n_target, -1),
            target_expanded.expand(-1, n_source, -1, -1)
        ], dim=-1)  # [B, N, M, 2D]
        
        # Compute similarity scores
        similarity = self.similarity_scorer(pairs).squeeze(-1)  # [B, N, M]
        
        return similarity
    
    def _match_concepts(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        similarity_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        개념 매칭 (M05 아이디어: 프로토타입 기반)
        
        Args:
            source_features: [B, N, D]
            target_features: [B, M, D]
            similarity_matrix: [B, N, M]
        
        Returns:
            matched_features: [B, N, D]
            match_weights: [B, N, M]
        """
        # Soft matching using similarity as attention weights
        match_weights = F.softmax(similarity_matrix, dim=-1)  # [B, N, M]
        
        # Weighted sum of target features
        matched_features = torch.bmm(match_weights, target_features)  # [B, N, D]
        
        # Concept-level refinement
        combined = torch.cat([source_features, matched_features], dim=-1)  # [B, N, 2D]
        refined_features = self.concept_matcher(combined)  # [B, N, D]
        
        return refined_features, match_weights
    
    def _generate_mapping(
        self,
        source_features: torch.Tensor,
        matched_features: torch.Tensor,
        match_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        매핑 생성 (구조 전이)
        
        Args:
            source_features: [B, N, D]
            matched_features: [B, N, D]
            match_weights: [B, N, M]
        
        Returns:
            mapping: [B, N, input_dim]
        """
        # Attention-based alignment
        aligned_features, _ = self.alignment_attention(
            matched_features,
            source_features,
            source_features
        )  # [B, N, D]
        
        # Generate final mapping
        mapping = self.mapping_generator(aligned_features)  # [B, N, input_dim]
        
        return mapping
    
    def _compute_confidence(
        self,
        source_features: torch.Tensor,
        matched_features: torch.Tensor,
        similarity_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        신뢰도 계산
        
        Args:
            source_features: [B, N, D]
            matched_features: [B, N, D]
            similarity_matrix: [B, N, M]
        
        Returns:
            confidence: [B]
        """
        # Average similarity as base confidence
        avg_similarity = similarity_matrix.mean(dim=[1, 2])  # [B]
        
        # Feature-based confidence
        feature_diff = torch.abs(source_features - matched_features)  # [B, N, D]
        pooled_diff = feature_diff.mean(dim=1)  # [B, D]
        feature_confidence = self.confidence_estimator(pooled_diff).squeeze(-1)  # [B]
        
        # Combined confidence
        confidence = (avg_similarity + feature_confidence) / 2.0
        
        return confidence
    
    def forward(
        self,
        source_structure: torch.Tensor,
        target_structure: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            source_structure: 소스 구조 [B, N, input_dim]
            target_structure: 타겟 구조 [B, M, input_dim]
            scale: 스케일 매개변수 [B, 1]
            context: 추가 맥락 정보
        
        Returns:
            Dict containing:
                - mapping: 매핑 결과 [B, N, input_dim]
                - similarity_score: 유사도 점수 [B]
                - confidence: 신뢰도 [B]
                - match_weights: 매칭 가중치 [B, N, M]
        """
        # 1. Structure Encoding (M01 아이디어)
        source_features = self.structure_encoder['source'](source_structure)  # [B, N, hidden_dim]
        target_features = self.structure_encoder['target'](target_structure)  # [B, M, hidden_dim]
        
        # 2. Structural Similarity Computation (A08 아이디어)
        similarity_matrix = self._compute_structural_similarity(
            source_features,
            target_features
        )  # [B, N, M]
        
        # 3. Concept Matching (M05 아이디어)
        matched_features, match_weights = self._match_concepts(
            source_features,
            target_features,
            similarity_matrix
        )  # [B, N, hidden_dim], [B, N, M]
        
        # 4. Mapping Generation
        mapping = self._generate_mapping(
            source_features,
            matched_features,
            match_weights
        )  # [B, N, input_dim]
        
        # 5. Similarity Score (overall)
        similarity_score = similarity_matrix.max(dim=-1)[0].mean(dim=-1)  # [B]
        
        # 6. Confidence Estimation
        confidence = self._compute_confidence(
            source_features,
            matched_features,
            similarity_matrix
        )  # [B]
        
        return {
            'mapping': mapping,
            'similarity_score': similarity_score,
            'confidence': confidence,
            'match_weights': match_weights
        }
    
    def get_metadata(self) -> Dict:
        """시드 메타데이터 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'seed_id': self.config.seed_id,
            'name': self.config.name,
            'level': self.config.level,
            'category': self.config.category,
            'bit_depth': self.config.bit_depth,
            'target_params': self.config.params,
            'actual_params': total_params,
            'input_dim': self.input_dim,
            'output_dim': self.config.output_dim,
            'hidden_dim': self.hidden_dim,
            'num_mapping_layers': self.num_mapping_layers,
            'composed_from': ['M01', 'A08', 'M05'],
            'description': '구조적 유사성 매핑 및 유추 추론'
        }


# Convenience function
def create_analogy_mapper(
    input_dim: int = 128,
    hidden_dim: int = 256,
    **kwargs
) -> AnalogyMapper:
    """
    Analogy Mapper 생성 헬퍼 함수
    
    Args:
        input_dim: 입력 차원
        hidden_dim: 은닉 차원
        **kwargs: 추가 설정
    
    Returns:
        AnalogyMapper 인스턴스
    """
    return AnalogyMapper(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        **kwargs
    )
