"""
SEED-C08: Novelty Assessor

참신성을 평가하여 새로운 개념과 기존 개념 간의 차이를 정량화합니다.

Category: Evaluation
Composed From: M05 (Concept Crystallizer) + M07 (Analogy Mapper) + A04 (Contrast Amplifier)
Target Params: ~1.5M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from seeds.base import BaseSeed, SeedConfig


@dataclass
class NoveltyAssessorConfig(SeedConfig):
    """Novelty Assessor 설정"""
    seed_id: str = "SEED-C08"
    name: str = "Novelty Assessor"
    level: int = 2
    category: str = "Evaluation"
    bit_depth: str = "FP8"
    params: int = 1500000
    input_dim: int = 128
    output_dim: int = 128
    hidden_dim: int = 256
    num_reference_concepts: int = 10
    novelty_dimensions: int = 3  # 구조적, 의미적, 기능적
    dropout: float = 0.1


class NoveltyAssessor(BaseSeed):
    """
    SEED-C08: Novelty Assessor
    
    참신성을 평가하여 새로운 개념과 기존 개념 간의 차이를 정량화합니다.
    
    주요 기능:
    - M05 기반 개념 추출 및 프로토타입 학습
    - M07 기반 구조적 유사성 분석
    - A04 기반 차이점 강조
    - 다차원 참신성 평가 (구조적/의미적/기능적)
    
    입력:
    - input_concept: 평가할 새로운 개념 [B, D]
    - reference_concepts: 기존 개념들 [B, N, D]
    
    출력:
    - novelty_score: 참신성 점수 [B] (0~1)
    - novelty_dimensions: 차원별 참신성 [B, 3]
    - explanation: 참신성 설명 정보
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_reference_concepts: int = 10,
        novelty_dimensions: int = 3,
        dropout: float = 0.1
    ):
        config = NoveltyAssessorConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_reference_concepts=num_reference_concepts,
            novelty_dimensions=novelty_dimensions,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_reference_concepts = num_reference_concepts
        self.novelty_dimensions = novelty_dimensions
        
        # 1. Concept Extractor (M05 아이디어: 프로토타입 학습)
        self.concept_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. Prototype Encoder (M05 아이디어: 기존 개념 표현)
        self.prototype_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. Similarity Analyzer (M07 아이디어: 구조적 유사성 매핑)
        self.similarity_analyzer = nn.ModuleDict({
            'structural': self._build_similarity_module(),
            'semantic': self._build_similarity_module(),
            'functional': self._build_similarity_module()
        })
        
        # 4. Contrast Amplifier (A04 아이디어: 차이점 강조)
        self.contrast_amplifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 5. Difference Extractor (차이점 추출)
        self.difference_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 6. Novelty Scorer (참신성 점수 계산)
        self.novelty_scorer = nn.Sequential(
            nn.Linear(hidden_dim + novelty_dimensions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 7. Dimension-specific Scorers (차원별 참신성 평가)
        self.dimension_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(novelty_dimensions)
        ])
        
        # 8. Attention mechanism for weighted comparison
        self.comparison_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 9. Explanation Generator (설명 생성)
        self.explanation_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable novelty threshold
        self.novelty_threshold = nn.Parameter(torch.tensor(0.5))
    
    def _build_similarity_module(self) -> nn.Module:
        """유사도 분석 모듈 구축"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def extract_concept(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력으로부터 개념 추출 (M05 아이디어)
        
        Args:
            x: [B, D] 또는 [B, L, D]
        
        Returns:
            concept: [B, H]
        """
        original_shape = x.shape
        
        # 2D로 변환
        if len(original_shape) == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
        
        B, L, D = x.shape
        
        # Concept extraction
        x_flat = x.view(B * L, D)
        concept = self.concept_extractor(x_flat)
        concept = concept.view(B, L, -1)
        
        # Average pooling
        concept = concept.mean(dim=1)  # [B, H]
        
        return concept
    
    def encode_prototypes(
        self, 
        reference_concepts: torch.Tensor
    ) -> torch.Tensor:
        """
        기존 개념들을 프로토타입으로 인코딩 (M05 아이디어)
        
        Args:
            reference_concepts: [B, N, D]
        
        Returns:
            prototypes: [B, N, H]
        """
        B, N, D = reference_concepts.shape
        
        # Flatten
        ref_flat = reference_concepts.view(B * N, D)
        
        # Encode
        prototypes = self.prototype_encoder(ref_flat)
        
        # Reshape
        prototypes = prototypes.view(B, N, -1)
        
        return prototypes
    
    def compute_dimensional_similarity(
        self,
        concept: torch.Tensor,
        prototypes: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        다차원 유사도 계산 (M07 아이디어)
        
        Args:
            concept: [B, H]
            prototypes: [B, N, H]
        
        Returns:
            similarities: 차원별 유사도 딕셔너리
        """
        B, N, H = prototypes.shape
        
        # Expand concept for comparison
        concept_expanded = concept.unsqueeze(1).expand(B, N, H)  # [B, N, H]
        
        # Concatenate for similarity computation
        combined = torch.cat([concept_expanded, prototypes], dim=-1)  # [B, N, 2H]
        
        similarities = {}
        for dim_name, analyzer in self.similarity_analyzer.items():
            # Flatten for processing
            combined_flat = combined.view(B * N, -1)
            sim = analyzer(combined_flat)  # [B*N, 1]
            sim = sim.view(B, N)  # [B, N]
            similarities[dim_name] = sim
        
        return similarities
    
    def amplify_differences(
        self,
        concept: torch.Tensor,
        closest_prototype: torch.Tensor
    ) -> torch.Tensor:
        """
        차이점 강조 (A04 아이디어)
        
        Args:
            concept: [B, H]
            closest_prototype: [B, H]
        
        Returns:
            amplified_diff: [B, H]
        """
        # Concatenate concept and prototype
        combined = torch.cat([concept, closest_prototype], dim=-1)  # [B, 2H]
        
        # Amplify contrast
        amplified = self.contrast_amplifier(combined)  # [B, H]
        
        # Extract differences
        diff = self.difference_extractor(amplified)  # [B, H]
        
        return diff
    
    def compute_novelty_dimensions(
        self,
        diff_features: torch.Tensor
    ) -> torch.Tensor:
        """
        차원별 참신성 계산
        
        Args:
            diff_features: [B, H]
        
        Returns:
            dim_scores: [B, D] (D = novelty_dimensions)
        """
        dim_scores = []
        for scorer in self.dimension_scorers:
            score = scorer(diff_features)  # [B, 1]
            dim_scores.append(score)
        
        dim_scores = torch.cat(dim_scores, dim=-1)  # [B, D]
        
        return dim_scores
    
    def forward(
        self,
        input_concept: torch.Tensor,
        reference_concepts: torch.Tensor,
        return_metadata: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass
        
        Args:
            input_concept: [B, D] - 평가할 새로운 개념
            reference_concepts: [B, N, D] - 기존 개념들
            return_metadata: 메타데이터 반환 여부
        
        Returns:
            novelty_score: [B] - 전체 참신성 점수 (0~1)
            novelty_dimensions: [B, 3] - 차원별 참신성
            metadata: 메타데이터 (선택적)
        """
        B = input_concept.shape[0]
        
        # 1. Extract concept from input (M05)
        concept = self.extract_concept(input_concept)  # [B, H]
        
        # 2. Encode reference concepts as prototypes (M05)
        prototypes = self.encode_prototypes(reference_concepts)  # [B, N, H]
        
        # 3. Compute dimensional similarities (M07)
        similarities = self.compute_dimensional_similarity(concept, prototypes)
        
        # 4. Find closest prototype
        # Average similarity across dimensions
        avg_similarity = torch.stack(list(similarities.values()), dim=-1).mean(dim=-1)  # [B, N]
        closest_idx = torch.argmax(avg_similarity, dim=1)  # [B]
        
        # Get closest prototype
        batch_indices = torch.arange(B, device=input_concept.device)
        closest_prototype = prototypes[batch_indices, closest_idx]  # [B, H]
        
        # 5. Amplify differences (A04)
        diff_features = self.amplify_differences(concept, closest_prototype)  # [B, H]
        
        # 6. Compute dimensional novelty scores
        dim_novelty = self.compute_novelty_dimensions(diff_features)  # [B, D]
        
        # 7. Compute overall novelty score
        # Combine diff_features and dim_novelty
        combined_features = torch.cat([diff_features, dim_novelty], dim=-1)  # [B, H+D]
        novelty_score = self.novelty_scorer(combined_features).squeeze(-1)  # [B]
        
        # 8. Generate explanation features
        explanation_features = self.explanation_generator(diff_features)  # [B, H]
        
        if return_metadata:
            metadata = {
                'concept_embedding': concept,
                'prototypes': prototypes,
                'similarities': similarities,
                'closest_prototype_idx': closest_idx,
                'closest_prototype': closest_prototype,
                'difference_features': diff_features,
                'dimensional_novelty': dim_novelty,
                'explanation_features': explanation_features,
                'novelty_threshold': self.novelty_threshold.item(),
                'is_novel': (novelty_score > self.novelty_threshold).float()
            }
            return novelty_score, dim_novelty, metadata
        
        return novelty_score, dim_novelty, None
    
    def get_config(self) -> Dict:
        """설정 반환"""
        return {
            'seed_id': self.config.seed_id,
            'name': self.config.name,
            'level': self.config.level,
            'category': self.config.category,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_reference_concepts': self.num_reference_concepts,
            'novelty_dimensions': self.novelty_dimensions,
            'params': self.count_parameters()
        }


# Alias for convenience
C08 = NoveltyAssessor
