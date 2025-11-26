"""
SEED-C01: Metaphor Engine

은유적 매핑 및 개념 전이를 수행하는 Cellular 레벨 시드입니다.

Category: Analogy
Composed From: M01 (Hierarchy Builder) + M07 (Analogy Mapper) + M05 (Concept Crystallizer)
Target Params: ~750K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from seeds.base import BaseSeed, SeedConfig


@dataclass
class MetaphorEngineConfig(SeedConfig):
    """Metaphor Engine 설정"""
    seed_id: str = "SEED-C01"
    name: str = "Metaphor Engine"
    level: int = 2
    category: str = "Analogy"
    bit_depth: str = "FP8"
    params: int = 750000
    input_dim: int = 128
    output_dim: int = 128
    hidden_dim: int = 180
    num_heads: int = 8
    dropout: float = 0.1
    metaphor_threshold: float = 0.6


class MetaphorEngine(BaseSeed):
    """
    SEED-C01: Metaphor Engine
    
    은유적 매핑 및 개념 전이를 수행합니다.
    
    주요 기능:
    - 소스 도메인 개념 추출 (M05 기반)
    - 타겟 도메인 개념 추출 (M05 기반)
    - 구조적 유사성 매핑 (M07 기반)
    - 계층적 관계 보존 (M01 기반)
    - 은유 표현 생성
    
    입력:
    - source: 소스 도메인 표현 [B, S, D]
    - target: 타겟 도메인 표현 [B, T, D]
    
    출력:
    - metaphor: 은유 표현 [B, D]
    - mapping_score: 매핑 품질 점수 [B]
    - structural_similarity: 구조적 유사도 [B]
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 180,
        num_heads: int = 8,
        dropout: float = 0.1,
        metaphor_threshold: float = 0.6
    ):
        config = MetaphorEngineConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            metaphor_threshold=metaphor_threshold
        )
        super().__init__(config)
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.metaphor_threshold = metaphor_threshold
        
        # 1. Source Domain Encoder (M05 아이디어: 개념 추출)
        self.source_encoder = self._build_domain_encoder()
        
        # 2. Target Domain Encoder (M05 아이디어: 개념 추출)
        self.target_encoder = self._build_domain_encoder()
        
        # 3. Structural Mapper (M07 아이디어: 구조적 유사성 매핑)
        self.structural_mapper = self._build_structural_mapper()
        
        # 4. Hierarchy Analyzer (M01 아이디어: 계층적 관계 분석)
        self.hierarchy_analyzer = self._build_hierarchy_analyzer()
        
        # 5. Cross-Domain Attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 6. Metaphor Generator
        self.metaphor_generator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 7. Quality Estimator
        self.quality_estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 8. Structural Similarity Estimator
        self.similarity_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _build_domain_encoder(self) -> nn.Module:
        """도메인 인코더 구축 (M05 개념 추출 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _build_structural_mapper(self) -> nn.Module:
        """구조적 매퍼 구축 (M07 구조 매핑 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_hierarchy_analyzer(self) -> nn.Module:
        """계층 분석기 구축 (M01 계층 구조 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        context: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            source: 소스 도메인 표현 [B, S, D]
            target: 타겟 도메인 표현 [B, T, D]
            scale: 스케일 매개변수 [B, 1]
            context: 추가 맥락 정보
        
        Returns:
            metaphor: 은유 표현 [B, D]
            mapping_score: 매핑 품질 점수 [B]
            structural_similarity: 구조적 유사도 [B]
        """
        batch_size = source.size(0)
        
        # 1. 소스 도메인 인코딩 (M05 개념 추출)
        source_encoded = self.source_encoder(source)  # [B, S, H]
        source_concept = source_encoded.mean(dim=1)   # [B, H] - 프로토타입
        
        # 2. 타겟 도메인 인코딩 (M05 개념 추출)
        target_encoded = self.target_encoder(target)  # [B, T, H]
        target_concept = target_encoded.mean(dim=1)   # [B, H] - 프로토타입
        
        # 3. 구조적 매핑 (M07 구조 매핑)
        # 소스와 타겟의 구조적 대응 관계 학습
        cross_features = torch.cat([source_concept, target_concept], dim=-1)  # [B, 2H]
        structural_mapping = self.structural_mapper(cross_features)  # [B, H]
        
        # 4. 계층적 관계 분석 (M01 계층 구조)
        # 소스와 타겟의 계층적 구조 보존
        hierarchy_source = self.hierarchy_analyzer(source_concept)  # [B, H]
        hierarchy_target = self.hierarchy_analyzer(target_concept)  # [B, H]
        
        # 5. Cross-Domain Attention
        # 소스와 타겟 간의 주의 메커니즘
        attended_features, attention_weights = self.cross_attention(
            query=source_encoded,
            key=target_encoded,
            value=target_encoded
        )  # [B, S, H]
        attended_concept = attended_features.mean(dim=1)  # [B, H]
        
        # 6. 은유 생성
        # 구조적 매핑, 계층 정보, 주의 특징 결합
        combined_features = torch.cat([
            structural_mapping,
            hierarchy_source,
            attended_concept
        ], dim=-1)  # [B, 3H]
        
        metaphor = self.metaphor_generator(combined_features)  # [B, D]
        
        # 7. 매핑 품질 평가
        mapping_score = self.quality_estimator(metaphor).squeeze(-1)  # [B]
        
        # 8. 구조적 유사도 평가
        similarity_features = torch.cat([
            hierarchy_source,
            hierarchy_target
        ], dim=-1)  # [B, 2H]
        structural_similarity = self.similarity_estimator(similarity_features).squeeze(-1)  # [B]
        
        return metaphor, mapping_score, structural_similarity
    
    def get_metadata(self) -> Dict:
        """시드 메타데이터 반환"""
        return {
            'seed_id': self.config.seed_id,
            'name': self.config.name,
            'level': self.config.level,
            'category': self.config.category,
            'composed_from': ['M01', 'M07', 'M05'],
            'input_shape': f'[B, S/T, {self.input_dim}]',
            'output_shape': f'[B, {self.input_dim}]',
            'parameters': sum(p.numel() for p in self.parameters()),
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'metaphor_threshold': self.metaphor_threshold
        }
    
    def compute_metaphor_quality(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        metaphor: torch.Tensor
    ) -> torch.Tensor:
        """
        은유 품질 계산
        
        Args:
            source: 소스 도메인 [B, S, D]
            target: 타겟 도메인 [B, T, D]
            metaphor: 생성된 은유 [B, D]
        
        Returns:
            quality: 품질 점수 [B]
        """
        # 소스와 타겟의 중심 계산
        source_center = source.mean(dim=1)  # [B, D]
        target_center = target.mean(dim=1)  # [B, D]
        
        # 은유가 소스와 타겟 사이에 위치하는지 평가
        source_distance = F.cosine_similarity(metaphor, source_center, dim=-1)
        target_distance = F.cosine_similarity(metaphor, target_center, dim=-1)
        
        # 균형 잡힌 은유일수록 높은 점수
        quality = (source_distance + target_distance) / 2.0
        
        return quality


# 별칭 지원
C01_MetaphorEngine = MetaphorEngine
