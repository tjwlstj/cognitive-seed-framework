"""
SEED-C03: Schema Learner

스키마 구조 학습 및 추상화를 수행하는 Cellular 레벨 시드입니다.

Category: Abstraction
Composed From: M01 (Hierarchy Builder) + M05 (Concept Crystallizer) + A05 (Grouping Nucleus)
Target Params: ~1.5M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from seeds.base import BaseSeed, SeedConfig


@dataclass
class SchemaLearnerConfig(SeedConfig):
    """Schema Learner 설정"""
    seed_id: str = "SEED-C03"
    name: str = "Schema Learner"
    level: int = 2
    category: str = "Abstraction"
    bit_depth: str = "FP8"
    params: int = 1500000
    input_dim: int = 128
    output_dim: int = 128
    hidden_dim: int = 200
    num_schema_slots: int = 8
    num_levels: int = 4
    dropout: float = 0.1


class SchemaLearner(BaseSeed):
    """
    SEED-C03: Schema Learner
    
    스키마 구조 학습 및 추상화를 수행합니다.
    
    주요 기능:
    - 패턴 인식 및 그룹화 (A05 기반)
    - 개념 프로토타입 추출 (M05 기반)
    - 계층적 스키마 구조 학습 (M01 기반)
    - 추상화된 스키마 표현 생성
    - 구조적 규칙 추출
    
    입력:
    - patterns: 패턴 텐서 [B, N, D]
    - context: 맥락 정보 [B, C, D] (선택)
    
    출력:
    - schema: 스키마 표현 [B, D]
    - hierarchy: 계층 구조 [B, num_levels, hidden_dim]
    - concepts: 추출된 개념 [B, num_schema_slots, hidden_dim]
    - rules: 구조적 규칙 [B, num_schema_slots, num_schema_slots]
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 200,
        num_schema_slots: int = 8,
        num_levels: int = 4,
        dropout: float = 0.1
    ):
        config = SchemaLearnerConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_schema_slots=num_schema_slots,
            num_levels=num_levels,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_schema_slots = num_schema_slots
        self.num_levels = num_levels
        
        # 1. Pattern Encoder
        self.pattern_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. Grouping Module (A05 아이디어: 유사 패턴 그룹화)
        self.grouping_module = self._build_grouping_module()
        
        # 3. Concept Crystallizer (M05 아이디어: 프로토타입 학습)
        self.concept_crystallizer = self._build_concept_crystallizer()
        
        # 4. Hierarchy Builder (M01 아이디어: 계층 구조 학습)
        self.hierarchy_builder = self._build_hierarchy_builder()
        
        # 5. Schema Slot Attention
        self.schema_slots = nn.Parameter(torch.randn(num_schema_slots, hidden_dim))
        self.slot_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 6. Schema Generator
        self.schema_generator = nn.Sequential(
            nn.Linear(hidden_dim * num_schema_slots, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 7. Rule Extractor (스키마 슬롯 간 관계)
        self.rule_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 8. Level Encoder (계층 레벨 인코딩)
        self.level_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )
    
    def _build_grouping_module(self) -> nn.Module:
        """그룹화 모듈 구축 (A05 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.num_schema_slots),
            nn.Softmax(dim=-1)
        )
    
    def _build_concept_crystallizer(self) -> nn.Module:
        """개념 결정화 모듈 구축 (M05 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _build_hierarchy_builder(self) -> nn.Module:
        """계층 구축 모듈 (M01 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward(
        self,
        patterns: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_metadata: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass
        
        Args:
            patterns: [B, N, D] - 입력 패턴
            context: [B, C, D] - 맥락 정보 (선택)
            return_metadata: 메타데이터 반환 여부
        
        Returns:
            schema: [B, D] - 스키마 표현
            metadata: 메타데이터 (선택적)
        """
        B, N, D = patterns.shape
        
        # 1. Pattern Encoding
        pattern_features = self.pattern_encoder(patterns)  # [B, N, hidden_dim]
        
        # 2. Pattern Grouping (A05 아이디어)
        group_assignments = self.grouping_module(pattern_features)  # [B, N, num_schema_slots]
        
        # 3. Concept Extraction (M05 아이디어)
        concepts = self._extract_concepts(pattern_features, group_assignments)  # [B, num_schema_slots, hidden_dim]
        
        # 4. Hierarchy Construction (M01 아이디어)
        hierarchy = self._build_hierarchy(concepts)  # [B, num_levels, hidden_dim]
        
        # 5. Schema Slot Refinement
        refined_concepts = self._refine_concepts(concepts, hierarchy)  # [B, num_schema_slots, hidden_dim]
        
        # 6. Rule Extraction
        rules = self._extract_rules(refined_concepts)  # [B, num_schema_slots, num_schema_slots]
        
        # 7. Schema Generation
        schema = self._generate_schema(refined_concepts)  # [B, D]
        
        if return_metadata:
            metadata = {
                'concepts': refined_concepts,
                'hierarchy': hierarchy,
                'rules': rules,
                'group_assignments': group_assignments,
                'pattern_features': pattern_features
            }
            return schema, metadata
        
        return schema, None
    
    def _extract_concepts(
        self,
        pattern_features: torch.Tensor,
        group_assignments: torch.Tensor
    ) -> torch.Tensor:
        """
        그룹별 개념 프로토타입 추출 (M05 아이디어)
        
        Args:
            pattern_features: [B, N, hidden_dim]
            group_assignments: [B, N, num_schema_slots]
        
        Returns:
            concepts: [B, num_schema_slots, hidden_dim]
        """
        B, N, hidden_dim = pattern_features.shape
        
        # Weighted average by group assignments
        # [B, num_schema_slots, N] @ [B, N, hidden_dim] -> [B, num_schema_slots, hidden_dim]
        concepts = torch.bmm(
            group_assignments.transpose(1, 2),
            pattern_features
        )
        
        # Normalize by assignment weights
        assignment_sums = group_assignments.sum(dim=1, keepdim=True).transpose(1, 2)  # [B, num_schema_slots, 1]
        concepts = concepts / (assignment_sums + 1e-8)
        
        # Concept refinement
        concepts = self.concept_crystallizer(concepts)
        
        return concepts
    
    def _build_hierarchy(
        self,
        concepts: torch.Tensor
    ) -> torch.Tensor:
        """
        계층 구조 구축 (M01 아이디어)
        
        Args:
            concepts: [B, num_schema_slots, hidden_dim]
        
        Returns:
            hierarchy: [B, num_levels, hidden_dim]
        """
        B, num_slots, hidden_dim = concepts.shape
        
        hierarchy = []
        current_level = concepts  # [B, num_slots, hidden_dim]
        
        for level in range(self.num_levels):
            # Level encoding
            level_value = torch.tensor([level / self.num_levels], device=concepts.device)
            level_emb = self.level_encoder(level_value.unsqueeze(0))  # [1, hidden_dim]
            level_emb = level_emb.expand(B, 1, -1)  # [B, 1, hidden_dim]
            
            # Aggregate current level
            level_summary = current_level.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
            
            # Combine with level encoding
            combined = torch.cat([level_summary, level_emb], dim=-1)  # [B, 1, hidden_dim * 2]
            level_features = self.hierarchy_builder(combined)  # [B, 1, hidden_dim]
            
            hierarchy.append(level_features)
            
            # Prepare next level (abstraction)
            if level < self.num_levels - 1:
                # Pool to half size for next level
                num_next = max(1, num_slots // 2)
                if num_next < current_level.shape[1]:
                    # Simple pooling
                    current_level = F.adaptive_avg_pool1d(
                        current_level.transpose(1, 2),
                        num_next
                    ).transpose(1, 2)
                num_slots = num_next
        
        # Stack hierarchy levels
        hierarchy = torch.cat(hierarchy, dim=1)  # [B, num_levels, hidden_dim]
        
        return hierarchy
    
    def _refine_concepts(
        self,
        concepts: torch.Tensor,
        hierarchy: torch.Tensor
    ) -> torch.Tensor:
        """
        계층 정보를 활용한 개념 정제
        
        Args:
            concepts: [B, num_schema_slots, hidden_dim]
            hierarchy: [B, num_levels, hidden_dim]
        
        Returns:
            refined_concepts: [B, num_schema_slots, hidden_dim]
        """
        B = concepts.shape[0]
        
        # Expand schema slots for batch
        slots = self.schema_slots.unsqueeze(0).expand(B, -1, -1)  # [B, num_schema_slots, hidden_dim]
        
        # Combine concepts and hierarchy for attention
        context = torch.cat([concepts, hierarchy], dim=1)  # [B, num_schema_slots + num_levels, hidden_dim]
        
        # Slot attention
        refined_concepts, _ = self.slot_attention(
            query=slots,
            key=context,
            value=context
        )  # [B, num_schema_slots, hidden_dim]
        
        # Residual connection
        refined_concepts = refined_concepts + concepts
        
        return refined_concepts
    
    def _extract_rules(
        self,
        concepts: torch.Tensor
    ) -> torch.Tensor:
        """
        개념 간 구조적 규칙 추출
        
        Args:
            concepts: [B, num_schema_slots, hidden_dim]
        
        Returns:
            rules: [B, num_schema_slots, num_schema_slots]
        """
        B, num_slots, hidden_dim = concepts.shape
        
        # Create all pairs
        # [B, num_slots, 1, hidden_dim] and [B, 1, num_slots, hidden_dim]
        concepts_i = concepts.unsqueeze(2).expand(B, num_slots, num_slots, hidden_dim)
        concepts_j = concepts.unsqueeze(1).expand(B, num_slots, num_slots, hidden_dim)
        
        # Concatenate pairs
        pairs = torch.cat([concepts_i, concepts_j], dim=-1)  # [B, num_slots, num_slots, hidden_dim * 2]
        
        # Extract rules (relationship strength)
        rules = self.rule_extractor(pairs).squeeze(-1)  # [B, num_slots, num_slots]
        
        # Make diagonal zero (no self-relation)
        mask = torch.eye(num_slots, device=concepts.device).unsqueeze(0).expand(B, -1, -1)
        rules = rules * (1 - mask)
        
        return rules
    
    def _generate_schema(
        self,
        concepts: torch.Tensor
    ) -> torch.Tensor:
        """
        최종 스키마 표현 생성
        
        Args:
            concepts: [B, num_schema_slots, hidden_dim]
        
        Returns:
            schema: [B, D]
        """
        B = concepts.shape[0]
        
        # Flatten concepts
        concepts_flat = concepts.reshape(B, -1)  # [B, num_schema_slots * hidden_dim]
        
        # Generate schema
        schema = self.schema_generator(concepts_flat)  # [B, D]
        
        return schema
    
    def get_config(self) -> Dict:
        """설정 반환"""
        return {
            'seed_id': self.config.seed_id,
            'name': self.config.name,
            'level': self.config.level,
            'category': self.config.category,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_schema_slots': self.num_schema_slots,
            'num_levels': self.num_levels,
            'params': self.count_parameters()
        }
    
    def visualize_schema(
        self,
        patterns: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        스키마 시각화를 위한 정보 추출
        
        Args:
            patterns: [B, N, D]
            context: [B, C, D] (선택)
        
        Returns:
            visualization_data: 시각화 데이터
        """
        schema, metadata = self.forward(patterns, context, return_metadata=True)
        
        return {
            'schema': schema,
            'concepts': metadata['concepts'],
            'hierarchy': metadata['hierarchy'],
            'rules': metadata['rules'],
            'group_assignments': metadata['group_assignments']
        }


# Alias for convenience
C03 = SchemaLearner


def create_schema_learner(
    input_dim: int = 128,
    hidden_dim: int = 200,
    num_schema_slots: int = 8,
    num_levels: int = 4,
    dropout: float = 0.1
) -> SchemaLearner:
    """Schema Learner 시드 생성 함수"""
    return SchemaLearner(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_schema_slots=num_schema_slots,
        num_levels=num_levels,
        dropout=dropout
    )
