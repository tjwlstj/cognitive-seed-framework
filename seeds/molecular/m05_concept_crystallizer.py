"""
SEED-M05: Concept Crystallizer

Few-shot learning을 통해 개념의 프로토타입을 학습하고 새로운 인스턴스를 분류합니다.

Category: Abstraction
Composed From: A05 (Grouping Nucleus) + M03 (Pattern Completer) + M01 (Hierarchy Builder)
Target Params: ~700K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from seeds.base import BaseSeed, SeedConfig


@dataclass
class ConceptCrystallizerConfig(SeedConfig):
    """Concept Crystallizer 설정"""
    seed_id: str = "SEED-M05"
    name: str = "Concept Crystallizer"
    level: int = 1
    category: str = "Abstraction"
    bit_depth: str = "FP8"
    params: int = 700000
    input_dim: int = 64
    output_dim: int = 320  # hidden_dim과 동일
    hidden_dim: int = 320
    n_way: int = 5
    k_shot: int = 5
    distance_metric: str = 'euclidean'
    dropout: float = 0.1


class ConceptCrystallizer(BaseSeed):
    """
    SEED-M05: Concept Crystallizer
    
    Few-shot learning을 통해 개념의 프로토타입을 학습하고
    새로운 인스턴스를 분류합니다.
    
    주요 기능:
    - Prototypical Networks 기반 few-shot learning
    - 계층적 개념 표현 학습
    - 거리 기반 분류
    - Meta-learning 지원
    """
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 320,
        n_way: int = 5,
        k_shot: int = 5,
        distance_metric: str = 'euclidean',
        dropout: float = 0.1
    ):
        config = ConceptCrystallizerConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_way=n_way,
            k_shot=k_shot,
            distance_metric=distance_metric,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_way = n_way
        self.k_shot = k_shot
        self.distance_metric = distance_metric
        
        # Lightweight embedding network
        # 구성 시드를 직접 사용하지 않고 그 아이디어만 차용
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Grouping-inspired layer (A05 아이디어)
        self.grouping_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Pattern completion layer (M03 아이디어)
        self.pattern_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Hierarchy layer (M01 아이디어)
        self.hierarchy_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Prototype refinement layer
        self.prototype_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Distance scaling parameter (learnable)
        self.distance_scale = nn.Parameter(torch.ones(1))
        
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력을 임베딩 공간으로 변환
        
        Args:
            x: [B, D] 또는 [B, L, D]
        
        Returns:
            embeddings: [B, D'] 또는 [B, L, D']
        """
        original_shape = x.shape
        
        # 2D로 변환
        if len(original_shape) == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
        
        B, L, D = x.shape
        
        # Embedding
        x_flat = x.view(B * L, D)
        emb = self.embedding_net(x_flat)
        emb = emb.view(B, L, -1)
        
        # Apply lightweight layers inspired by atomic/molecular seeds
        emb = self.grouping_layer(emb)
        emb = self.pattern_layer(emb)
        emb = self.hierarchy_layer(emb)
        
        # 원래 shape으로 복원
        if len(original_shape) == 2:
            emb = emb.squeeze(1)
        
        return emb
    
    def compute_prototypes(
        self, 
        support_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Support set으로부터 프로토타입 계산
        
        Args:
            support_embeddings: [N, K, D]
        
        Returns:
            prototypes: [N, D]
        """
        # 각 클래스의 평균 임베딩을 프로토타입으로 사용
        prototypes = support_embeddings.mean(dim=1)
        
        # Prototype refinement
        prototypes = self.prototype_refiner(prototypes)
        
        return prototypes
    
    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        쿼리와 프로토타입 간 거리 계산
        
        Args:
            query_embeddings: [Q, D]
            prototypes: [N, D]
        
        Returns:
            distances: [Q, N]
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance
            # [Q, 1, D] - [1, N, D] -> [Q, N, D]
            diff = query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
            distances = torch.norm(diff, dim=2)
        elif self.distance_metric == 'cosine':
            # Cosine similarity (음수로 변환하여 거리처럼 사용)
            query_norm = F.normalize(query_embeddings, dim=1)
            proto_norm = F.normalize(prototypes, dim=1)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1.0 - similarities  # [0, 2] 범위의 거리
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Learnable scaling
        distances = distances * self.distance_scale
        
        return distances
    
    def forward(
        self,
        support_set: torch.Tensor,
        query_set: torch.Tensor,
        return_metadata: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass
        
        Args:
            support_set: [N, K, D] - Support set
            query_set: [Q, D] - Query set
            return_metadata: 메타데이터 반환 여부
        
        Returns:
            logits: [Q, N] - 클래스별 로짓
            metadata: 메타데이터 (선택적)
        """
        N, K, D = support_set.shape
        Q = query_set.shape[0]
        
        # 1. Support set 임베딩
        # [N, K, D] -> [N*K, D]
        support_flat = support_set.view(N * K, D)
        support_emb = self.embed(support_flat)
        
        # [N*K, D'] -> [N, K, D']
        support_embeddings = support_emb.view(N, K, -1)
        
        # 2. Query set 임베딩
        query_embeddings = self.embed(query_set)
        
        # 3. 프로토타입 계산
        prototypes = self.compute_prototypes(support_embeddings)
        
        # 4. 거리 계산
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # 5. 로짓 계산 (거리의 음수를 로짓으로 사용)
        logits = -distances
        
        # 6. 예측
        predictions = torch.argmax(logits, dim=1)
        
        if return_metadata:
            metadata = {
                'prototypes': prototypes,
                'distances': distances,
                'support_embeddings': support_embeddings,
                'query_embeddings': query_embeddings,
                'predictions': predictions,
                'distance_scale': self.distance_scale.item()
            }
            return logits, metadata
        
        return logits, None
    
    def get_config(self) -> Dict:
        """설정 반환"""
        return {
            'seed_id': self.config.seed_id,
            'name': self.config.name,
            'level': self.config.level,
            'category': self.config.category,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'n_way': self.n_way,
            'k_shot': self.k_shot,
            'distance_metric': self.distance_metric,
            'params': self.count_parameters()
        }


# Alias for convenience
M05 = ConceptCrystallizer
