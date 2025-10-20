"""
SEED-A05 — Grouping Nucleus

유사도 기반 클러스터링을 수행하는 원자 시드입니다.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig


class GroupingNucleus(BaseSeed):
    """
    SEED-A05: Grouping Nucleus
    
    Category: Relation
    Bit: INT8
    Params: ~256
    Purpose: 유사도 기반 클러스터 시드
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, num_clusters: int = 8):
        config = SeedConfig(
            seed_id="SEED-A05",
            name="Grouping Nucleus",
            level=0,
            category="Relation",
            bit_depth="INT8",
            params=256,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E", "H"]  # 계층적 관계는 쌍곡 공간에서 유용
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        
        # 특징 임베딩
        self.feature_embedder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 클러스터 중심 학습
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, hidden_dim))
        
        # 클러스터 할당 네트워크
        self.assignment_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_clusters),
            nn.Softmax(dim=-1)
        )
        
        # 그룹 정보 인코더
        self.group_encoder = nn.Sequential(
            nn.Linear(hidden_dim + num_clusters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 형태의 입력 텐서
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보
        Returns:
            [B, L, D] 형태의 그룹 정보가 인코딩된 텐서
        """
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 특징 임베딩
        features = self.feature_embedder(x)  # [B, L, hidden_dim]
        
        # 클러스터 할당
        assignments = self.assignment_net(features)  # [B, L, num_clusters]
        
        # 특징과 할당 정보 결합
        combined = torch.cat([features, assignments], dim=-1)  # [B, L, hidden_dim + num_clusters]
        
        # 그룹 정보 인코딩
        group_info = self.group_encoder(combined)  # [B, L, D]
        
        # 원본과 그룹 정보 결합
        output = x + group_info
        
        return output
    
    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """
        클러스터 할당을 반환합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            [B, L, num_clusters] 형태의 클러스터 할당 확률
        """
        features = self.feature_embedder(x)
        assignments = self.assignment_net(features)
        return assignments
    
    def get_hard_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """
        하드 클러스터 할당을 반환합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            [B, L] 형태의 클러스터 인덱스
        """
        assignments = self.get_cluster_assignments(x)
        hard_assignments = torch.argmax(assignments, dim=-1)
        return hard_assignments
    
    def compute_cluster_distances(self, x: torch.Tensor) -> torch.Tensor:
        """
        각 샘플과 클러스터 중심 간의 거리를 계산합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
        Returns:
            [B, L, num_clusters] 형태의 거리 텐서
        """
        features = self.feature_embedder(x)  # [B, L, hidden_dim]
        
        # 유클리드 거리 계산
        # features: [B, L, hidden_dim], cluster_centers: [num_clusters, hidden_dim]
        distances = torch.cdist(
            features.view(-1, self.hidden_dim),
            self.cluster_centers
        )  # [B*L, num_clusters]
        
        distances = distances.view(x.size(0), x.size(1), self.num_clusters)
        
        return distances


def create_grouping_nucleus(input_dim: int = 128, hidden_dim: int = 128, 
                            num_clusters: int = 8) -> GroupingNucleus:
    """Grouping Nucleus 시드 생성 함수"""
    return GroupingNucleus(input_dim=input_dim, hidden_dim=hidden_dim, num_clusters=num_clusters)

