"""
SEED-M01 — Hierarchy Builder

상하/포함 관계를 파악하여 트리 또는 DAG 구조를 구축하는 분자 시드입니다.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig
from ..atomic.a05_grouping_nucleus import GroupingNucleus
from ..atomic.a08_binary_comparator import BinaryComparator
from ..atomic.a07_scale_normalizer import ScaleNormalizer


class HierarchyBuilder(BaseSeed):
    """
    SEED-M01: Hierarchy Builder
    
    Category: Relation
    Bit: INT8/FP8
    Params: ~500K
    Purpose: 상하 관계를 파악하여 계층 구조 구축
    I/O: [B,N,D] → [B,N,D]
    Composed From: A05 + A08 + A07
    """
    
    def __init__(self, input_dim: int = 128, num_clusters: int = 16):
        config = SeedConfig(
            seed_id="SEED-M01",
            name="Hierarchy Builder",
            level=1,
            category="Relation",
            bit_depth="FP8",
            params=500000,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E", "H"]  # Euclidean + Hyperbolic (계층 구조)
        )
        super().__init__(config)
        
        # Atomic seeds
        self.grouping = GroupingNucleus(input_dim, num_clusters=num_clusters)  # A05
        self.comparator = BinaryComparator(input_dim)                          # A08
        self.normalizer = ScaleNormalizer(input_dim)                           # A07
        
        # Hierarchy construction network
        self.hierarchy_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim)
        )
        
        # Parent-child relationship predictor
        self.relation_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 부모-자식 관계 확률
        )
        
        # Level encoder (각 노드의 계층 레벨 인코딩)
        self.level_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] - N개 노드의 특징
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보
        Returns:
            hierarchy: [B, N, D] - 계층 정보가 인코딩된 특징
        """
        batch_size, num_nodes, dim = x.shape
        
        # 1. 스케일 정규화 (A07)
        x_norm = self.normalizer(x, scale)
        
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x_norm = self.cse(x_norm, scale)
        
        # MGP: 기하학적 투영 (Hyperbolic space for hierarchy)
        if self.config.use_mgp:
            x_proj = self.mgp(x_norm)
        else:
            x_proj = x_norm
        
        # 2. 초기 그룹화 (클러스터링) (A05)
        cluster_features = self.grouping(x_proj)
        
        # 3. 쌍별 비교로 상하 관계 파악
        hierarchy_matrix = self.build_hierarchy_matrix(x_proj)
        
        # 4. 계층 정보 인코딩
        hierarchy_features = self.encode_hierarchy(x_proj, hierarchy_matrix)
        
        # 5. 클러스터 정보와 계층 정보 결합
        output = hierarchy_features + 0.3 * cluster_features
        
        return output
    
    def build_hierarchy_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        N x N 계층 관계 행렬 구축
        matrix[i, j] = P(i is parent of j)
        
        Args:
            x: [B, N, D]
        Returns:
            hierarchy_matrix: [B, N, N]
        """
        B, N, D = x.shape
        hierarchy_matrix = torch.zeros(B, N, N, device=x.device)
        
        # 효율적인 배치 처리
        # x_i: [B, N, 1, D], x_j: [B, 1, N, D]
        x_i = x.unsqueeze(2).expand(B, N, N, D)
        x_j = x.unsqueeze(1).expand(B, N, N, D)
        
        # 모든 쌍 결합: [B, N, N, 2D]
        pairs = torch.cat([x_i, x_j], dim=-1)
        
        # 관계 예측: [B, N, N, 1]
        relation_probs = self.relation_predictor(pairs)
        hierarchy_matrix = relation_probs.squeeze(-1)
        
        # 대각선은 0 (자기 자신과의 관계 없음)
        mask = torch.eye(N, device=x.device).unsqueeze(0).expand(B, N, N)
        hierarchy_matrix = hierarchy_matrix * (1 - mask)
        
        return hierarchy_matrix
    
    def encode_hierarchy(self, x: torch.Tensor, hierarchy_matrix: torch.Tensor) -> torch.Tensor:
        """
        계층 관계 행렬을 사용하여 각 노드에 계층 정보 인코딩
        
        Args:
            x: [B, N, D]
            hierarchy_matrix: [B, N, N]
        Returns:
            hierarchy_features: [B, N, D]
        """
        B, N, D = x.shape
        
        # 각 노드의 부모 노드들의 가중 평균 계산
        # hierarchy_matrix[b, i, j]: i가 j의 부모일 확률
        # 따라서 j의 부모들을 찾으려면 [:, :, j]를 봐야 함
        
        parent_weights = hierarchy_matrix.transpose(1, 2)  # [B, N, N] - parent_weights[b, j, i]: i가 j의 부모일 확률
        parent_weights = F.normalize(parent_weights, p=1, dim=-1)  # 정규화
        
        # 부모 정보 집계
        parent_info = torch.bmm(parent_weights, x)  # [B, N, D]
        
        # 자식 정보 집계
        child_weights = F.normalize(hierarchy_matrix, p=1, dim=-1)
        child_info = torch.bmm(child_weights, x)  # [B, N, D]
        
        # 원본 특징과 부모/자식 정보 결합
        combined = torch.cat([x, parent_info], dim=-1)  # [B, N, 2D]
        hierarchy_features = self.hierarchy_encoder(combined)  # [B, N, D]
        
        # 레벨 정보 추가
        levels = self.compute_levels(hierarchy_matrix)  # [B, N]
        level_features = self.level_encoder(levels.unsqueeze(-1))  # [B, N, D]
        
        hierarchy_features = hierarchy_features + 0.2 * level_features
        
        return hierarchy_features
    
    def compute_levels(self, hierarchy_matrix: torch.Tensor, max_depth: int = 10) -> torch.Tensor:
        """
        각 노드의 계층 레벨 계산 (루트로부터의 거리)
        
        Args:
            hierarchy_matrix: [B, N, N]
            max_depth: 최대 깊이
        Returns:
            levels: [B, N] - 각 노드의 레벨 (0 = root)
        """
        B, N, _ = hierarchy_matrix.shape
        
        # 인접 행렬 (임계값 적용)
        adjacency = (hierarchy_matrix > 0.5).float()
        
        # 각 노드의 부모 수
        num_parents = adjacency.sum(dim=1)  # [B, N]
        
        # 루트 노드 (부모가 없는 노드)
        levels = torch.zeros(B, N, device=hierarchy_matrix.device)
        is_root = (num_parents == 0).float()
        
        # BFS-like level assignment
        current_level = is_root
        for depth in range(1, max_depth):
            # 현재 레벨 노드들의 자식 찾기
            # adjacency[b, i, j] = 1 if i is parent of j
            children = torch.bmm(current_level.unsqueeze(1), adjacency).squeeze(1)  # [B, N]
            children = (children > 0).float()
            
            # 아직 레벨이 할당되지 않은 노드만 업데이트
            new_nodes = children * (levels == 0).float() * (1 - is_root)
            levels = levels + new_nodes * depth
            
            if new_nodes.sum() == 0:
                break
            
            current_level = new_nodes
        
        return levels
    
    def get_tree_structure(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        트리 구조 추출
        
        Args:
            x: [B, N, D]
        Returns:
            {
                'adjacency_matrix': [B, N, N],
                'levels': [B, N],
                'hierarchy_matrix': [B, N, N],
                'num_roots': [B],
                'max_depth': [B]
            }
        """
        # 계층 관계 행렬 구축
        x_norm = self.normalizer(x)
        if self.config.use_mgp:
            x_proj = self.mgp(x_norm)
        else:
            x_proj = x_norm
        
        hierarchy_matrix = self.build_hierarchy_matrix(x_proj)
        
        # 인접 행렬 (임계값 적용)
        adjacency = (hierarchy_matrix > 0.5).float()
        
        # 레벨 계산
        levels = self.compute_levels(hierarchy_matrix)
        
        # 루트 노드 수
        num_parents = adjacency.sum(dim=1)
        num_roots = (num_parents == 0).sum(dim=1)
        
        # 최대 깊이
        max_depth = levels.max(dim=1)[0]
        
        return {
            'adjacency_matrix': adjacency,
            'levels': levels,
            'hierarchy_matrix': hierarchy_matrix,
            'num_roots': num_roots,
            'max_depth': max_depth
        }


def create_hierarchy_builder(input_dim: int = 128, num_clusters: int = 16) -> HierarchyBuilder:
    """Hierarchy Builder 시드 생성 함수"""
    return HierarchyBuilder(input_dim=input_dim, num_clusters=num_clusters)

