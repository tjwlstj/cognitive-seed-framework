"""
Base Seed Classes

모든 인지 시드의 기본 클래스와 공통 컴포넌트를 정의합니다.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SeedConfig:
    """시드 설정 클래스"""
    seed_id: str
    name: str
    level: int  # 0=Atomic, 1=Molecular, 2=Cellular, 3=Tissue
    category: str  # Pattern|Relation|Temporal|Spatial|Logic|Analogy|Abstraction|Composition
    bit_depth: str  # INT8|FP8|FP16|BF16
    params: int
    input_dim: int
    output_dim: int
    use_mgp: bool = True  # Multi-Geometry Projection 사용 여부
    use_cse: bool = True  # Continuous Scale-Equivariant 사용 여부
    geometries: List[str] = None  # ["E", "H", "S"]
    
    def __post_init__(self):
        if self.geometries is None:
            self.geometries = ["E", "H", "S"]


class MGPBlock(nn.Module):
    """
    Multi-Geometry Projection Block
    
    입력을 Euclidean(E), Hyperbolic(H), Spherical(S) 3개의 기하학적 공간으로 투영하고
    게이트 네트워크를 통해 가중 결합합니다.
    """
    
    def __init__(self, dim: int, geometries: List[str] = None):
        super().__init__()
        self.dim = dim
        self.geometries = geometries or ["E", "H", "S"]
        self.num_geometries = len(self.geometries)
        
        # 각 기하학 공간으로의 투영
        if "E" in self.geometries:
            self.proj_E = nn.Linear(dim, dim)
        if "H" in self.geometries:
            self.proj_H = nn.Linear(dim, dim)
        if "S" in self.geometries:
            self.proj_S = nn.Linear(dim, dim)
        
        # 게이트 네트워크 (어떤 기하학을 얼마나 사용할지 결정)
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, self.num_geometries),
            nn.Softmax(dim=-1)
        )
    
    def euclidean_projection(self, x: torch.Tensor) -> torch.Tensor:
        """유클리드 공간 투영"""
        return self.proj_E(x)
    
    def hyperbolic_projection(self, x: torch.Tensor) -> torch.Tensor:
        """쌍곡 공간 투영 (Poincaré ball model)"""
        x = self.proj_H(x)
        # 쌍곡 공간으로 매핑 (norm < 1로 제한)
        norm = torch.norm(x, dim=-1, keepdim=True)
        x = x / (norm + 1e-5) * torch.tanh(norm)
        return x
    
    def spherical_projection(self, x: torch.Tensor) -> torch.Tensor:
        """구면 공간 투영 (unit sphere)"""
        x = self.proj_S(x)
        # 구면으로 정규화
        return F.normalize(x, p=2, dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, *, D] 형태의 입력 텐서
        Returns:
            [B, *, D] 형태의 출력 텐서
        """
        projections = []
        
        if "E" in self.geometries:
            projections.append(self.euclidean_projection(x))
        if "H" in self.geometries:
            projections.append(self.hyperbolic_projection(x))
        if "S" in self.geometries:
            projections.append(self.spherical_projection(x))
        
        # 게이트 가중치 계산
        # x를 [B, D, *] 형태로 변환하여 AdaptiveAvgPool1d 적용
        original_shape = x.shape
        x_flat = x.view(x.size(0), x.size(-1), -1)  # [B, D, *]
        gates = self.gate_net(x_flat)  # [B, num_geometries]
        
        # 가중 결합
        output = torch.zeros_like(projections[0])
        for i, proj in enumerate(projections):
            gate_weight = gates[:, i].view(-1, *([1] * (len(original_shape) - 1)))
            output = output + gate_weight * proj
        
        return output


class CSEBlock(nn.Module):
    """
    Continuous Scale-Equivariant Block
    
    연속 스케일 매개변수를 받아 조건부 정규화를 수행합니다.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # 스케일 인코더
        self.scale_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, dim * 2)  # gamma와 beta
        )
        
        # Layer Normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, *, D] 형태의 입력 텐서
            scale: [B, 1] 형태의 스케일 매개변수 (None이면 1.0 사용)
        Returns:
            [B, *, D] 형태의 출력 텐서
        """
        # 정규화
        x = self.norm(x)
        
        # 스케일이 주어지지 않으면 기본값 사용
        if scale is None:
            return x
        
        # 스케일 조건부 변조
        scale_params = self.scale_encoder(scale)  # [B, D*2]
        gamma, beta = torch.chunk(scale_params, 2, dim=-1)  # [B, D], [B, D]
        
        # 브로드캐스팅을 위한 shape 조정
        original_shape = x.shape
        gamma = gamma.view(x.size(0), *([1] * (len(original_shape) - 2)), -1)
        beta = beta.view(x.size(0), *([1] * (len(original_shape) - 2)), -1)
        
        return gamma * x + beta


class BaseSeed(nn.Module):
    """
    모든 인지 시드의 기본 클래스
    
    MGP와 CSE를 통합하여 기하학적 적합성과 스케일 강건성을 제공합니다.
    """
    
    def __init__(self, config: SeedConfig):
        super().__init__()
        self.config = config
        
        # MGP 블록
        if config.use_mgp:
            self.mgp = MGPBlock(config.input_dim, config.geometries)
        
        # CSE 블록
        if config.use_cse:
            self.cse = CSEBlock(config.input_dim)
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None, 
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, *, D] 형태의 입력 텐서
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보 (attention_mask, position_encoding 등)
        Returns:
            [B, *, D] 형태의 출력 텐서
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_metadata(self) -> Dict[str, Any]:
        """시드 메타데이터 반환"""
        return {
            "seed_id": self.config.seed_id,
            "name": self.config.name,
            "level": self.config.level,
            "category": self.config.category,
            "bit_depth": self.config.bit_depth,
            "params": self.config.params,
            "geometries": self.config.geometries,
        }
    
    def count_parameters(self) -> int:
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

