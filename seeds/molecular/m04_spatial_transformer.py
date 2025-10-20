"""
SEED-M04 — Spatial Transformer

회전, 스케일, 평행이동 등의 공간 변환을 수행하여 입력을 정렬하는 분자 시드입니다.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig
from ..atomic.a02_symmetry_detector import SymmetryDetector
from ..atomic.a07_scale_normalizer import ScaleNormalizer
from ..atomic.a01_edge_detector import EdgeDetector


class SpatialTransformer(BaseSeed):
    """
    SEED-M04: Spatial Transformer
    
    Category: Spatial
    Bit: FP8
    Params: ~450K
    Purpose: 공간 변환을 통해 입력을 정규 좌표계로 정렬
    I/O: [B,L,D] → [B,L,D]
    Composed From: A02 + A07 + A01
    """
    
    def __init__(self, input_dim: int = 128):
        config = SeedConfig(
            seed_id="SEED-M04",
            name="Spatial Transformer",
            level=1,
            category="Spatial",
            bit_depth="FP8",
            params=450000,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E", "S"]  # Euclidean + Spherical (회전 대칭)
        )
        super().__init__(config)
        
        # Atomic seeds
        self.symmetry_detector = SymmetryDetector(input_dim)  # A02
        self.scale_normalizer = ScaleNormalizer(input_dim)    # A07
        self.edge_detector = EdgeDetector(input_dim)          # A01
        
        # Transformation parameter predictor
        self.transform_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # [tx, ty, rotation, scale_x, scale_y, shear]
        )
        
        # Equivariant feature extractor
        self.equivariant_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
        # Transformation refinement network
        self.refinement_net = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] - 입력 특징
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보
        Returns:
            transformed: [B, L, D] - 변환된 특징
        """
        batch_size, seq_len, dim = x.shape
        
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # 1. 대칭성 분석 (A02)
        symmetry_features = self.symmetry_detector(x)
        
        # 2. 스케일 정규화 (A07)
        x_normalized = self.scale_normalizer(x, scale)
        
        # 3. 경계 검출 (A01) - 변환 기준점
        edge_features = self.edge_detector(x_normalized)
        
        # 4. 변환 파라미터 예측
        # 전역 풀링으로 전체 정보 집약
        global_features = x_normalized.mean(dim=1)  # [B, D]
        transform_params = self.transform_predictor(global_features)  # [B, 6]
        
        # 5. 변환 적용
        transformed = self.apply_transformation(x_normalized, transform_params)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            transformed = self.mgp(transformed)
        
        # 6. 등변성 보장 인코딩
        transformed = self.equivariant_encoder(transformed)
        
        # 7. 대칭성 및 경계 정보와 결합하여 정제
        combined = torch.cat([transformed, symmetry_features], dim=-1)
        refined = self.refinement_net(combined)
        
        return refined
    
    def apply_transformation(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        아핀 변환 적용
        
        Args:
            x: [B, L, D]
            params: [B, 6] - [tx, ty, rotation, scale_x, scale_y, shear]
        Returns:
            transformed: [B, L, D]
        """
        B, L, D = x.shape
        
        # 파라미터 추출
        tx = params[:, 0]
        ty = params[:, 1]
        theta = params[:, 2]
        sx = params[:, 3]
        sy = params[:, 4]
        shear = params[:, 5]
        
        # 회전 행렬 요소
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # 아핀 변환 행렬 구성 [B, 2, 3]
        affine_matrix = torch.zeros(B, 2, 3, device=x.device)
        affine_matrix[:, 0, 0] = sx * cos_theta
        affine_matrix[:, 0, 1] = -sy * sin_theta + shear
        affine_matrix[:, 0, 2] = tx
        affine_matrix[:, 1, 0] = sx * sin_theta
        affine_matrix[:, 1, 1] = sy * cos_theta
        affine_matrix[:, 1, 2] = ty
        
        # D 차원을 D/2개의 2D 포인트로 해석
        if D % 2 != 0:
            # 홀수 차원인 경우 패딩
            x = F.pad(x, (0, 1))
            D = D + 1
        
        x_2d = x.view(B, L, D // 2, 2)  # [B, L, D/2, 2]
        x_2d_flat = x_2d.view(B, L * D // 2, 2)  # [B, L*D/2, 2]
        
        # 동차 좌표로 변환
        ones = torch.ones(B, L * D // 2, 1, device=x.device)
        x_homogeneous = torch.cat([x_2d_flat, ones], dim=-1)  # [B, L*D/2, 3]
        
        # 행렬 곱으로 변환 적용
        transformed_2d = torch.bmm(x_homogeneous, affine_matrix.transpose(1, 2))  # [B, L*D/2, 2]
        
        # 원래 형태로 복원
        transformed = transformed_2d.view(B, L, D)
        
        # 패딩 제거 (필요한 경우)
        if self.config.input_dim % 2 != 0:
            transformed = transformed[:, :, :-1]
        
        return transformed
    
    def estimate_transformation(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        입력의 변환 파라미터 추정
        
        Args:
            x: [B, L, D]
        Returns:
            params: 변환 파라미터 딕셔너리
        """
        # 스케일 정규화
        x_normalized = self.scale_normalizer(x)
        
        # 전역 특징 추출
        global_features = x_normalized.mean(dim=1)  # [B, D]
        
        # 변환 파라미터 예측
        transform_params = self.transform_predictor(global_features)  # [B, 6]
        
        # 대칭성 정보 (tuple 반환)
        symmetry_types, symmetry_axis = self.symmetry_detector.detect_symmetry_type(x_normalized)
        
        # 스케일 추정
        estimated_scale = self.scale_normalizer.estimate_scale(x)
        
        params = {
            'translation': transform_params[:, :2],      # [B, 2]
            'rotation': transform_params[:, 2],          # [B]
            'scale': transform_params[:, 3:5],           # [B, 2]
            'shear': transform_params[:, 5],             # [B]
            'symmetry_types': symmetry_types,            # [B, L, 3]
            'symmetry_axis': symmetry_axis,              # [B, L, D]
            'estimated_scale': estimated_scale           # [B]
        }
        
        return params
    
    def align_to_canonical(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        정규 좌표계로 정렬
        
        Args:
            x: [B, L, D]
        Returns:
            transformed: [B, L, D] - 정렬된 특징
            params: 변환 파라미터
        """
        # 변환 파라미터 추정
        params = self.estimate_transformation(x)
        
        # 스케일 정규화
        x_normalized = self.scale_normalizer(x)
        
        # 변환 적용
        transform_params = torch.cat([
            params['translation'],
            params['rotation'].unsqueeze(-1),
            params['scale'],
            params['shear'].unsqueeze(-1)
        ], dim=-1)  # [B, 6]
        
        transformed = self.apply_transformation(x_normalized, transform_params)
        
        # 등변성 인코딩
        if self.config.use_mgp:
            transformed = self.mgp(transformed)
        
        transformed = self.equivariant_encoder(transformed)
        
        return transformed, params
    
    def inverse_transform(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        역변환 적용
        
        Args:
            x: [B, L, D] - 변환된 특징
            params: 변환 파라미터
        Returns:
            original: [B, L, D] - 역변환된 특징
        """
        # 역변환 파라미터 계산
        tx = -params['translation'][:, 0]
        ty = -params['translation'][:, 1]
        theta = -params['rotation']
        sx = 1.0 / (params['scale'][:, 0] + 1e-8)
        sy = 1.0 / (params['scale'][:, 1] + 1e-8)
        shear = -params['shear']
        
        inverse_params = torch.stack([tx, ty, theta, sx, sy, shear], dim=-1)
        
        # 역변환 적용
        original = self.apply_transformation(x, inverse_params)
        
        return original


def create_spatial_transformer(input_dim: int = 128) -> SpatialTransformer:
    """Spatial Transformer 시드 생성 함수"""
    return SpatialTransformer(input_dim=input_dim)

