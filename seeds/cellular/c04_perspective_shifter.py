"""
SEED-C04: Perspective Shifter

관점 전환을 통해 다중 관점을 생성하고 관점 간 일관성을 유지하는 Cellular 레벨 시드입니다.

Category: Spatial/Analogy
Composed From: M04 (Spatial Transformer) + M07 (Analogy Mapper) + A02 (Symmetry Detector)
Target Params: ~1.2M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from seeds.base import BaseSeed, SeedConfig


@dataclass
class PerspectiveShifterConfig(SeedConfig):
    """Perspective Shifter 설정"""
    seed_id: str = "SEED-C04"
    name: str = "Perspective Shifter"
    level: int = 2
    category: str = "Spatial/Analogy"
    bit_depth: str = "FP8"
    params: int = 1200000
    input_dim: int = 128
    output_dim: int = 128
    hidden_dim: int = 192
    num_perspectives: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    consistency_weight: float = 0.3


class PerspectiveShifter(BaseSeed):
    """
    SEED-C04: Perspective Shifter
    
    관점 전환을 통해 다중 관점을 생성하고 관점 간 일관성을 유지합니다.
    
    주요 기능:
    - 다중 관점 생성 (M04 기반 공간 변환)
    - 관점 간 구조적 매핑 (M07 기반)
    - 대칭성 기반 관점 추론 (A02 기반)
    - 관점 일관성 유지
    - 관점 전환 설명 생성
    
    입력:
    - input_view: 입력 관점 [B, L, D]
    - target_perspective: 목표 관점 정보 (선택적)
    
    출력:
    - perspectives: 생성된 관점들 [B, num_perspectives, L, D]
    - consistency_score: 관점 간 일관성 점수 [B]
    - transformation_params: 변환 파라미터
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 192,
        num_perspectives: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        consistency_weight: float = 0.3
    ):
        config = PerspectiveShifterConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_perspectives=num_perspectives,
            num_heads=num_heads,
            dropout=dropout,
            consistency_weight=consistency_weight
        )
        super().__init__(config)
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_perspectives = num_perspectives
        self.num_heads = num_heads
        self.consistency_weight = consistency_weight
        
        # 1. Spatial Transformer (M04 아이디어: 공간 변환)
        self.spatial_encoder = self._build_spatial_encoder()
        
        # 2. Perspective Generator (M04 기반: 다중 관점 생성)
        self.perspective_generators = nn.ModuleList([
            self._build_perspective_generator() for _ in range(num_perspectives)
        ])
        
        # 3. Symmetry Analyzer (A02 아이디어: 대칭성 분석)
        self.symmetry_analyzer = self._build_symmetry_analyzer()
        
        # 4. Structural Mapper (M07 아이디어: 관점 간 매핑)
        self.structural_mapper = self._build_structural_mapper()
        
        # 5. Consistency Enforcer
        self.consistency_enforcer = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 6. Perspective Fusion
        self.perspective_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_perspectives, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # 7. Transformation Parameter Predictor
        self.transform_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6 * num_perspectives)  # 각 관점당 6개 파라미터
        )
        
        # 8. Consistency Scorer
        self.consistency_scorer = nn.Sequential(
            nn.Linear(hidden_dim * num_perspectives, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _build_spatial_encoder(self) -> nn.Module:
        """공간 특징 인코더 (M04 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_perspective_generator(self) -> nn.Module:
        """개별 관점 생성기"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def _build_symmetry_analyzer(self) -> nn.Module:
        """대칭성 분석기 (A02 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        )
    
    def _build_structural_mapper(self) -> nn.Module:
        """구조적 매핑 네트워크 (M07 아이디어)"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        target_perspective: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        context: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: [B, L, D] - 입력 관점
            target_perspective: [B, L, D] - 목표 관점 (선택적)
            scale: [B, 1] - 스케일 매개변수
            context: 추가 맥락 정보
        
        Returns:
            perspectives: [B, num_perspectives, L, D] - 생성된 관점들
            consistency_score: [B] - 관점 간 일관성 점수
            info: 변환 정보 딕셔너리
        """
        batch_size, seq_len, dim = x.shape
        
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 1. 공간 특징 추출 (M04 아이디어)
        spatial_features = self.spatial_encoder(x)  # [B, L, hidden_dim]
        
        # 2. 대칭성 분석 (A02 아이디어)
        symmetry_features = self.symmetry_analyzer(spatial_features)  # [B, L, hidden_dim]
        
        # 3. 변환 파라미터 예측
        global_features = spatial_features.mean(dim=1)  # [B, hidden_dim]
        transform_params = self.transform_predictor(global_features)  # [B, 6*num_perspectives]
        transform_params = transform_params.view(batch_size, self.num_perspectives, 6)
        
        # 4. 다중 관점 생성
        perspectives = []
        for i, generator in enumerate(self.perspective_generators):
            # 각 관점 생성기 적용
            perspective = generator(spatial_features)  # [B, L, hidden_dim]
            
            # 변환 적용
            transformed = self._apply_transformation(
                perspective, 
                transform_params[:, i, :]
            )  # [B, L, hidden_dim]
            
            # 대칭성 정보 통합
            transformed = transformed + symmetry_features * 0.1
            
            perspectives.append(transformed)
        
        perspectives_tensor = torch.stack(perspectives, dim=1)  # [B, num_perspectives, L, hidden_dim]
        
        # 5. 관점 간 구조적 매핑 (M07 아이디어)
        mapped_perspectives = []
        for i in range(self.num_perspectives):
            for j in range(i + 1, self.num_perspectives):
                # 두 관점 간 매핑
                p_i = perspectives_tensor[:, i, :, :]  # [B, L, hidden_dim]
                p_j = perspectives_tensor[:, j, :, :]  # [B, L, hidden_dim]
                
                # 구조적 유사성 계산
                combined = torch.cat([p_i, p_j], dim=-1)  # [B, L, hidden_dim*2]
                mapped = self.structural_mapper(combined)  # [B, L, hidden_dim]
                mapped_perspectives.append(mapped)
        
        # 6. 관점 일관성 강화
        if len(mapped_perspectives) > 0:
            mapped_stack = torch.stack(mapped_perspectives, dim=1)  # [B, num_pairs, L, hidden_dim]
            mapped_flat = mapped_stack.view(batch_size, -1, self.hidden_dim)  # [B, num_pairs*L, hidden_dim]
            
            # Self-attention으로 일관성 강화
            consistent_features, _ = self.consistency_enforcer(
                mapped_flat, mapped_flat, mapped_flat
            )  # [B, num_pairs*L, hidden_dim]
        else:
            consistent_features = perspectives_tensor.view(batch_size, -1, self.hidden_dim)
        
        # 7. 관점 융합
        # 각 관점의 대표 특징 추출
        perspective_reps = []
        for i in range(self.num_perspectives):
            rep = perspectives_tensor[:, i, :, :].mean(dim=1)  # [B, hidden_dim]
            perspective_reps.append(rep)
        
        perspective_concat = torch.cat(perspective_reps, dim=-1)  # [B, hidden_dim*num_perspectives]
        fused = self.perspective_fusion(perspective_concat)  # [B, input_dim]
        
        # 8. 일관성 점수 계산
        consistency_score = self.consistency_scorer(perspective_concat).squeeze(-1)  # [B]
        
        # 9. 출력 형태 조정
        # perspectives_tensor를 input_dim으로 프로젝션
        output_projection = nn.Linear(self.hidden_dim, self.input_dim).to(x.device)
        perspectives_output = output_projection(
            perspectives_tensor.view(batch_size * self.num_perspectives, seq_len, self.hidden_dim)
        ).view(batch_size, self.num_perspectives, seq_len, self.input_dim)
        
        # 변환 정보 딕셔너리
        info = {
            'transform_params': transform_params,  # [B, num_perspectives, 6]
            'symmetry_features': symmetry_features,  # [B, L, hidden_dim]
            'fused_perspective': fused,  # [B, input_dim]
            'consistency_score': consistency_score  # [B]
        }
        
        return perspectives_output, consistency_score, info
    
    def _apply_transformation(
        self,
        x: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
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
        sx = torch.sigmoid(params[:, 3]) + 0.5  # 0.5 ~ 1.5
        sy = torch.sigmoid(params[:, 4]) + 0.5
        shear = torch.tanh(params[:, 5]) * 0.5  # -0.5 ~ 0.5
        
        # 회전 행렬 요소
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # 아핀 변환 행렬 구성 [B, 2, 3]
        affine_matrix = torch.zeros(B, 2, 3, device=x.device)
        affine_matrix[:, 0, 0] = sx * cos_theta
        affine_matrix[:, 0, 1] = -sy * sin_theta + shear
        affine_matrix[:, 0, 2] = tx * 0.1  # 작은 평행이동
        affine_matrix[:, 1, 0] = sx * sin_theta
        affine_matrix[:, 1, 1] = sy * cos_theta
        affine_matrix[:, 1, 2] = ty * 0.1
        
        # D 차원을 D/2개의 2D 포인트로 해석
        D_padded = D
        if D % 2 != 0:
            x = F.pad(x, (0, 1))
            D_padded = D + 1
        
        x_2d = x.view(B, L, D_padded // 2, 2)  # [B, L, D/2, 2]
        x_2d_flat = x_2d.view(B, L * D_padded // 2, 2)  # [B, L*D/2, 2]
        
        # 동차 좌표로 변환
        ones = torch.ones(B, L * D_padded // 2, 1, device=x.device)
        x_homogeneous = torch.cat([x_2d_flat, ones], dim=-1)  # [B, L*D/2, 3]
        
        # 행렬 곱으로 변환 적용
        transformed_2d = torch.bmm(x_homogeneous, affine_matrix.transpose(1, 2))  # [B, L*D/2, 2]
        
        # 원래 형태로 복원
        transformed = transformed_2d.view(B, L, D_padded)
        
        # 패딩 제거
        if D % 2 != 0:
            transformed = transformed[:, :, :-1]
        
        return transformed
    
    def shift_perspective(
        self,
        x: torch.Tensor,
        target_view_idx: int = 0
    ) -> Tuple[torch.Tensor, Dict]:
        """
        특정 관점으로 전환
        
        Args:
            x: [B, L, D] - 입력
            target_view_idx: 목표 관점 인덱스
        
        Returns:
            shifted_view: [B, L, D] - 전환된 관점
            info: 변환 정보
        """
        perspectives, consistency_score, info = self.forward(x)
        
        # 목표 관점 선택
        shifted_view = perspectives[:, target_view_idx, :, :]  # [B, L, D]
        
        info['selected_perspective'] = target_view_idx
        info['consistency_score'] = consistency_score
        
        return shifted_view, info
    
    def compare_perspectives(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        생성된 관점들 간의 비교 분석
        
        Args:
            x: [B, L, D] - 입력
        
        Returns:
            comparison: 비교 결과 딕셔너리
        """
        perspectives, consistency_score, info = self.forward(x)
        
        B, num_p, L, D = perspectives.shape
        
        # 관점 간 유사도 계산
        similarities = torch.zeros(B, num_p, num_p, device=x.device)
        for i in range(num_p):
            for j in range(num_p):
                p_i = perspectives[:, i, :, :].reshape(B, -1)
                p_j = perspectives[:, j, :, :].reshape(B, -1)
                sim = F.cosine_similarity(p_i, p_j, dim=-1)
                similarities[:, i, j] = sim
        
        comparison = {
            'perspectives': perspectives,
            'similarities': similarities,
            'consistency_score': consistency_score,
            'transform_params': info['transform_params'],
            'diversity_score': 1.0 - similarities.mean(dim=[1, 2])  # [B]
        }
        
        return comparison


def create_perspective_shifter(
    input_dim: int = 128,
    hidden_dim: int = 192,
    num_perspectives: int = 3
) -> PerspectiveShifter:
    """Perspective Shifter 시드 생성 함수"""
    return PerspectiveShifter(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_perspectives=num_perspectives
    )
