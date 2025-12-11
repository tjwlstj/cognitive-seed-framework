"""
SEED-C07: Boundary Detector

의미 경계를 탐지하는 Cellular 레벨 시드입니다.

구성 시드:
- A01: Edge Detector (저수준 경계 검출)
- M03: Pattern Completer (패턴 완성 및 분석)
- M06: Context Integrator (맥락 기반 경계 정제)

주요 기능:
- 의미 단위 경계 탐지
- 계층적 경계 표현 (단어/구/문장/문단)
- 경계 신뢰도 점수
- 맥락 기반 경계 조정

Author: Manus AI
Date: 2025-12-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from seeds.base import BaseSeed, SeedConfig
from seeds.atomic.a01_edge_detector import EdgeDetector
from seeds.molecular.m03_pattern_completer import PatternCompleter
from seeds.molecular.m06_context_integrator import ContextIntegrator


@dataclass
class BoundaryDetectorConfig(SeedConfig):
    """Boundary Detector 설정"""
    seed_id: str = "SEED-C07"
    name: str = "Boundary Detector"
    level: int = 2
    category: str = "Pattern"
    bit_depth: str = "FP8"
    params: int = 800000
    input_dim: int = 128
    output_dim: int = 128
    
    # C07 특화 설정
    num_boundary_levels: int = 4  # 단어/구/문장/문단
    confidence_threshold: float = 0.5
    context_window: int = 5
    num_heads: int = 8
    dropout: float = 0.1


class BoundaryDetector(BaseSeed):
    """
    SEED-C07: Boundary Detector
    
    의미 경계를 탐지하고 계층적으로 표현합니다.
    
    주요 기능:
    - 저수준 경계 검출 (A01)
    - 패턴 기반 경계 분석 (M03)
    - 맥락 기반 경계 정제 (M06)
    - 계층적 경계 분류
    - 경계 신뢰도 계산
    
    Examples:
        >>> detector = BoundaryDetector(input_dim=128)
        >>> x = torch.randn(4, 50, 128)
        >>> boundaries = detector(x)
        >>> boundaries['boundaries'].shape
        torch.Size([4, 50, 4])  # 4 levels
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_boundary_levels: int = 4,
        confidence_threshold: float = 0.5,
        context_window: int = 5,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        config = BoundaryDetectorConfig(
            input_dim=input_dim,
            output_dim=input_dim,
            num_boundary_levels=num_boundary_levels,
            confidence_threshold=confidence_threshold,
            context_window=context_window,
            num_heads=num_heads,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        
        # 컴포넌트 초기화
        self._init_atomic_molecular_seeds()
        self._init_boundary_modules()
        self._init_hierarchical_classifier()
        self._init_confidence_estimator()
    
    def _init_atomic_molecular_seeds(self):
        """Atomic/Molecular seeds 초기화"""
        
        # A01: Edge Detector (저수준 경계 검출)
        self.edge_detector = EdgeDetector(self.config.input_dim)
        
        # M03: Pattern Completer (패턴 완성 및 분석)
        self.pattern_completer = PatternCompleter(self.config.input_dim)
        
        # M06: Context Integrator (맥락 통합)
        self.context_integrator = ContextIntegrator(
            input_dim=self.config.input_dim,
            num_heads=self.config.num_heads,
            context_window=self.config.context_window,
            dropout=self.config.dropout
        )
    
    def _init_boundary_modules(self):
        """경계 검출 모듈 초기화"""
        
        # Edge feature processor
        self.edge_processor = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.input_dim),
            nn.LayerNorm(self.config.input_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
        
        # Pattern feature processor
        self.pattern_processor = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.input_dim),
            nn.LayerNorm(self.config.input_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
        
        # Context feature processor
        self.context_processor = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.input_dim),
            nn.LayerNorm(self.config.input_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
        
        # Feature fusion
        self.feature_fusion = nn.MultiheadAttention(
            embed_dim=self.config.input_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True
        )
        
        self.fusion_norm = nn.LayerNorm(self.config.input_dim)
    
    def _init_hierarchical_classifier(self):
        """계층적 경계 분류기 초기화"""
        
        # 각 레벨별 분류기
        self.level_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.input_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.input_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(self.config.num_boundary_levels)
        ])
        
        # 계층 간 관계 모델링
        self.hierarchy_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.input_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.input_dim * 4,
                dropout=self.config.dropout,
                batch_first=True
            ),
            num_layers=2
        )
    
    def _init_confidence_estimator(self):
        """신뢰도 추정기 초기화"""
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.config.input_dim + self.config.num_boundary_levels, 
                     self.config.input_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: [B, L, D] - 입력 시퀀스
            mask: [B, L] - 패딩 마스크 (1=유효, 0=패딩)
            return_details: 상세 정보 반환 여부
        
        Returns:
            Dict containing:
                - boundaries: [B, L, num_levels] - 각 레벨별 경계 확률
                - confidence: [B, L] - 경계 신뢰도
                - features: [B, L, D] - 통합 특징 (if return_details)
        """
        B, L, D = x.shape
        
        # 1. 저수준 경계 검출 (A01)
        edge_features = self.edge_detector(x)  # [B, L, D]
        edge_features = self.edge_processor(edge_features)
        
        # 2. 패턴 기반 분석 (M03)
        pattern_features = self.pattern_completer(x, mask=mask)  # [B, L, D]
        pattern_features = self.pattern_processor(pattern_features)
        
        # 3. 맥락 통합 (M06)
        context_features = self.context_integrator(x)  # [B, L, D]
        context_features = self.context_processor(context_features)
        
        # 4. 특징 융합
        fused_features = self._fuse_features(
            edge_features, pattern_features, context_features
        )  # [B, L, D]
        
        # 5. 계층적 경계 분류
        boundaries = self._classify_boundaries(fused_features)  # [B, L, num_levels]
        
        # 6. 신뢰도 추정
        confidence = self._estimate_confidence(fused_features, boundaries)  # [B, L]
        
        # 7. 결과 구성
        result = {
            'boundaries': boundaries,
            'confidence': confidence
        }
        
        if return_details:
            result.update({
                'features': fused_features,
                'edge_features': edge_features,
                'pattern_features': pattern_features,
                'context_features': context_features
            })
        
        return result
    
    def _fuse_features(
        self,
        edge_features: torch.Tensor,
        pattern_features: torch.Tensor,
        context_features: torch.Tensor
    ) -> torch.Tensor:
        """
        특징 융합
        
        Args:
            edge_features: [B, L, D]
            pattern_features: [B, L, D]
            context_features: [B, L, D]
        
        Returns:
            fused: [B, L, D]
        """
        # Stack features
        stacked = torch.stack([
            edge_features,
            pattern_features,
            context_features
        ], dim=2)  # [B, L, 3, D]
        
        B, L, N, D = stacked.shape
        stacked = stacked.reshape(B * L, N, D)  # [B*L, 3, D]
        
        # Multi-head attention fusion
        fused, _ = self.feature_fusion(
            stacked, stacked, stacked
        )  # [B*L, 3, D]
        
        # Take mean across features
        fused = fused.mean(dim=1)  # [B*L, D]
        fused = fused.reshape(B, L, D)  # [B, L, D]
        
        # Residual connection
        fused = self.fusion_norm(fused + context_features)
        
        return fused
    
    def _classify_boundaries(self, features: torch.Tensor) -> torch.Tensor:
        """
        계층적 경계 분류
        
        Args:
            features: [B, L, D]
        
        Returns:
            boundaries: [B, L, num_levels]
        """
        # 계층 간 관계 인코딩
        encoded = self.hierarchy_encoder(features)  # [B, L, D]
        
        # 각 레벨별 분류
        level_probs = []
        for classifier in self.level_classifiers:
            prob = classifier(encoded)  # [B, L, 1]
            level_probs.append(prob)
        
        boundaries = torch.cat(level_probs, dim=-1)  # [B, L, num_levels]
        
        # 계층적 제약: 상위 레벨 경계는 하위 레벨 경계의 부분집합
        # (단어 경계 ⊆ 구 경계 ⊆ 문장 경계 ⊆ 문단 경계)
        boundaries = self._apply_hierarchical_constraints(boundaries)
        
        return boundaries
    
    def _apply_hierarchical_constraints(self, boundaries: torch.Tensor) -> torch.Tensor:
        """
        계층적 제약 적용
        
        Args:
            boundaries: [B, L, num_levels]
        
        Returns:
            constrained: [B, L, num_levels]
        """
        # 하위 레벨부터 상위 레벨로 제약 전파
        constrained = boundaries.clone()
        
        for i in range(1, self.config.num_boundary_levels):
            # 상위 레벨은 하위 레벨보다 작거나 같아야 함
            constrained[:, :, i] = torch.min(
                constrained[:, :, i],
                constrained[:, :, i-1]
            )
        
        return constrained
    
    def _estimate_confidence(
        self,
        features: torch.Tensor,
        boundaries: torch.Tensor
    ) -> torch.Tensor:
        """
        경계 신뢰도 추정
        
        Args:
            features: [B, L, D]
            boundaries: [B, L, num_levels]
        
        Returns:
            confidence: [B, L]
        """
        # 특징과 경계 정보 결합
        combined = torch.cat([features, boundaries], dim=-1)  # [B, L, D+num_levels]
        
        # 신뢰도 추정
        confidence = self.confidence_estimator(combined).squeeze(-1)  # [B, L]
        
        return confidence
    
    def detect_boundaries(
        self,
        x: torch.Tensor,
        level: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        경계 검출 (이진 마스크)
        
        Args:
            x: [B, L, D] - 입력 시퀀스
            level: 검출할 경계 레벨 (None이면 모든 레벨)
            threshold: 임계값 (None이면 config 값 사용)
        
        Returns:
            mask: [B, L] or [B, L, num_levels] - 경계 마스크
        """
        if threshold is None:
            threshold = self.config.confidence_threshold
        
        result = self.forward(x)
        boundaries = result['boundaries']  # [B, L, num_levels]
        confidence = result['confidence']  # [B, L]
        
        # 신뢰도 필터링
        boundaries = boundaries * (confidence.unsqueeze(-1) > threshold).float()
        
        if level is not None:
            # 특정 레벨만 반환
            mask = (boundaries[:, :, level] > threshold).float()
        else:
            # 모든 레벨 반환
            mask = (boundaries > threshold).float()
        
        return mask
    
    def get_boundary_segments(
        self,
        x: torch.Tensor,
        level: int = 0
    ) -> List[List[Tuple[int, int]]]:
        """
        경계로 구분된 세그먼트 추출
        
        Args:
            x: [B, L, D] - 입력 시퀀스
            level: 경계 레벨
        
        Returns:
            segments: 배치별 세그먼트 리스트 [(start, end), ...]
        """
        boundaries = self.detect_boundaries(x, level=level)  # [B, L]
        B, L = boundaries.shape
        
        segments_batch = []
        for b in range(B):
            boundary_indices = torch.where(boundaries[b] > 0)[0].tolist()
            
            # 시작과 끝 추가
            boundary_indices = [0] + boundary_indices + [L]
            
            # 세그먼트 생성
            segments = [
                (boundary_indices[i], boundary_indices[i+1])
                for i in range(len(boundary_indices) - 1)
            ]
            
            segments_batch.append(segments)
        
        return segments_batch
    
    def compute_boundary_metrics(
        self,
        predicted: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        경계 검출 성능 평가
        
        Args:
            predicted: [B, L, num_levels] - 예측된 경계
            ground_truth: [B, L, num_levels] - 정답 경계
        
        Returns:
            metrics: 평가 메트릭
        """
        # 이진화
        pred_binary = (predicted > self.config.confidence_threshold).float()
        gt_binary = (ground_truth > 0.5).float()
        
        # 레벨별 메트릭 계산
        metrics = {}
        for level in range(self.config.num_boundary_levels):
            pred_level = pred_binary[:, :, level]
            gt_level = gt_binary[:, :, level]
            
            # Precision, Recall, F1
            tp = (pred_level * gt_level).sum()
            fp = (pred_level * (1 - gt_level)).sum()
            fn = ((1 - pred_level) * gt_level).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f'level_{level}_precision'] = precision.item()
            metrics[f'level_{level}_recall'] = recall.item()
            metrics[f'level_{level}_f1'] = f1.item()
        
        # 전체 평균
        metrics['avg_precision'] = sum(
            metrics[f'level_{i}_precision'] 
            for i in range(self.config.num_boundary_levels)
        ) / self.config.num_boundary_levels
        
        metrics['avg_recall'] = sum(
            metrics[f'level_{i}_recall'] 
            for i in range(self.config.num_boundary_levels)
        ) / self.config.num_boundary_levels
        
        metrics['avg_f1'] = sum(
            metrics[f'level_{i}_f1'] 
            for i in range(self.config.num_boundary_levels)
        ) / self.config.num_boundary_levels
        
        return metrics


def create_boundary_detector(
    input_dim: int = 128,
    num_boundary_levels: int = 4,
    **kwargs
) -> BoundaryDetector:
    """Boundary Detector 시드 생성 함수"""
    return BoundaryDetector(
        input_dim=input_dim,
        num_boundary_levels=num_boundary_levels,
        **kwargs
    )
