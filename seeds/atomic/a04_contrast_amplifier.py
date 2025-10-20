"""
SEED-A04 — Contrast Amplifier

신호 대비를 증폭하고 노이즈를 억제하는 원자 시드입니다.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSeed, SeedConfig


class ContrastAmplifier(BaseSeed):
    """
    SEED-A04: Contrast Amplifier
    
    Category: Pattern
    Bit: INT8
    Params: ~64
    Purpose: 신호 대비 증폭·노이즈 억제
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 32):
        config = SeedConfig(
            seed_id="SEED-A04",
            name="Contrast Amplifier",
            level=0,
            category="Pattern",
            bit_depth="INT8",
            params=64,
            input_dim=input_dim,
            output_dim=input_dim,
            use_mgp=True,
            use_cse=True,
            geometries=["E"]  # 대비 증폭은 유클리드 공간에서 수행
        )
        super().__init__(config)
        
        self.hidden_dim = hidden_dim
        
        # 신호/노이즈 분리기
        self.signal_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 대비 증폭 게이트
        self.contrast_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # 노이즈 억제 임계값 학습
        self.noise_threshold = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 형태의 입력 텐서
            scale: [B, 1] 형태의 스케일 매개변수
            context: 추가 맥락 정보
        Returns:
            [B, L, D] 형태의 대비가 증폭된 텐서
        """
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x = self.mgp(x)
        
        # 신호 추출
        signal = self.signal_extractor(x)  # [B, L, D]
        
        # 대비 게이트 계산
        gate = self.contrast_gate(x)  # [B, L, D]
        
        # 노이즈 억제 (작은 값 제거)
        magnitude = torch.abs(signal)
        noise_mask = (magnitude > self.noise_threshold).float()
        signal = signal * noise_mask
        
        # 대비 증폭
        amplified = signal * (1.0 + gate)
        
        # 원본과 증폭된 신호 결합
        output = x + amplified
        
        return output
    
    def compute_snr(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        SNR (Signal-to-Noise Ratio) 개선을 계산합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
            output: [B, L, D] 형태의 출력 텐서
        Returns:
            [B] 형태의 SNR 개선 비율
        """
        # 입력의 신호 대 노이즈 비율
        input_signal = torch.mean(torch.abs(x), dim=[1, 2])
        input_noise = torch.std(x, dim=[1, 2])
        input_snr = input_signal / (input_noise + 1e-8)
        
        # 출력의 신호 대 노이즈 비율
        output_signal = torch.mean(torch.abs(output), dim=[1, 2])
        output_noise = torch.std(output, dim=[1, 2])
        output_snr = output_signal / (output_noise + 1e-8)
        
        # SNR 개선 비율
        snr_improvement = output_snr / (input_snr + 1e-8)
        
        return snr_improvement
    
    def denoise(self, x: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        """
        노이즈를 제거합니다.
        
        Args:
            x: [B, L, D] 형태의 입력 텐서
            threshold: 노이즈 임계값 (None이면 학습된 값 사용)
        Returns:
            [B, L, D] 형태의 노이즈가 제거된 텐서
        """
        if threshold is None:
            threshold = self.noise_threshold.item()
        
        magnitude = torch.abs(x)
        noise_mask = (magnitude > threshold).float()
        denoised = x * noise_mask
        
        return denoised


def create_contrast_amplifier(input_dim: int = 128, hidden_dim: int = 32) -> ContrastAmplifier:
    """Contrast Amplifier 시드 생성 함수"""
    return ContrastAmplifier(input_dim=input_dim, hidden_dim=hidden_dim)

