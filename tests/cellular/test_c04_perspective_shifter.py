"""
SEED-C04 Perspective Shifter 테스트
"""

import pytest
import torch
import torch.nn as nn

from seeds.cellular.c04_perspective_shifter import (
    PerspectiveShifter,
    create_perspective_shifter
)


@pytest.fixture
def device():
    """테스트용 디바이스"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def perspective_shifter(device):
    """테스트용 Perspective Shifter 인스턴스"""
    model = create_perspective_shifter(
        input_dim=128,
        hidden_dim=192,
        num_perspectives=3
    )
    return model.to(device)


@pytest.fixture
def sample_input(device):
    """테스트용 샘플 입력"""
    batch_size = 4
    seq_len = 16
    input_dim = 128
    return torch.randn(batch_size, seq_len, input_dim, device=device)


def test_perspective_shifter_initialization(perspective_shifter):
    """초기화 테스트"""
    assert isinstance(perspective_shifter, PerspectiveShifter)
    assert perspective_shifter.input_dim == 128
    assert perspective_shifter.hidden_dim == 192
    assert perspective_shifter.num_perspectives == 3


def test_perspective_shifter_forward(perspective_shifter, sample_input):
    """Forward pass 테스트"""
    perspectives, consistency_score, info = perspective_shifter(sample_input)
    
    batch_size, seq_len, input_dim = sample_input.shape
    
    # 출력 형태 검증
    assert perspectives.shape == (batch_size, 3, seq_len, input_dim)
    assert consistency_score.shape == (batch_size,)
    
    # 일관성 점수 범위 검증 (0~1)
    assert torch.all(consistency_score >= 0.0)
    assert torch.all(consistency_score <= 1.0)
    
    # 변환 파라미터 검증
    assert 'transform_params' in info
    assert info['transform_params'].shape == (batch_size, 3, 6)


def test_perspective_shifter_output_shape(perspective_shifter):
    """다양한 입력 크기에 대한 출력 형태 테스트"""
    test_cases = [
        (2, 8, 128),
        (4, 16, 128),
        (8, 32, 128),
    ]
    
    for batch_size, seq_len, input_dim in test_cases:
        x = torch.randn(batch_size, seq_len, input_dim, device=perspective_shifter.spatial_encoder[0].weight.device)
        perspectives, consistency_score, info = perspective_shifter(x)
        
        assert perspectives.shape == (batch_size, 3, seq_len, input_dim)
        assert consistency_score.shape == (batch_size,)


def test_shift_perspective(perspective_shifter, sample_input):
    """관점 전환 기능 테스트"""
    for target_idx in range(3):
        shifted_view, info = perspective_shifter.shift_perspective(
            sample_input,
            target_view_idx=target_idx
        )
        
        batch_size, seq_len, input_dim = sample_input.shape
        
        # 출력 형태 검증
        assert shifted_view.shape == (batch_size, seq_len, input_dim)
        
        # 메타데이터 검증
        assert 'selected_perspective' in info
        assert info['selected_perspective'] == target_idx
        assert 'consistency_score' in info


def test_compare_perspectives(perspective_shifter, sample_input):
    """관점 비교 기능 테스트"""
    comparison = perspective_shifter.compare_perspectives(sample_input)
    
    batch_size = sample_input.shape[0]
    
    # 비교 결과 검증
    assert 'perspectives' in comparison
    assert 'similarities' in comparison
    assert 'consistency_score' in comparison
    assert 'diversity_score' in comparison
    
    # 유사도 행렬 형태 검증
    assert comparison['similarities'].shape == (batch_size, 3, 3)
    
    # 대각선 요소는 1.0에 가까워야 함 (자기 자신과의 유사도)
    diagonal = torch.diagonal(comparison['similarities'], dim1=1, dim2=2)
    assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=0.1)
    
    # 다양성 점수 범위 검증
    assert torch.all(comparison['diversity_score'] >= 0.0)
    assert torch.all(comparison['diversity_score'] <= 1.0)


def test_transformation_application(perspective_shifter, sample_input):
    """변환 적용 테스트"""
    batch_size, seq_len, input_dim = sample_input.shape
    
    # 테스트용 변환 파라미터
    transform_params = torch.randn(batch_size, 6, device=sample_input.device)
    
    # 변환 적용
    transformed = perspective_shifter._apply_transformation(
        sample_input,
        transform_params
    )
    
    # 출력 형태 검증
    assert transformed.shape == sample_input.shape
    
    # 변환 후에도 유효한 텐서인지 확인
    assert not torch.isnan(transformed).any()
    assert not torch.isinf(transformed).any()


def test_consistency_across_perspectives(perspective_shifter, sample_input):
    """관점 간 일관성 테스트"""
    perspectives, consistency_score, info = perspective_shifter(sample_input)
    
    # 일관성 점수가 합리적인 범위인지 확인
    assert consistency_score.mean() > 0.0
    assert consistency_score.mean() < 1.0
    
    # 대칭성 특징이 모든 관점에 반영되었는지 확인
    assert 'symmetry_features' in info
    symmetry_features = info['symmetry_features']
    assert symmetry_features.shape[0] == sample_input.shape[0]


def test_gradient_flow(perspective_shifter, sample_input):
    """그래디언트 흐름 테스트"""
    sample_input.requires_grad = True
    
    perspectives, consistency_score, info = perspective_shifter(sample_input)
    
    # 손실 함수 정의 (예: 일관성 최대화)
    loss = -consistency_score.mean()
    
    # 역전파
    loss.backward()
    
    # 그래디언트가 생성되었는지 확인
    assert sample_input.grad is not None
    assert not torch.isnan(sample_input.grad).any()


def test_parameter_count(perspective_shifter):
    """파라미터 수 테스트"""
    total_params = sum(p.numel() for p in perspective_shifter.parameters())
    
    # 목표 파라미터 수: ~1.2M
    target_params = 1_200_000
    tolerance = 0.2  # 20% 허용 오차
    
    assert total_params > target_params * (1 - tolerance)
    assert total_params < target_params * (1 + tolerance)
    
    print(f"Total parameters: {total_params:,}")


def test_with_scale_parameter(perspective_shifter, sample_input):
    """스케일 파라미터를 사용한 테스트"""
    batch_size = sample_input.shape[0]
    scale = torch.rand(batch_size, 1, device=sample_input.device)
    
    perspectives, consistency_score, info = perspective_shifter(
        sample_input,
        scale=scale
    )
    
    # 정상적으로 처리되었는지 확인
    assert perspectives.shape[0] == batch_size
    assert not torch.isnan(perspectives).any()


def test_reproducibility(perspective_shifter, sample_input):
    """재현성 테스트"""
    # 동일한 시드로 두 번 실행
    torch.manual_seed(42)
    perspectives1, score1, info1 = perspective_shifter(sample_input)
    
    torch.manual_seed(42)
    perspectives2, score2, info2 = perspective_shifter(sample_input)
    
    # 결과가 동일한지 확인
    assert torch.allclose(perspectives1, perspectives2, atol=1e-6)
    assert torch.allclose(score1, score2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
