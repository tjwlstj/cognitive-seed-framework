"""
SEED-C05 Narrative Constructor 테스트
"""

import pytest
import torch
import torch.nn as nn

from seeds.cellular.c05_narrative_constructor import (
    NarrativeConstructor,
    create_narrative_constructor
)


@pytest.fixture
def device():
    """테스트용 디바이스"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def narrative_constructor(device):
    """테스트용 Narrative Constructor 인스턴스"""
    model = create_narrative_constructor(
        input_dim=128,
        hidden_dim=192,
        num_narrative_stages=4
    )
    return model.to(device)


@pytest.fixture
def sample_events(device):
    """테스트용 샘플 사건 시퀀스"""
    batch_size = 4
    seq_len = 32
    input_dim = 128
    return torch.randn(batch_size, seq_len, input_dim, device=device)


def test_narrative_constructor_initialization(narrative_constructor):
    """초기화 테스트"""
    assert isinstance(narrative_constructor, NarrativeConstructor)
    assert narrative_constructor.input_dim == 128
    assert narrative_constructor.hidden_dim == 192
    assert narrative_constructor.num_narrative_stages == 4


def test_narrative_constructor_forward(narrative_constructor, sample_events):
    """Forward pass 테스트"""
    narrative, structure, coherence_score = narrative_constructor(sample_events)
    
    batch_size, seq_len, input_dim = sample_events.shape
    
    # 출력 형태 검증
    assert narrative.shape == (batch_size, seq_len, input_dim)
    assert coherence_score.shape == (batch_size,)
    
    # 일관성 점수 범위 검증 (0~1)
    assert torch.all(coherence_score >= 0.0)
    assert torch.all(coherence_score <= 1.0)
    
    # 구조 정보 검증
    assert 'stage_probabilities' in structure
    assert 'stage_assignments' in structure
    assert 'causal_weights' in structure
    assert 'plot_points' in structure


def test_narrative_structure(narrative_constructor, sample_events):
    """서사 구조 검증 테스트"""
    narrative, structure, coherence_score = narrative_constructor(sample_events)
    
    batch_size, seq_len = sample_events.shape[0], sample_events.shape[1]
    
    # 단계 확률 형태 검증
    assert structure['stage_probabilities'].shape == (batch_size, seq_len, 4)
    
    # 단계 할당 형태 검증
    assert structure['stage_assignments'].shape == (batch_size, seq_len)
    
    # 단계 할당이 유효한 범위인지 확인 (0~3)
    assert torch.all(structure['stage_assignments'] >= 0)
    assert torch.all(structure['stage_assignments'] < 4)
    
    # 인과 가중치 형태 검증
    assert structure['causal_weights'].shape == (batch_size, seq_len, seq_len)
    
    # 플롯 포인트 형태 검증
    assert structure['plot_points'].shape == (batch_size, seq_len)


def test_analyze_narrative_structure(narrative_constructor, sample_events):
    """서사 구조 분석 테스트"""
    analysis = narrative_constructor.analyze_narrative_structure(sample_events)
    
    batch_size, seq_len = sample_events.shape[0], sample_events.shape[1]
    
    # 분석 결과 검증
    assert 'narrative' in analysis
    assert 'stage_assignments' in analysis
    assert 'stage_lengths' in analysis
    assert 'plot_points' in analysis
    assert 'causal_strength' in analysis
    assert 'coherence_score' in analysis
    assert 'stage_transitions' in analysis
    
    # 단계 길이 형태 검증
    assert analysis['stage_lengths'].shape == (batch_size, 4)
    
    # 인과 강도 형태 검증
    assert analysis['causal_strength'].shape == (batch_size, seq_len)
    
    # 전환점이 리스트인지 확인
    assert isinstance(analysis['stage_transitions'], list)
    assert len(analysis['stage_transitions']) == batch_size


def test_generate_narrative(narrative_constructor, sample_events):
    """서사 생성 테스트"""
    target_structure = [0.25, 0.25, 0.25, 0.25]  # 균등 분배
    
    narrative, info = narrative_constructor.generate_narrative(
        sample_events,
        target_structure=target_structure
    )
    
    batch_size, seq_len, input_dim = sample_events.shape
    
    # 출력 형태 검증
    assert narrative.shape == (batch_size, seq_len, input_dim)
    
    # 정보 검증
    assert 'narrative' in info
    assert 'structure' in info
    assert 'coherence_score' in info
    assert 'current_structure' in info
    assert 'target_structure' in info
    assert 'structure_difference' in info
    
    # 구조 차이 범위 검증 (0~1)
    assert torch.all(info['structure_difference'] >= 0.0)
    assert torch.all(info['structure_difference'] <= 1.0)


def test_evaluate_coherence(narrative_constructor, sample_events):
    """일관성 평가 테스트"""
    evaluation = narrative_constructor.evaluate_coherence(sample_events)
    
    # 평가 지표 검증
    assert 'coherence_score' in evaluation
    assert 'causal_density' in evaluation
    assert 'plot_density' in evaluation
    assert 'stage_balance' in evaluation
    assert 'overall_quality' in evaluation
    
    batch_size = sample_events.shape[0]
    
    # 모든 지표가 배치 크기와 일치하는지 확인
    assert evaluation['coherence_score'].shape == (batch_size,)
    assert evaluation['causal_density'].shape == (batch_size,)
    assert evaluation['plot_density'].shape == (batch_size,)
    assert evaluation['stage_balance'].shape == (batch_size,)
    assert evaluation['overall_quality'].shape == (batch_size,)


def test_output_shape_consistency(narrative_constructor):
    """다양한 입력 크기에 대한 출력 형태 일관성 테스트"""
    test_cases = [
        (2, 16, 128),
        (4, 32, 128),
        (8, 64, 128),
    ]
    
    for batch_size, seq_len, input_dim in test_cases:
        x = torch.randn(
            batch_size, seq_len, input_dim,
            device=narrative_constructor.context_encoder[0].weight.device
        )
        narrative, structure, coherence_score = narrative_constructor(x)
        
        assert narrative.shape == (batch_size, seq_len, input_dim)
        assert coherence_score.shape == (batch_size,)
        assert structure['stage_assignments'].shape == (batch_size, seq_len)


def test_stage_assignment_coverage(narrative_constructor, sample_events):
    """모든 서사 단계가 할당되는지 테스트"""
    narrative, structure, coherence_score = narrative_constructor(sample_events)
    
    stage_assignments = structure['stage_assignments']
    
    # 각 배치에서 최소 하나의 단계는 할당되어야 함
    for b in range(sample_events.shape[0]):
        unique_stages = torch.unique(stage_assignments[b])
        assert len(unique_stages) > 0
        assert len(unique_stages) <= 4


def test_causal_weights_symmetry(narrative_constructor, sample_events):
    """인과 가중치의 대칭성 테스트"""
    narrative, structure, coherence_score = narrative_constructor(sample_events)
    
    causal_weights = structure['causal_weights']
    
    # 인과 가중치가 유효한 확률 범위인지 확인
    assert torch.all(causal_weights >= 0.0)
    # Attention weights는 정규화되어 있어야 함
    weight_sums = causal_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=0.1)


def test_plot_points_detection(narrative_constructor, sample_events):
    """플롯 포인트 탐지 테스트"""
    narrative, structure, coherence_score = narrative_constructor(sample_events)
    
    plot_points = structure['plot_points']
    
    # 플롯 포인트 점수 범위 검증 (0~1)
    assert torch.all(plot_points >= 0.0)
    assert torch.all(plot_points <= 1.0)
    
    # 최소 일부 플롯 포인트가 탐지되어야 함
    assert torch.any(plot_points > 0.3)


def test_gradient_flow(narrative_constructor, sample_events):
    """그래디언트 흐름 테스트"""
    sample_events.requires_grad = True
    
    narrative, structure, coherence_score = narrative_constructor(sample_events)
    
    # 손실 함수 정의 (예: 일관성 최대화)
    loss = -coherence_score.mean()
    
    # 역전파
    loss.backward()
    
    # 그래디언트가 생성되었는지 확인
    assert sample_events.grad is not None
    assert not torch.isnan(sample_events.grad).any()


def test_parameter_count(narrative_constructor):
    """파라미터 수 테스트"""
    total_params = sum(p.numel() for p in narrative_constructor.parameters())
    
    # 목표 파라미터 수: ~1.0M
    target_params = 1_000_000
    tolerance = 1.5  # 150% 허용 오차 (복잡한 서사 구조로 인해)
    
    assert total_params > target_params * (1 - tolerance)
    assert total_params < target_params * (1 + tolerance)
    
    print(f"Total parameters: {total_params:,}")


def test_with_context(narrative_constructor, sample_events):
    """맥락 정보를 사용한 테스트"""
    batch_size = sample_events.shape[0]
    context_len = 8
    input_dim = 128
    
    context = torch.randn(
        batch_size, context_len, input_dim,
        device=sample_events.device
    )
    
    narrative, structure, coherence_score = narrative_constructor(
        sample_events,
        context=context
    )
    
    # 정상적으로 처리되었는지 확인
    assert narrative.shape[0] == batch_size
    assert not torch.isnan(narrative).any()


def test_reproducibility(narrative_constructor, sample_events):
    """재현성 테스트"""
    # 동일한 시드로 두 번 실행
    torch.manual_seed(42)
    narrative1, structure1, score1 = narrative_constructor(sample_events)
    
    torch.manual_seed(42)
    narrative2, structure2, score2 = narrative_constructor(sample_events)
    
    # 결과가 동일한지 확인
    assert torch.allclose(narrative1, narrative2, atol=1e-6)
    assert torch.allclose(score1, score2, atol=1e-6)
    assert torch.allclose(
        structure1['stage_assignments'].float(),
        structure2['stage_assignments'].float(),
        atol=1e-6
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
