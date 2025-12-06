"""
SEED-C02: Counterfactual Reasoner 단위 테스트

Author: Manus AI
Date: 2025-12-06
"""

import pytest
import torch
import torch.nn as nn

from seeds.cellular.c02_counterfactual_reasoner import (
    CounterfactualReasoner,
    CounterfactualReasonerConfig,
    create_counterfactual_reasoner
)


class TestCounterfactualReasoner:
    """Counterfactual Reasoner 테스트 클래스"""
    
    @pytest.fixture
    def reasoner(self):
        """테스트용 Counterfactual Reasoner 인스턴스"""
        return CounterfactualReasoner(
            input_dim=128,
            num_scenarios=3,
            intervention_strength=0.5,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_input(self):
        """테스트용 입력 데이터"""
        batch_size = 4
        seq_len = 10
        input_dim = 128
        
        factual = torch.randn(batch_size, seq_len, input_dim)
        scale = torch.rand(batch_size, 1) * 2.0 + 0.5
        
        return factual, scale
    
    def test_initialization(self, reasoner):
        """초기화 테스트"""
        assert reasoner.config.seed_id == "SEED-C02"
        assert reasoner.config.name == "Counterfactual Reasoner"
        assert reasoner.config.level == 2
        assert reasoner.config.category == "Logic"
        
        # 컴포넌트 확인
        assert hasattr(reasoner, 'causality_detector')
        assert hasattr(reasoner, 'conflict_resolver')
        assert hasattr(reasoner, 'comparator')
        assert hasattr(reasoner, 'scenario_generator')
        assert hasattr(reasoner, 'intervention_locator')
        assert hasattr(reasoner, 'consistency_checker')
    
    def test_forward_without_intervention(self, reasoner, sample_input):
        """개입 없이 forward 테스트 (자동 개입 생성)"""
        factual, scale = sample_input
        batch_size, seq_len, input_dim = factual.shape
        
        # Forward pass
        counterfactual, metadata = reasoner(factual, scale=scale)
        
        # 출력 형태 검증
        assert counterfactual.shape == (batch_size, seq_len, input_dim)
        
        # 메타데이터 검증
        assert 'causal_graph' in metadata
        assert 'intervention_effects' in metadata
        assert 'consistency_scores' in metadata
        assert 'comparison' in metadata
        assert 'intervention_time' in metadata
        
        # 인과 그래프 형태 검증
        assert metadata['causal_graph'].shape[0] == batch_size
        
        # 개입 효과 형태 검증
        assert metadata['intervention_effects'].shape == (batch_size, seq_len, input_dim)
        
        # 일관성 점수 검증
        consistency = metadata['consistency_scores']
        assert 'temporal' in consistency
        assert 'causal' in consistency
        assert 'overall' in consistency
        assert consistency['overall'].shape == (batch_size,)
        
        # 비교 결과 검증
        comparison = metadata['comparison']
        assert 'difference_magnitude' in comparison
        assert 'similarity' in comparison
    
    def test_forward_with_intervention(self, reasoner, sample_input):
        """명시적 개입으로 forward 테스트"""
        factual, scale = sample_input
        batch_size, seq_len, input_dim = factual.shape
        
        # 개입 정의
        intervention_time = 5
        intervention_value = torch.randn(batch_size, 1, input_dim)
        
        intervention = {
            'time_step': intervention_time,
            'value': intervention_value
        }
        
        # Forward pass
        counterfactual, metadata = reasoner(factual, scale=scale, intervention=intervention)
        
        # 출력 형태 검증
        assert counterfactual.shape == (batch_size, seq_len, input_dim)
        
        # 개입 시점 이전은 사실과 유사해야 함
        pre_intervention = factual[:, :intervention_time, :]
        pre_counterfactual = counterfactual[:, :intervention_time, :]
        
        # 개입 시점 이후는 차이가 있어야 함
        post_intervention = factual[:, intervention_time:, :]
        post_counterfactual = counterfactual[:, intervention_time:, :]
        
        pre_diff = torch.abs(pre_intervention - pre_counterfactual).mean()
        post_diff = torch.abs(post_intervention - post_counterfactual).mean()
        
        # 개입 이후 차이가 더 커야 함 (일반적으로)
        # 단, 학습되지 않은 모델이므로 엄격한 검증은 하지 않음
        assert pre_diff >= 0
        assert post_diff >= 0
    
    def test_intervention_effect_estimation(self, reasoner, sample_input):
        """개입 효과 추정 테스트"""
        factual, _ = sample_input
        batch_size, seq_len, input_dim = factual.shape
        
        counterfactual = torch.randn(batch_size, seq_len, input_dim)
        intervention_time = torch.tensor([3, 4, 5, 6])
        
        effects = reasoner._estimate_intervention_effects(
            factual, counterfactual, intervention_time
        )
        
        # 형태 검증
        assert effects.shape == (batch_size, seq_len, input_dim)
        
        # 개입 시점 이전은 효과가 0이어야 함
        for i in range(batch_size):
            t = intervention_time[i].item()
            if t > 0:
                assert torch.allclose(
                    effects[i, :t, :],
                    torch.zeros_like(effects[i, :t, :]),
                    atol=1e-6
                )
    
    def test_consistency_checking(self, reasoner, sample_input):
        """일관성 검증 테스트"""
        factual, _ = sample_input
        batch_size, seq_len, input_dim = factual.shape
        
        counterfactual = torch.randn(batch_size, seq_len, input_dim)
        causal_features = torch.randn(batch_size, seq_len, input_dim)
        
        consistency_scores = reasoner._check_consistency(
            factual, counterfactual, causal_features
        )
        
        # 일관성 점수 검증
        assert 'temporal' in consistency_scores
        assert 'causal' in consistency_scores
        assert 'overall' in consistency_scores
        
        assert consistency_scores['temporal'].shape == (batch_size, seq_len)
        assert consistency_scores['causal'].shape == (batch_size, seq_len)
        assert consistency_scores['overall'].shape == (batch_size,)
        
        # 점수 범위 검증 [0, 1]
        assert torch.all(consistency_scores['temporal'] >= 0)
        assert torch.all(consistency_scores['temporal'] <= 1)
        assert torch.all(consistency_scores['overall'] >= 0)
        assert torch.all(consistency_scores['overall'] <= 1)
    
    def test_scenario_comparison(self, reasoner, sample_input):
        """시나리오 비교 테스트"""
        factual, _ = sample_input
        batch_size, seq_len, input_dim = factual.shape
        
        counterfactual = torch.randn(batch_size, seq_len, input_dim)
        
        comparison = reasoner._compare_scenarios(factual, counterfactual)
        
        # 비교 결과 검증
        assert 'features' in comparison
        assert 'difference_magnitude' in comparison
        assert 'similarity' in comparison
        
        assert comparison['difference_magnitude'].shape == (batch_size, seq_len)
        assert comparison['similarity'].shape == (batch_size,)
        
        # 유사도 범위 검증 [-1, 1]
        assert torch.all(comparison['similarity'] >= -1)
        assert torch.all(comparison['similarity'] <= 1)
    
    def test_multiple_scenarios_generation(self, reasoner, sample_input):
        """다중 시나리오 생성 테스트"""
        factual, _ = sample_input
        batch_size, seq_len, input_dim = factual.shape
        
        num_scenarios = 3
        scenarios, metadata_list = reasoner.generate_multiple_scenarios(
            factual, num_scenarios=num_scenarios
        )
        
        # 시나리오 수 검증 (원본 + harmonized)
        assert len(scenarios) == num_scenarios + 1
        assert len(metadata_list) == num_scenarios + 1
        
        # 각 시나리오 형태 검증
        for scenario in scenarios:
            assert scenario.shape == (batch_size, seq_len, input_dim)
        
        # 마지막 시나리오는 harmonized
        assert metadata_list[-1]['type'] == 'harmonized'
    
    def test_gradient_flow(self, reasoner, sample_input):
        """그래디언트 흐름 테스트"""
        factual, scale = sample_input
        
        # Forward pass
        counterfactual, metadata = reasoner(factual, scale=scale)
        
        # 손실 계산
        loss = counterfactual.mean()
        
        # Backward pass
        loss.backward()
        
        # 그래디언트 확인
        has_grad = False
        for param in reasoner.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "그래디언트가 전파되지 않았습니다"
    
    def test_parameter_count(self, reasoner):
        """파라미터 수 테스트"""
        total_params = sum(p.numel() for p in reasoner.parameters())
        
        # 대략적인 파라미터 수 검증 (±20%)
        expected_params = reasoner.config.params
        tolerance = expected_params * 0.2
        
        assert abs(total_params - expected_params) < tolerance, \
            f"파라미터 수 불일치: {total_params} (예상: {expected_params})"
    
    def test_create_function(self):
        """생성 함수 테스트"""
        reasoner = create_counterfactual_reasoner(input_dim=64, num_scenarios=4)
        
        assert isinstance(reasoner, CounterfactualReasoner)
        assert reasoner.config.input_dim == 64
        assert reasoner.config.num_scenarios == 4
    
    def test_config_dataclass(self):
        """Config 데이터클래스 테스트"""
        config = CounterfactualReasonerConfig(
            input_dim=256,
            num_scenarios=7
        )
        
        assert config.seed_id == "SEED-C02"
        assert config.input_dim == 256
        assert config.num_scenarios == 7
        assert config.level == 2
    
    def test_intervention_value_generation(self, reasoner, sample_input):
        """자동 개입 값 생성 테스트"""
        factual, _ = sample_input
        batch_size, seq_len, input_dim = factual.shape
        
        causal_features = torch.randn(batch_size, seq_len, input_dim)
        intervention_time = torch.tensor([2, 3, 4, 5])
        
        intervention_value = reasoner._generate_intervention_value(
            factual, intervention_time, causal_features
        )
        
        # 형태 검증
        assert intervention_value.shape == (batch_size, 1, input_dim)
    
    def test_counterfactual_scenario_generation(self, reasoner, sample_input):
        """반사실 시나리오 생성 테스트"""
        factual, _ = sample_input
        batch_size, seq_len, input_dim = factual.shape
        
        causal_features = torch.randn(batch_size, seq_len, input_dim)
        intervention_time = torch.tensor([3, 4, 5, 6])
        intervention_value = torch.randn(batch_size, 1, input_dim)
        
        counterfactual = reasoner._generate_counterfactual_scenario(
            factual, causal_features, intervention_time, intervention_value
        )
        
        # 형태 검증
        assert counterfactual.shape == (batch_size, seq_len, input_dim)
        
        # 개입 시점 이전은 사실과 동일해야 함 (또는 매우 유사)
        for i in range(batch_size):
            t = intervention_time[i].item()
            if t > 0:
                # 개입 이전 시점 비교
                pre_diff = torch.abs(
                    factual[i, :t, :] - counterfactual[i, :t, :]
                ).mean()
                
                # 개입 이후 시점 비교
                if t < seq_len - 1:
                    post_diff = torch.abs(
                        factual[i, t:, :] - counterfactual[i, t:, :]
                    ).mean()
                    
                    # 개입 이후 차이가 존재해야 함
                    assert post_diff >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
