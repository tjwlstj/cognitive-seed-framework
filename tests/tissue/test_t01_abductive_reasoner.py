"""
SEED-T01: Abductive Reasoner 단위 테스트

Author: Manus AI (누스양)
Date: 2026-01-06
"""

import pytest
import torch
import torch.nn as nn

from seeds.tissue.t01_abductive_reasoner import (
    T01AbductiveReasoner,
    AbductiveReasonerConfig,
    create_t01_abductive_reasoner
)


class TestT01AbductiveReasoner:
    """T01 Abductive Reasoner 테스트"""
    
    @pytest.fixture
    def seed(self):
        """테스트용 시드 인스턴스"""
        config = AbductiveReasonerConfig(
            input_dim=128,
            num_hypotheses=8,
            dropout=0.1
        )
        return T01AbductiveReasoner(config)
    
    @pytest.fixture
    def sample_input(self):
        """테스트용 입력 데이터"""
        B, T, D = 2, 10, 128
        return torch.randn(B, T, D)
    
    def test_initialization(self, seed):
        """초기화 테스트"""
        assert seed.config.seed_id == "SEED-T01"
        assert seed.config.name == "Abductive Reasoner"
        assert seed.config.level == 3
        assert seed.config.category == "Logic"
        assert seed.config.input_dim == 128
        assert seed.config.output_dim == 128
        assert seed.config.num_hypotheses == 8
    
    def test_forward_shape(self, seed, sample_input):
        """Forward pass 출력 형태 테스트"""
        B, T, D = sample_input.shape
        
        # 기본 forward
        output = seed(sample_input)
        assert output.shape == (B, T, D)
        
        # 가설 정보 포함 forward
        output, hypotheses_info = seed(sample_input, return_hypotheses=True)
        assert output.shape == (B, T, D)
        assert 'hypotheses' in hypotheses_info
        assert 'counterfactuals' in hypotheses_info
        assert 'explanation_scores' in hypotheses_info
        assert 'plausibility_scores' in hypotheses_info
        assert 'parsimony_scores' in hypotheses_info
        assert 'combined_scores' in hypotheses_info
        assert 'attention_weights' in hypotheses_info
    
    def test_hypothesis_generation(self, seed, sample_input):
        """가설 생성 테스트"""
        B, T, D = sample_input.shape
        output, hypotheses_info = seed(sample_input, return_hypotheses=True)
        
        hypotheses = hypotheses_info['hypotheses']
        assert hypotheses.shape == (B, T, seed.config.num_hypotheses, D)
        
        # 각 가설이 서로 다른지 확인
        for i in range(seed.config.num_hypotheses - 1):
            h1 = hypotheses[:, :, i, :]
            h2 = hypotheses[:, :, i+1, :]
            # 완전히 동일하지 않아야 함
            assert not torch.allclose(h1, h2, atol=1e-6)
    
    def test_counterfactual_reasoning(self, seed, sample_input):
        """반사실 추론 테스트"""
        B, T, D = sample_input.shape
        output, hypotheses_info = seed(sample_input, return_hypotheses=True)
        
        counterfactuals = hypotheses_info['counterfactuals']
        assert counterfactuals.shape == (B, T, seed.config.num_hypotheses, D)
        
        # 반사실이 가설과 다른지 확인
        hypotheses = hypotheses_info['hypotheses']
        for i in range(seed.config.num_hypotheses):
            h = hypotheses[:, :, i, :]
            cf = counterfactuals[:, :, i, :]
            # 완전히 동일하지 않아야 함
            assert not torch.allclose(h, cf, atol=1e-6)
    
    def test_scoring_mechanisms(self, seed, sample_input):
        """스코어링 메커니즘 테스트"""
        B, T, _ = sample_input.shape
        output, hypotheses_info = seed(sample_input, return_hypotheses=True)
        
        # 설명력 스코어
        explanation_scores = hypotheses_info['explanation_scores']
        assert explanation_scores.shape == (B, T, seed.config.num_hypotheses, 1)
        assert torch.all(explanation_scores >= 0) and torch.all(explanation_scores <= 1)
        
        # 그럴듯함 스코어
        plausibility_scores = hypotheses_info['plausibility_scores']
        assert plausibility_scores.shape == (B, T, seed.config.num_hypotheses, 1)
        assert torch.all(plausibility_scores >= 0) and torch.all(plausibility_scores <= 1)
        
        # 간결성 스코어
        parsimony_scores = hypotheses_info['parsimony_scores']
        assert parsimony_scores.shape == (B, T, seed.config.num_hypotheses, 1)
        assert torch.all(parsimony_scores >= 0) and torch.all(parsimony_scores <= 1)
        
        # 종합 스코어
        combined_scores = hypotheses_info['combined_scores']
        assert combined_scores.shape == (B, T, seed.config.num_hypotheses, 1)
        assert torch.all(combined_scores >= 0) and torch.all(combined_scores <= 1)
    
    def test_attention_weights(self, seed, sample_input):
        """어텐션 가중치 테스트"""
        B, T, _ = sample_input.shape
        output, hypotheses_info = seed(sample_input, return_hypotheses=True)
        
        attention_weights = hypotheses_info['attention_weights']
        assert attention_weights.shape == (B, T, seed.config.num_hypotheses)
        
        # 어텐션 가중치 합이 1에 가까운지 확인
        attention_sum = attention_weights.sum(dim=-1)
        assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5)
    
    def test_best_hypothesis_index(self, seed, sample_input):
        """최선 가설 인덱스 반환 테스트"""
        B, T, _ = sample_input.shape
        best_indices = seed.get_best_hypothesis_index(sample_input)
        
        assert best_indices.shape == (B, T)
        assert torch.all(best_indices >= 0) and torch.all(best_indices < seed.config.num_hypotheses)
    
    def test_explanation_quality(self, seed, sample_input):
        """설명 품질 지표 테스트"""
        B, T, _ = sample_input.shape
        quality_metrics = seed.get_explanation_quality(sample_input)
        
        assert 'explanation_score' in quality_metrics
        assert 'plausibility_score' in quality_metrics
        assert 'parsimony_score' in quality_metrics
        assert 'combined_score' in quality_metrics
        assert 'confidence' in quality_metrics
        
        for key, value in quality_metrics.items():
            assert value.shape == (B, T)
            assert torch.all(value >= 0) and torch.all(value <= 1)
    
    def test_causal_structure_detection(self, seed, sample_input):
        """인과 구조 탐지 테스트"""
        B, T, D = sample_input.shape
        output, hypotheses_info = seed(sample_input, return_hypotheses=True)
        
        causal_structure = hypotheses_info['causal_structure']
        assert causal_structure.shape == (B, T, D)
    
    def test_concept_abstraction(self, seed, sample_input):
        """개념 추상화 테스트"""
        B, T, D = sample_input.shape
        output, hypotheses_info = seed(sample_input, return_hypotheses=True)
        
        abstract_concepts = hypotheses_info['abstract_concepts']
        assert abstract_concepts.shape == (B, T, D)
    
    def test_residual_connection(self, seed, sample_input):
        """잔차 연결 테스트"""
        output = seed(sample_input)
        
        # 출력이 입력과 완전히 다르지 않은지 확인 (잔차 연결 효과)
        # 하지만 완전히 동일하지도 않아야 함
        assert not torch.allclose(output, sample_input, atol=1e-6)
        
        # 잔차 가중치가 학습 가능한 파라미터인지 확인
        assert seed.residual_weight.requires_grad
    
    def test_batch_consistency(self, seed):
        """배치 일관성 테스트"""
        # 배치 크기 1과 2의 결과가 일관되는지 확인
        x1 = torch.randn(1, 10, 128)
        x2 = torch.cat([x1, x1], dim=0)  # [2, 10, 128]
        
        seed.eval()
        with torch.no_grad():
            output1 = seed(x1)
            output2 = seed(x2)
        
        # 첫 번째 배치 요소가 동일한지 확인
        assert torch.allclose(output1[0], output2[0], atol=1e-5)
        assert torch.allclose(output2[0], output2[1], atol=1e-5)
    
    def test_gradient_flow(self, seed, sample_input):
        """그래디언트 흐름 테스트"""
        seed.train()
        output = seed(sample_input)
        loss = output.sum()
        loss.backward()
        
        # 주요 파라미터에 그래디언트가 있는지 확인
        assert seed.observation_encoder[0].weight.grad is not None
        assert seed.explanation_scorer[0].weight.grad is not None
        assert seed.output_projection[0].weight.grad is not None
    
    def test_factory_function(self):
        """팩토리 함수 테스트"""
        seed = create_t01_abductive_reasoner(
            input_dim=64,
            num_hypotheses=5
        )
        
        assert isinstance(seed, T01AbductiveReasoner)
        assert seed.config.input_dim == 64
        assert seed.config.output_dim == 64
        assert seed.config.num_hypotheses == 5
        
        # Forward pass 테스트
        x = torch.randn(2, 10, 64)
        output = seed(x)
        assert output.shape == (2, 10, 64)
    
    def test_parameter_count(self, seed):
        """파라미터 수 테스트"""
        total_params = sum(p.numel() for p in seed.parameters())
        
        # ~3.0M 파라미터 목표
        # 실제 파라미터 수는 구성 시드 포함하여 더 많을 수 있음
        print(f"Total parameters: {total_params:,}")
        
        # 최소 1M 이상의 파라미터가 있어야 함
        assert total_params >= 1_000_000
    
    def test_eval_mode(self, seed, sample_input):
        """평가 모드 테스트"""
        seed.eval()
        
        with torch.no_grad():
            output1 = seed(sample_input)
            output2 = seed(sample_input)
        
        # 평가 모드에서는 동일한 입력에 대해 동일한 출력
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_different_sequence_lengths(self, seed):
        """다양한 시퀀스 길이 테스트"""
        B, D = 2, 128
        
        for T in [5, 10, 20]:
            x = torch.randn(B, T, D)
            output = seed(x)
            assert output.shape == (B, T, D)
    
    def test_mask_support(self, seed, sample_input):
        """마스크 지원 테스트"""
        B, T, D = sample_input.shape
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, T//2:] = False  # 후반부 마스킹
        
        # 마스크를 사용한 forward (현재는 사용하지 않지만 인터페이스 테스트)
        output = seed(sample_input, mask=mask)
        assert output.shape == (B, T, D)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
