"""
SEED-T01: Abductive Reasoner 테스트

Author: Manus AI
Date: 2025-12-31
"""

import pytest
import torch
import torch.nn as nn

from seeds.tissue.t01_abductive_reasoner import (
    AbductiveReasoner,
    AbductiveReasonerConfig,
    create_abductive_reasoner
)


class TestAbductiveReasonerConfig:
    """AbductiveReasonerConfig 테스트"""
    
    def test_default_config(self):
        """기본 설정 테스트"""
        config = AbductiveReasonerConfig()
        
        assert config.seed_id == "SEED-T01"
        assert config.name == "Abductive Reasoner"
        assert config.level == 3
        assert config.category == "Logic"
        assert config.bit_depth == "FP16"
        assert config.input_dim == 256
        assert config.output_dim == 256
        assert config.num_hypotheses == 10
        assert config.max_explanation_length == 20
    
    def test_custom_config(self):
        """커스텀 설정 테스트"""
        config = AbductiveReasonerConfig(
            input_dim=512,
            num_hypotheses=5,
            max_explanation_length=15
        )
        
        assert config.input_dim == 512
        assert config.output_dim == 512
        assert config.num_hypotheses == 5
        assert config.max_explanation_length == 15


class TestAbductiveReasoner:
    """AbductiveReasoner 테스트"""
    
    @pytest.fixture
    def reasoner(self):
        """테스트용 reasoner 인스턴스"""
        return AbductiveReasoner(input_dim=128, num_hypotheses=5, max_explanation_length=10)
    
    def test_initialization(self, reasoner):
        """초기화 테스트"""
        assert isinstance(reasoner, AbductiveReasoner)
        assert isinstance(reasoner, nn.Module)
        assert reasoner.config.input_dim == 128
        assert reasoner.config.num_hypotheses == 5
        assert reasoner.config.max_explanation_length == 10
    
    def test_forward_shape(self, reasoner):
        """순전파 출력 shape 테스트"""
        batch_size = 4
        num_observations = 15
        dim = 128
        
        observations = torch.randn(batch_size, num_observations, dim)
        explanation, confidence = reasoner(observations)
        
        assert explanation.shape == (batch_size, reasoner.config.max_explanation_length, dim)
        assert confidence.shape == (batch_size, 1)
    
    def test_forward_with_scale(self, reasoner):
        """스케일 매개변수를 사용한 순전파 테스트"""
        batch_size = 4
        num_observations = 15
        dim = 128
        
        observations = torch.randn(batch_size, num_observations, dim)
        scale = torch.rand(batch_size, 1)
        
        explanation, confidence = reasoner(observations, scale=scale)
        
        assert explanation.shape == (batch_size, reasoner.config.max_explanation_length, dim)
        assert confidence.shape == (batch_size, 1)
    
    def test_confidence_range(self, reasoner):
        """신뢰도 범위 테스트 (0~1)"""
        batch_size = 4
        num_observations = 15
        dim = 128
        
        observations = torch.randn(batch_size, num_observations, dim)
        _, confidence = reasoner(observations)
        
        assert torch.all(confidence >= 0.0)
        assert torch.all(confidence <= 1.0)
    
    def test_encode_observations(self, reasoner):
        """관찰 인코딩 테스트"""
        batch_size = 4
        num_observations = 15
        dim = 128
        
        observations = torch.randn(batch_size, num_observations, dim)
        encoded, summary = reasoner.encode_observations(observations)
        
        assert encoded.shape == (batch_size, num_observations, dim)
        assert summary.shape == (batch_size, dim)
    
    def test_generate_hypotheses(self, reasoner):
        """가설 생성 테스트"""
        batch_size = 4
        num_observations = 15
        dim = 128
        
        observations = torch.randn(batch_size, num_observations, dim)
        obs_encoded, obs_summary = reasoner.encode_observations(observations)
        
        hypotheses = reasoner.generate_hypotheses(obs_encoded, obs_summary)
        
        expected_shape = (
            batch_size,
            reasoner.config.num_hypotheses,
            reasoner.config.max_explanation_length,
            dim
        )
        assert hypotheses.shape == expected_shape
    
    def test_evaluate_causality(self, reasoner):
        """인과성 평가 테스트"""
        batch_size = 4
        num_hypotheses = 5
        max_len = 10
        dim = 128
        
        hypotheses = torch.randn(batch_size, num_hypotheses, max_len, dim)
        obs_summary = torch.randn(batch_size, dim)
        
        causality_scores = reasoner.evaluate_causality(hypotheses, obs_summary)
        
        assert causality_scores.shape == (batch_size, num_hypotheses)
        assert torch.all(causality_scores >= 0.0)
        assert torch.all(causality_scores <= 1.0)
    
    def test_evaluate_consistency(self, reasoner):
        """일관성 평가 테스트"""
        batch_size = 4
        num_hypotheses = 5
        max_len = 10
        dim = 128
        
        hypotheses = torch.randn(batch_size, num_hypotheses, max_len, dim)
        obs_summary = torch.randn(batch_size, dim)
        
        consistency_scores = reasoner.evaluate_consistency(hypotheses, obs_summary)
        
        assert consistency_scores.shape == (batch_size, num_hypotheses)
        assert torch.all(consistency_scores >= 0.0)
        assert torch.all(consistency_scores <= 1.0)
    
    def test_evaluate_novelty(self, reasoner):
        """참신성 평가 테스트"""
        batch_size = 4
        num_hypotheses = 5
        max_len = 10
        dim = 128
        
        hypotheses = torch.randn(batch_size, num_hypotheses, max_len, dim)
        
        novelty_scores = reasoner.evaluate_novelty(hypotheses)
        
        assert novelty_scores.shape == (batch_size, num_hypotheses)
        assert torch.all(novelty_scores >= 0.0)
        assert torch.all(novelty_scores <= 1.0)
    
    def test_evaluate_simplicity(self, reasoner):
        """단순성 평가 테스트"""
        batch_size = 4
        num_hypotheses = 5
        max_len = 10
        dim = 128
        
        hypotheses = torch.randn(batch_size, num_hypotheses, max_len, dim)
        
        simplicity_scores = reasoner.evaluate_simplicity(hypotheses)
        
        assert simplicity_scores.shape == (batch_size, num_hypotheses)
        assert torch.all(simplicity_scores >= 0.0)
        assert torch.all(simplicity_scores <= 1.0)
    
    def test_select_best_explanation(self, reasoner):
        """최선의 설명 선택 테스트"""
        batch_size = 4
        num_hypotheses = 5
        max_len = 10
        dim = 128
        
        hypotheses = torch.randn(batch_size, num_hypotheses, max_len, dim)
        causality_scores = torch.rand(batch_size, num_hypotheses)
        consistency_scores = torch.rand(batch_size, num_hypotheses)
        novelty_scores = torch.rand(batch_size, num_hypotheses)
        simplicity_scores = torch.rand(batch_size, num_hypotheses)
        
        best_explanation, confidence, all_scores = reasoner.select_best_explanation(
            hypotheses,
            causality_scores,
            consistency_scores,
            novelty_scores,
            simplicity_scores
        )
        
        assert best_explanation.shape == (batch_size, max_len, dim)
        assert confidence.shape == (batch_size, 1)
        assert all_scores.shape == (batch_size, num_hypotheses)
    
    def test_get_all_hypotheses(self, reasoner):
        """모든 가설 반환 테스트"""
        batch_size = 4
        num_observations = 15
        dim = 128
        
        observations = torch.randn(batch_size, num_observations, dim)
        hypotheses, scores = reasoner.get_all_hypotheses(observations)
        
        expected_shape = (
            batch_size,
            reasoner.config.num_hypotheses,
            reasoner.config.max_explanation_length,
            dim
        )
        assert hypotheses.shape == expected_shape
        
        assert "causality" in scores
        assert "consistency" in scores
        assert "novelty" in scores
        assert "simplicity" in scores
        assert "final" in scores
        
        for score_name, score_tensor in scores.items():
            assert score_tensor.shape == (batch_size, reasoner.config.num_hypotheses)
    
    def test_parameter_count(self, reasoner):
        """파라미터 수 테스트"""
        param_count = reasoner.count_parameters()
        
        # Level 3 시드는 대략 10M~15M 파라미터 예상
        assert param_count > 1_000_000  # 최소 1M
        assert param_count < 20_000_000  # 최대 20M
    
    def test_get_metadata(self, reasoner):
        """메타데이터 테스트"""
        metadata = reasoner.get_metadata()
        
        assert metadata["seed_id"] == "SEED-T01"
        assert metadata["name"] == "Abductive Reasoner"
        assert metadata["level"] == 3
        assert metadata["category"] == "Logic"
        assert metadata["bit_depth"] == "FP16"
    
    def test_different_batch_sizes(self, reasoner):
        """다양한 배치 크기 테스트"""
        dim = 128
        num_observations = 15
        
        for batch_size in [1, 2, 8, 16]:
            observations = torch.randn(batch_size, num_observations, dim)
            explanation, confidence = reasoner(observations)
            
            assert explanation.shape[0] == batch_size
            assert confidence.shape[0] == batch_size
    
    def test_different_observation_lengths(self, reasoner):
        """다양한 관찰 길이 테스트"""
        batch_size = 4
        dim = 128
        
        for num_observations in [5, 10, 20, 30]:
            observations = torch.randn(batch_size, num_observations, dim)
            explanation, confidence = reasoner(observations)
            
            assert explanation.shape == (batch_size, reasoner.config.max_explanation_length, dim)
            assert confidence.shape == (batch_size, 1)
    
    def test_gradient_flow(self, reasoner):
        """그래디언트 흐름 테스트"""
        batch_size = 4
        num_observations = 15
        dim = 128
        
        observations = torch.randn(batch_size, num_observations, dim, requires_grad=True)
        explanation, confidence = reasoner(observations)
        
        # 손실 계산 및 역전파
        loss = explanation.mean() + confidence.mean()
        loss.backward()
        
        # 입력에 그래디언트가 전파되었는지 확인
        assert observations.grad is not None
        assert not torch.all(observations.grad == 0)
    
    def test_eval_mode(self, reasoner):
        """평가 모드 테스트"""
        batch_size = 4
        num_observations = 15
        dim = 128
        
        observations = torch.randn(batch_size, num_observations, dim)
        
        # 학습 모드
        reasoner.train()
        output_train, _ = reasoner(observations)
        
        # 평가 모드
        reasoner.eval()
        output_eval, _ = reasoner(observations)
        
        # 출력 shape은 동일해야 함
        assert output_train.shape == output_eval.shape


class TestCreateAbductiveReasoner:
    """create_abductive_reasoner 함수 테스트"""
    
    def test_create_with_defaults(self):
        """기본 설정으로 생성 테스트"""
        reasoner = create_abductive_reasoner()
        
        assert isinstance(reasoner, AbductiveReasoner)
        assert reasoner.config.input_dim == 256
    
    def test_create_with_custom_dim(self):
        """커스텀 차원으로 생성 테스트"""
        reasoner = create_abductive_reasoner(input_dim=512)
        
        assert reasoner.config.input_dim == 512
        assert reasoner.config.output_dim == 512
    
    def test_create_with_kwargs(self):
        """추가 인자로 생성 테스트"""
        reasoner = create_abductive_reasoner(
            input_dim=128,
            num_hypotheses=8,
            max_explanation_length=15
        )
        
        assert reasoner.config.input_dim == 128
        assert reasoner.config.num_hypotheses == 8
        assert reasoner.config.max_explanation_length == 15


class TestAbductiveReasoningScenarios:
    """실제 귀추적 추론 시나리오 테스트"""
    
    @pytest.fixture
    def reasoner(self):
        """테스트용 reasoner 인스턴스"""
        return AbductiveReasoner(input_dim=128, num_hypotheses=5, max_explanation_length=10)
    
    def test_medical_diagnosis_scenario(self, reasoner):
        """의료 진단 시나리오 (증상 → 질병 추론)"""
        batch_size = 2
        num_symptoms = 10  # 10개 증상 관찰
        dim = 128
        
        # 증상 데이터 (관찰)
        symptoms = torch.randn(batch_size, num_symptoms, dim)
        
        # 진단 (설명) 생성
        diagnosis, confidence = reasoner(symptoms)
        
        assert diagnosis.shape == (batch_size, reasoner.config.max_explanation_length, dim)
        assert confidence.shape == (batch_size, 1)
        
        # 신뢰도가 합리적인 범위인지 확인
        assert torch.all(confidence > 0.0)
        assert torch.all(confidence < 1.0)
    
    def test_fault_diagnosis_scenario(self, reasoner):
        """고장 진단 시나리오 (오류 로그 → 원인 추론)"""
        batch_size = 3
        num_logs = 20  # 20개 로그 관찰
        dim = 128
        
        # 오류 로그 데이터
        error_logs = torch.randn(batch_size, num_logs, dim)
        
        # 원인 추론
        root_cause, confidence = reasoner(error_logs)
        
        assert root_cause.shape == (batch_size, reasoner.config.max_explanation_length, dim)
        
        # 모든 가설 확인
        hypotheses, scores = reasoner.get_all_hypotheses(error_logs)
        
        # 인과성 점수가 가장 높은 가설이 선택되었는지 확인
        assert scores["causality"].max(dim=1)[0].mean() > 0.0
    
    def test_scientific_discovery_scenario(self, reasoner):
        """과학적 발견 시나리오 (실험 데이터 → 이론 추론)"""
        batch_size = 1
        num_experiments = 30  # 30개 실험 결과
        dim = 128
        
        # 실험 데이터
        experimental_data = torch.randn(batch_size, num_experiments, dim)
        
        # 이론 추론
        theory, confidence = reasoner(experimental_data)
        
        # 모든 가설과 점수 확인
        hypotheses, scores = reasoner.get_all_hypotheses(experimental_data)
        
        # 일관성과 단순성이 중요한 시나리오
        assert scores["consistency"].mean() > 0.0
        assert scores["simplicity"].mean() > 0.0
    
    def test_detective_reasoning_scenario(self, reasoner):
        """탐정 추론 시나리오 (증거 → 범인 추론)"""
        batch_size = 2
        num_evidences = 12  # 12개 증거
        dim = 128
        
        # 증거 데이터
        evidences = torch.randn(batch_size, num_evidences, dim)
        
        # 범인 추론
        culprit_explanation, confidence = reasoner(evidences)
        
        # 모든 가설 확인
        hypotheses, scores = reasoner.get_all_hypotheses(evidences)
        
        # 최종 점수가 합리적인 분포를 가지는지 확인
        final_scores = scores["final"]
        assert final_scores.std() > 0.0  # 점수가 다양해야 함
        assert final_scores.max() > final_scores.mean()  # 최고 점수가 평균보다 높아야 함


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
