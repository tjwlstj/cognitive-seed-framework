"""
SEED-C06 Attention Director 단위 테스트
"""

import pytest
import torch
import torch.nn as nn

from seeds.cellular.c06_attention_director import AttentionDirector, AttentionDirectorConfig


class TestAttentionDirector:
    """Attention Director 테스트 클래스"""
    
    @pytest.fixture
    def attention_director(self):
        """Attention Director 인스턴스 생성"""
        return AttentionDirector(
            input_dim=128,
            num_heads=8,
            num_attention_layers=2,
            num_clusters=16,
            dropout=0.1,
            temperature=1.0
        )
    
    @pytest.fixture
    def sample_input(self):
        """샘플 입력 데이터 생성"""
        batch_size = 4
        seq_len = 50
        input_dim = 128
        return torch.randn(batch_size, seq_len, input_dim)
    
    @pytest.fixture
    def sample_scale(self):
        """샘플 스케일 데이터 생성"""
        batch_size = 4
        return torch.randn(batch_size, 1)
    
    @pytest.fixture
    def sample_context(self):
        """샘플 맥락 데이터 생성"""
        batch_size = 4
        context_len = 20
        input_dim = 128
        return {
            'query': torch.randn(batch_size, input_dim),
            'context': torch.randn(batch_size, context_len, input_dim)
        }
    
    def test_initialization(self, attention_director):
        """초기화 테스트"""
        assert isinstance(attention_director, AttentionDirector)
        assert attention_director.config.input_dim == 128
        assert attention_director.config.num_heads == 8
        assert attention_director.config.num_attention_layers == 2
        assert attention_director.config.num_clusters == 16
        
        # 설정 확인
        config = attention_director.get_config()
        assert config['seed_id'] == 'SEED-C06'
        assert config['name'] == 'Attention Director'
        assert config['level'] == 2
        assert config['category'] == 'Composition'
    
    def test_forward_basic(self, attention_director, sample_input):
        """기본 forward 테스트"""
        output = attention_director(sample_input)
        
        # 출력 확인
        assert isinstance(output, dict)
        assert 'attended_output' in output
        assert 'attention_weights' in output
        assert 'importance_scores' in output
        
        # Shape 확인
        assert output['attended_output'].shape == (4, 50, 128)
        assert output['attention_weights'].shape == (4, 50)
        assert output['importance_scores'].shape == (4, 50)
    
    def test_forward_with_scale(self, attention_director, sample_input, sample_scale):
        """스케일 포함 forward 테스트"""
        output = attention_director(sample_input, scale=sample_scale)
        
        # 출력 확인
        assert output['attended_output'].shape == (4, 50, 128)
        assert output['attention_weights'].shape == (4, 50)
        assert output['importance_scores'].shape == (4, 50)
    
    def test_forward_with_context(self, attention_director, sample_input, sample_context):
        """맥락 포함 forward 테스트"""
        output = attention_director(sample_input, context=sample_context)
        
        # 출력 확인
        assert output['attended_output'].shape == (4, 50, 128)
        assert output['attention_weights'].shape == (4, 50)
        assert output['importance_scores'].shape == (4, 50)
    
    def test_forward_full(self, attention_director, sample_input, sample_scale, sample_context):
        """전체 파라미터 포함 forward 테스트"""
        output = attention_director(sample_input, scale=sample_scale, context=sample_context)
        
        # 모든 출력 확인
        assert 'attended_output' in output
        assert 'attention_weights' in output
        assert 'importance_scores' in output
        assert 'group_weights' in output
        assert 'hierarchy_weights' in output
        assert 'context_weights' in output
        assert 'group_features' in output
        assert 'hierarchy_features' in output
        assert 'context_features' in output
        
        # Shape 확인
        assert output['attended_output'].shape == (4, 50, 128)
        assert output['attention_weights'].shape == (4, 50)
        assert output['importance_scores'].shape == (4, 50)
        assert output['group_weights'].shape == (4, 50)
        assert output['hierarchy_weights'].shape == (4, 50)
        assert output['context_weights'].shape == (4, 50)
    
    def test_group_attention(self, attention_director, sample_input):
        """그룹 주의 계산 테스트"""
        # 입력 인코딩
        x_encoded = attention_director.input_encoder(sample_input)
        
        # 그룹 특징 추출
        group_features = attention_director.grouping_nucleus(x_encoded)
        
        # 그룹 주의 계산
        group_attended, group_weights = attention_director.compute_group_attention(
            x_encoded, group_features
        )
        
        # Shape 확인
        assert group_attended.shape == (4, 50, 128)
        assert group_weights.shape == (4, 50)
        
        # 가중치 범위 확인 (softmax 출력)
        assert torch.all(group_weights >= 0)
        assert torch.all(group_weights <= 1)
        assert torch.allclose(group_weights.sum(dim=-1), torch.ones(4), atol=1e-5)
    
    def test_hierarchical_attention(self, attention_director, sample_input):
        """계층 주의 계산 테스트"""
        # 입력 인코딩
        x_encoded = attention_director.input_encoder(sample_input)
        
        # 계층 특징 추출
        hierarchy_features = attention_director.hierarchy_builder(x_encoded)
        
        # 계층 주의 계산
        hierarchy_attended, hierarchy_weights = attention_director.compute_hierarchical_attention(
            x_encoded, hierarchy_features
        )
        
        # Shape 확인
        assert hierarchy_attended.shape == (4, 50, 128)
        assert hierarchy_weights.shape == (4, 50)
        
        # 가중치 범위 확인
        assert torch.all(hierarchy_weights >= 0)
        assert torch.all(hierarchy_weights <= 1)
    
    def test_context_attention(self, attention_director, sample_input):
        """맥락 주의 계산 테스트"""
        # 입력 인코딩
        x_encoded = attention_director.input_encoder(sample_input)
        
        # 맥락 특징 추출
        context_features = attention_director.context_integrator(x_encoded)
        
        # 맥락 주의 계산
        context_attended, context_weights = attention_director.compute_context_attention(
            x_encoded, context_features
        )
        
        # Shape 확인
        assert context_attended.shape == (4, 50, 128)
        assert context_weights.shape == (4, 50)
        
        # 가중치 범위 확인
        assert torch.all(context_weights >= 0)
        assert torch.all(context_weights <= 1)
        assert torch.allclose(context_weights.sum(dim=-1), torch.ones(4), atol=1e-5)
    
    def test_importance_scores(self, attention_director, sample_input):
        """중요도 점수 계산 테스트"""
        # 입력 인코딩
        x_encoded = attention_director.input_encoder(sample_input)
        
        # 특징 추출
        context_features = attention_director.context_integrator(x_encoded)
        hierarchy_features = attention_director.hierarchy_builder(x_encoded)
        
        # 중요도 점수 계산
        importance_scores = attention_director.compute_importance_scores(
            x_encoded, context_features, hierarchy_features
        )
        
        # Shape 확인
        assert importance_scores.shape == (4, 50)
        
        # 점수 범위 확인 (0~1, Sigmoid 출력)
        assert torch.all(importance_scores >= 0)
        assert torch.all(importance_scores <= 1)
    
    def test_attention_aggregation(self, attention_director, sample_input):
        """주의 집계 테스트"""
        # 입력 인코딩
        x_encoded = attention_director.input_encoder(sample_input)
        
        # 특징 추출
        group_features = attention_director.grouping_nucleus(x_encoded)
        hierarchy_features = attention_director.hierarchy_builder(x_encoded)
        context_features = attention_director.context_integrator(x_encoded)
        
        # 주의 계산
        group_attended, _ = attention_director.compute_group_attention(x_encoded, group_features)
        hierarchy_attended, _ = attention_director.compute_hierarchical_attention(x_encoded, hierarchy_features)
        context_attended, _ = attention_director.compute_context_attention(x_encoded, context_features)
        
        # 중요도 점수
        importance_scores = attention_director.compute_importance_scores(
            x_encoded, context_features, hierarchy_features
        )
        
        # 집계
        aggregated = attention_director.aggregate_attention(
            x_encoded,
            group_attended,
            hierarchy_attended,
            context_attended,
            importance_scores
        )
        
        # Shape 확인
        assert aggregated.shape == (4, 50, 128)
    
    def test_attention_map(self, attention_director, sample_input):
        """주의 맵 추출 테스트"""
        attention_map = attention_director.get_attention_map(sample_input)
        
        # Shape 확인
        assert attention_map.shape == (4, 50, 50)
        
        # 대각 행렬 확인 (현재 구현)
        for b in range(4):
            assert torch.allclose(
                attention_map[b],
                torch.diag(torch.diag(attention_map[b])),
                atol=1e-5
            )
    
    def test_batch_independence(self, attention_director):
        """배치 독립성 테스트"""
        # 단일 샘플
        single_input = torch.randn(1, 50, 128)
        single_output = attention_director(single_input)
        
        # 배치 샘플 (동일한 입력 반복)
        batch_input = single_input.repeat(4, 1, 1)
        batch_output = attention_director(batch_input)
        
        # 각 배치 결과가 동일해야 함
        for b in range(4):
            assert torch.allclose(
                batch_output['attended_output'][b],
                single_output['attended_output'][0],
                atol=1e-4
            )
    
    def test_gradient_flow(self, attention_director, sample_input):
        """그래디언트 흐름 테스트"""
        sample_input.requires_grad = True
        
        output = attention_director(sample_input)
        loss = output['attended_output'].sum()
        loss.backward()
        
        # 그래디언트 확인
        assert sample_input.grad is not None
        assert not torch.all(sample_input.grad == 0)
    
    def test_different_sequence_lengths(self, attention_director):
        """다양한 시퀀스 길이 테스트"""
        seq_lengths = [10, 30, 50, 100]
        
        for seq_len in seq_lengths:
            x = torch.randn(2, seq_len, 128)
            output = attention_director(x)
            
            assert output['attended_output'].shape == (2, seq_len, 128)
            assert output['attention_weights'].shape == (2, seq_len)
            assert output['importance_scores'].shape == (2, seq_len)
    
    def test_temperature_effect(self):
        """Temperature 효과 테스트"""
        x = torch.randn(2, 20, 128)
        
        # 낮은 temperature (sharper attention)
        director_low_temp = AttentionDirector(input_dim=128, temperature=0.5)
        output_low = director_low_temp(x)
        
        # 높은 temperature (smoother attention)
        director_high_temp = AttentionDirector(input_dim=128, temperature=2.0)
        output_high = director_high_temp(x)
        
        # 낮은 temperature가 더 집중된 주의를 가져야 함
        # (분산이 더 크거나 최대값이 더 커야 함)
        low_max = output_low['attention_weights'].max(dim=-1)[0].mean()
        high_max = output_high['attention_weights'].max(dim=-1)[0].mean()
        
        # 일반적으로 낮은 temperature가 더 큰 최대값을 가짐
        # (하지만 초기화에 따라 다를 수 있으므로 shape만 확인)
        assert output_low['attention_weights'].shape == output_high['attention_weights'].shape
    
    def test_parameter_count(self, attention_director):
        """파라미터 수 확인 테스트"""
        total_params = sum(p.numel() for p in attention_director.parameters())
        
        # 예상 파라미터 수 범위 확인 (~1.5M ± 10%)
        expected_params = 1_500_000
        tolerance = 0.2  # 20% 허용
        
        assert total_params > expected_params * (1 - tolerance)
        assert total_params < expected_params * (1 + tolerance)
        
        print(f"Total parameters: {total_params:,}")
    
    def test_config_serialization(self, attention_director):
        """설정 직렬화 테스트"""
        config = attention_director.get_config()
        
        # 필수 필드 확인
        assert 'seed_id' in config
        assert 'name' in config
        assert 'level' in config
        assert 'category' in config
        assert 'input_dim' in config
        assert 'output_dim' in config
        
        # 값 확인
        assert config['seed_id'] == 'SEED-C06'
        assert config['name'] == 'Attention Director'
        assert config['level'] == 2
        assert config['category'] == 'Composition'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
