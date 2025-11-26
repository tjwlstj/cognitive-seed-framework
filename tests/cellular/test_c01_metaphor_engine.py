"""
SEED-C01: Metaphor Engine 단위 테스트
"""

import pytest
import torch
import torch.nn as nn

from seeds.cellular.c01_metaphor_engine import MetaphorEngine


class TestMetaphorEngine:
    """Metaphor Engine 테스트"""
    
    @pytest.fixture
    def seed(self):
        """테스트용 시드 인스턴스"""
        return MetaphorEngine(
            input_dim=128,
            hidden_dim=180,
            num_heads=8,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터"""
        batch_size = 4
        source_len = 10
        target_len = 12
        input_dim = 128
        
        source = torch.randn(batch_size, source_len, input_dim)
        target = torch.randn(batch_size, target_len, input_dim)
        
        return source, target
    
    def test_initialization(self, seed):
        """초기화 테스트"""
        assert seed is not None
        assert seed.input_dim == 128
        assert seed.hidden_dim == 180
        assert seed.num_heads == 8
        
        # 메타데이터 확인
        metadata = seed.get_metadata()
        assert metadata['seed_id'] == 'SEED-C01'
        assert metadata['name'] == 'Metaphor Engine'
        assert metadata['level'] == 2
        assert metadata['category'] == 'Analogy'
        assert 'M01' in metadata['composed_from']
        assert 'M07' in metadata['composed_from']
        assert 'M05' in metadata['composed_from']
    
    def test_forward_pass(self, seed, sample_data):
        """기본 forward 테스트"""
        source, target = sample_data
        batch_size = source.size(0)
        
        # Forward pass
        metaphor, mapping_score, structural_similarity = seed(source, target)
        
        # 출력 형태 확인
        assert metaphor.shape == (batch_size, 128)
        assert mapping_score.shape == (batch_size,)
        assert structural_similarity.shape == (batch_size,)
        
        # 값 범위 확인
        assert torch.all(mapping_score >= 0) and torch.all(mapping_score <= 1)
        assert torch.all(structural_similarity >= 0) and torch.all(structural_similarity <= 1)
    
    def test_metaphor_generation(self, seed, sample_data):
        """은유 생성 테스트"""
        source, target = sample_data
        
        # 은유 생성
        metaphor, mapping_score, structural_similarity = seed(source, target)
        
        # 은유가 유효한 텐서인지 확인
        assert not torch.isnan(metaphor).any()
        assert not torch.isinf(metaphor).any()
        
        # 매핑 점수가 합리적인 범위인지 확인
        assert mapping_score.mean() > 0.0
        assert mapping_score.mean() < 1.0
    
    def test_mapping_quality(self, seed, sample_data):
        """매핑 품질 평가 테스트"""
        source, target = sample_data
        
        # 은유 생성
        metaphor, mapping_score, structural_similarity = seed(source, target)
        
        # 품질 계산
        quality = seed.compute_metaphor_quality(source, target, metaphor)
        
        # 품질 점수 확인
        assert quality.shape == (source.size(0),)
        assert torch.all(quality >= -1) and torch.all(quality <= 1)  # 코사인 유사도 범위
    
    def test_structural_similarity(self, seed):
        """구조적 유사도 테스트"""
        batch_size = 4
        seq_len = 10
        input_dim = 128
        
        # 유사한 구조
        source = torch.randn(batch_size, seq_len, input_dim)
        target = source + torch.randn(batch_size, seq_len, input_dim) * 0.1  # 약간의 노이즈
        
        # 은유 생성
        metaphor1, _, similarity1 = seed(source, target)
        
        # 완전히 다른 구조
        target_random = torch.randn(batch_size, seq_len, input_dim)
        metaphor2, _, similarity2 = seed(source, target_random)
        
        # 유사한 구조의 유사도가 더 높아야 함
        assert similarity1.mean() > similarity2.mean()
    
    def test_gradient_flow(self, seed, sample_data):
        """그래디언트 흐름 테스트"""
        source, target = sample_data
        
        # Forward pass
        metaphor, mapping_score, structural_similarity = seed(source, target)
        
        # 손실 계산
        loss = metaphor.sum() + mapping_score.sum() + structural_similarity.sum()
        
        # Backward pass
        loss.backward()
        
        # 그래디언트 확인
        for name, param in seed.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_parameter_count(self, seed):
        """파라미터 수 검증"""
        total_params = sum(p.numel() for p in seed.parameters())
        
        # 목표 파라미터 수: ~750K (±10%)
        target_params = 750000
        tolerance = 0.10
        
        assert total_params >= target_params * (1 - tolerance), \
            f"Too few parameters: {total_params} < {target_params * (1 - tolerance)}"
        assert total_params <= target_params * (1 + tolerance), \
            f"Too many parameters: {total_params} > {target_params * (1 + tolerance)}"
        
        print(f"Total parameters: {total_params:,} (target: {target_params:,})")
    
    def test_batch_independence(self, seed):
        """배치 독립성 테스트"""
        batch_size = 4
        seq_len = 10
        input_dim = 128
        
        # 배치 데이터
        source_batch = torch.randn(batch_size, seq_len, input_dim)
        target_batch = torch.randn(batch_size, seq_len, input_dim)
        
        # 배치 처리
        metaphor_batch, _, _ = seed(source_batch, target_batch)
        
        # 개별 처리
        metaphors_individual = []
        for i in range(batch_size):
            source_single = source_batch[i:i+1]
            target_single = target_batch[i:i+1]
            metaphor_single, _, _ = seed(source_single, target_single)
            metaphors_individual.append(metaphor_single)
        
        metaphors_individual = torch.cat(metaphors_individual, dim=0)
        
        # 배치와 개별 처리 결과가 유사해야 함
        assert torch.allclose(metaphor_batch, metaphors_individual, atol=1e-5)
    
    def test_different_sequence_lengths(self, seed):
        """다양한 시퀀스 길이 테스트"""
        batch_size = 2
        input_dim = 128
        
        # 다양한 길이
        lengths = [(5, 7), (10, 15), (20, 25)]
        
        for source_len, target_len in lengths:
            source = torch.randn(batch_size, source_len, input_dim)
            target = torch.randn(batch_size, target_len, input_dim)
            
            # Forward pass
            metaphor, mapping_score, structural_similarity = seed(source, target)
            
            # 출력 형태 확인
            assert metaphor.shape == (batch_size, input_dim)
            assert mapping_score.shape == (batch_size,)
            assert structural_similarity.shape == (batch_size,)
    
    def test_metadata_completeness(self, seed):
        """메타데이터 완전성 테스트"""
        metadata = seed.get_metadata()
        
        # 필수 필드 확인
        required_fields = [
            'seed_id', 'name', 'level', 'category',
            'composed_from', 'input_shape', 'output_shape',
            'parameters', 'hidden_dim', 'num_heads'
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        # 값 확인
        assert metadata['seed_id'] == 'SEED-C01'
        assert metadata['level'] == 2
        assert metadata['category'] == 'Analogy'
        assert len(metadata['composed_from']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
