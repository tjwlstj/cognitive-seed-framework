"""
Unit tests for SEED-M07: Analogy Mapper
"""

import pytest
import torch
import torch.nn as nn
from seeds.molecular.m07_analogy_mapper import AnalogyMapper, create_analogy_mapper


class TestAnalogyMapper:
    """Test suite for Analogy Mapper"""
    
    @pytest.fixture
    def mapper(self):
        """Create a test Analogy Mapper instance"""
        return AnalogyMapper(
            input_dim=128,
            hidden_dim=192,
            num_mapping_layers=2,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample input data"""
        batch_size = 4
        n_source = 10
        n_target = 12
        input_dim = 128
        
        source_structure = torch.randn(batch_size, n_source, input_dim)
        target_structure = torch.randn(batch_size, n_target, input_dim)
        
        return source_structure, target_structure
    
    def test_initialization(self, mapper):
        """Test 1: 모델 초기화 검증"""
        assert isinstance(mapper, AnalogyMapper)
        assert isinstance(mapper, nn.Module)
        assert mapper.input_dim == 128
        assert mapper.hidden_dim == 192
        assert mapper.num_mapping_layers == 2
        
        # Check components exist
        assert hasattr(mapper, 'structure_encoder')
        assert hasattr(mapper, 'concept_matcher')
        assert hasattr(mapper, 'similarity_scorer')
        assert hasattr(mapper, 'mapping_generator')
        assert hasattr(mapper, 'alignment_attention')
        assert hasattr(mapper, 'confidence_estimator')
    
    def test_forward_output_shape(self, mapper, sample_data):
        """Test 2: Forward pass 출력 형상 검증"""
        source_structure, target_structure = sample_data
        batch_size, n_source, input_dim = source_structure.shape
        
        with torch.no_grad():
            output = mapper(source_structure, target_structure)
        
        # Check output is a dictionary
        assert isinstance(output, dict)
        
        # Check required keys
        required_keys = ['mapping', 'similarity_score', 'confidence', 'match_weights']
        for key in required_keys:
            assert key in output, f"Missing key: {key}"
        
        # Check shapes
        assert output['mapping'].shape == (batch_size, n_source, input_dim)
        assert output['similarity_score'].shape == (batch_size,)
        assert output['confidence'].shape == (batch_size,)
        assert output['match_weights'].shape[0] == batch_size
        assert output['match_weights'].shape[1] == n_source
    
    def test_similarity_score_range(self, mapper, sample_data):
        """Test 3: 유사도 점수 범위 검증 (0~1)"""
        source_structure, target_structure = sample_data
        
        with torch.no_grad():
            output = mapper(source_structure, target_structure)
        
        similarity_score = output['similarity_score']
        
        # Similarity scores should be in [0, 1]
        assert torch.all(similarity_score >= 0.0)
        assert torch.all(similarity_score <= 1.0)
        
        # Confidence should also be in [0, 1]
        confidence = output['confidence']
        assert torch.all(confidence >= 0.0)
        assert torch.all(confidence <= 1.0)
    
    def test_match_weights_properties(self, mapper, sample_data):
        """Test 4: 매칭 가중치 속성 검증"""
        source_structure, target_structure = sample_data
        batch_size, n_source, _ = source_structure.shape
        _, n_target, _ = target_structure.shape
        
        with torch.no_grad():
            output = mapper(source_structure, target_structure)
        
        match_weights = output['match_weights']
        
        # Check shape
        assert match_weights.shape == (batch_size, n_source, n_target)
        
        # Check weights are non-negative
        assert torch.all(match_weights >= 0.0)
        
        # Check weights sum to approximately 1 along target dimension
        weight_sums = match_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    
    def test_gradient_flow(self, mapper, sample_data):
        """Test 5: 그래디언트 흐름 검증"""
        source_structure, target_structure = sample_data
        
        # Enable gradient computation
        source_structure.requires_grad = True
        target_structure.requires_grad = True
        
        output = mapper(source_structure, target_structure)
        
        # Compute loss (sum of all outputs)
        loss = (
            output['mapping'].sum() +
            output['similarity_score'].sum() +
            output['confidence'].sum()
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert source_structure.grad is not None
        assert target_structure.grad is not None
        
        # Check gradients are not all zeros
        assert not torch.all(source_structure.grad == 0)
        assert not torch.all(target_structure.grad == 0)
    
    def test_parameter_count(self, mapper):
        """Test 6: 파라미터 수 검증 (~750K 목표)"""
        total_params = sum(p.numel() for p in mapper.parameters())
        
        # Target is ~750K, allow ±20% tolerance
        target_params = 750000
        lower_bound = target_params * 0.8
        upper_bound = target_params * 1.2
        
        assert lower_bound <= total_params <= upper_bound, \
            f"Parameter count {total_params} outside target range [{lower_bound}, {upper_bound}]"
    
    def test_metadata(self, mapper):
        """Test 7: 메타데이터 검증"""
        metadata = mapper.get_metadata()
        
        # Check required fields
        assert metadata['seed_id'] == 'SEED-M07'
        assert metadata['name'] == 'Analogy Mapper'
        assert metadata['level'] == 1
        assert metadata['category'] == 'Analogy'
        assert metadata['bit_depth'] == 'FP8'
        assert metadata['input_dim'] == 128
        assert metadata['output_dim'] == 128
        assert metadata['hidden_dim'] == 192
        assert metadata['composed_from'] == ['M01', 'A08', 'M05']
        
        # Check parameter count is reported
        assert 'actual_params' in metadata
        assert metadata['actual_params'] > 0
    
    def test_different_sequence_lengths(self, mapper):
        """Test 8: 다양한 시퀀스 길이 처리"""
        batch_size = 2
        input_dim = 128
        
        # Test with different source and target lengths
        test_cases = [
            (5, 5),    # Same length
            (10, 5),   # Source longer
            (5, 10),   # Target longer
            (1, 10),   # Single source
            (10, 1),   # Single target
        ]
        
        for n_source, n_target in test_cases:
            source = torch.randn(batch_size, n_source, input_dim)
            target = torch.randn(batch_size, n_target, input_dim)
            
            with torch.no_grad():
                output = mapper(source, target)
            
            # Check output shapes are correct
            assert output['mapping'].shape == (batch_size, n_source, input_dim)
            assert output['match_weights'].shape == (batch_size, n_source, n_target)
    
    def test_batch_independence(self, mapper):
        """Test 9: 배치 독립성 검증"""
        mapper.eval()  # Disable dropout for deterministic behavior
        
        batch_size = 4
        n_source = 8
        n_target = 10
        input_dim = 128
        
        # Create batch data
        source_batch = torch.randn(batch_size, n_source, input_dim)
        target_batch = torch.randn(batch_size, n_target, input_dim)
        
        with torch.no_grad():
            # Process as batch
            batch_output = mapper(source_batch, target_batch)
            
            # Process individually
            individual_outputs = []
            for i in range(batch_size):
                source_single = source_batch[i:i+1]
                target_single = target_batch[i:i+1]
                output = mapper(source_single, target_single)
                individual_outputs.append(output)
        
        # Compare batch vs individual processing
        for i in range(batch_size):
            batch_mapping = batch_output['mapping'][i]
            individual_mapping = individual_outputs[i]['mapping'][0]
            
            # Should be approximately equal
            assert torch.allclose(batch_mapping, individual_mapping, atol=1e-5)
    
    def test_create_helper_function(self):
        """Test 10: 헬퍼 함수 검증"""
        mapper = create_analogy_mapper(
            input_dim=64,
            hidden_dim=128,
            num_mapping_layers=2
        )
        
        assert isinstance(mapper, AnalogyMapper)
        assert mapper.input_dim == 64
        assert mapper.hidden_dim == 128
        assert mapper.num_mapping_layers == 2
    
    def test_deterministic_output(self, mapper):
        """Test 11: 결정론적 출력 검증 (eval 모드)"""
        mapper.eval()
        
        source = torch.randn(2, 5, 128)
        target = torch.randn(2, 7, 128)
        
        with torch.no_grad():
            output1 = mapper(source, target)
            output2 = mapper(source, target)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1['mapping'], output2['mapping'])
        assert torch.allclose(output1['similarity_score'], output2['similarity_score'])
        assert torch.allclose(output1['confidence'], output2['confidence'])
    
    def test_structural_similarity_computation(self, mapper):
        """Test 12: 구조적 유사도 계산 검증"""
        # Create identical structures
        batch_size = 2
        n_nodes = 5
        input_dim = 128
        
        identical_structure = torch.randn(batch_size, n_nodes, input_dim)
        
        with torch.no_grad():
            output = mapper(identical_structure, identical_structure)
        
        # Similarity should be reasonably high for identical structures
        similarity_score = output['similarity_score']
        assert torch.all(similarity_score > 0.3), \
            "Similarity score should be reasonably high for identical structures"
        
        # Confidence should also be reasonably high
        confidence = output['confidence']
        assert torch.all(confidence > 0.3), \
            "Confidence should be reasonably high for identical structures"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
