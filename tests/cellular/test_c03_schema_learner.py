"""
SEED-C03 Schema Learner 단위 테스트
"""

import pytest
import torch
import torch.nn as nn

from seeds.cellular.c03_schema_learner import SchemaLearner, create_schema_learner


class TestSchemaLearner:
    """Schema Learner 테스트 클래스"""
    
    @pytest.fixture
    def schema_learner(self):
        """Schema Learner 인스턴스 생성"""
        return create_schema_learner(
            input_dim=128,
            hidden_dim=200,
            num_schema_slots=8,
            num_levels=4,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_patterns(self):
        """샘플 패턴 데이터 생성"""
        batch_size = 4
        num_patterns = 16
        input_dim = 128
        return torch.randn(batch_size, num_patterns, input_dim)
    
    @pytest.fixture
    def sample_context(self):
        """샘플 맥락 데이터 생성"""
        batch_size = 4
        context_len = 8
        input_dim = 128
        return torch.randn(batch_size, context_len, input_dim)
    
    def test_initialization(self, schema_learner):
        """초기화 테스트"""
        assert isinstance(schema_learner, SchemaLearner)
        assert schema_learner.input_dim == 128
        assert schema_learner.hidden_dim == 200
        assert schema_learner.num_schema_slots == 8
        assert schema_learner.num_levels == 4
        
        # 설정 확인
        config = schema_learner.get_config()
        assert config['seed_id'] == 'SEED-C03'
        assert config['name'] == 'Schema Learner'
        assert config['level'] == 2
        assert config['category'] == 'Abstraction'
    
    def test_forward_basic(self, schema_learner, sample_patterns):
        """기본 forward 테스트"""
        schema, metadata = schema_learner(sample_patterns, return_metadata=False)
        
        # 출력 shape 확인
        assert schema.shape == (4, 128)
        assert metadata is None
    
    def test_forward_with_metadata(self, schema_learner, sample_patterns):
        """메타데이터 포함 forward 테스트"""
        schema, metadata = schema_learner(sample_patterns, return_metadata=True)
        
        # 출력 shape 확인
        assert schema.shape == (4, 128)
        assert metadata is not None
        
        # 메타데이터 확인
        assert 'concepts' in metadata
        assert 'hierarchy' in metadata
        assert 'rules' in metadata
        assert 'group_assignments' in metadata
        assert 'pattern_features' in metadata
        
        # 메타데이터 shape 확인
        assert metadata['concepts'].shape == (4, 8, 200)  # [B, num_schema_slots, hidden_dim]
        assert metadata['hierarchy'].shape == (4, 4, 200)  # [B, num_levels, hidden_dim]
        assert metadata['rules'].shape == (4, 8, 8)  # [B, num_schema_slots, num_schema_slots]
        assert metadata['group_assignments'].shape == (4, 16, 8)  # [B, N, num_schema_slots]
    
    def test_forward_with_context(self, schema_learner, sample_patterns, sample_context):
        """맥락 정보 포함 forward 테스트"""
        schema, metadata = schema_learner(
            sample_patterns,
            context=sample_context,
            return_metadata=True
        )
        
        # 출력 shape 확인
        assert schema.shape == (4, 128)
        assert metadata is not None
    
    def test_concept_extraction(self, schema_learner, sample_patterns):
        """개념 추출 테스트"""
        schema, metadata = schema_learner(sample_patterns, return_metadata=True)
        
        concepts = metadata['concepts']
        
        # Shape 확인
        assert concepts.shape == (4, 8, 200)
        
        # 개념이 유효한 값인지 확인
        assert not torch.isnan(concepts).any()
        assert not torch.isinf(concepts).any()
    
    def test_hierarchy_construction(self, schema_learner, sample_patterns):
        """계층 구조 구축 테스트"""
        schema, metadata = schema_learner(sample_patterns, return_metadata=True)
        
        hierarchy = metadata['hierarchy']
        
        # Shape 확인
        assert hierarchy.shape == (4, 4, 200)
        
        # 계층이 유효한 값인지 확인
        assert not torch.isnan(hierarchy).any()
        assert not torch.isinf(hierarchy).any()
    
    def test_rule_extraction(self, schema_learner, sample_patterns):
        """규칙 추출 테스트"""
        schema, metadata = schema_learner(sample_patterns, return_metadata=True)
        
        rules = metadata['rules']
        
        # Shape 확인
        assert rules.shape == (4, 8, 8)
        
        # 규칙이 0-1 범위인지 확인 (Sigmoid 출력)
        assert (rules >= 0).all()
        assert (rules <= 1).all()
        
        # 대각선이 0인지 확인 (자기 자신과의 관계 없음)
        for b in range(4):
            assert (torch.diag(rules[b]) == 0).all()
    
    def test_group_assignments(self, schema_learner, sample_patterns):
        """그룹 할당 테스트"""
        schema, metadata = schema_learner(sample_patterns, return_metadata=True)
        
        group_assignments = metadata['group_assignments']
        
        # Shape 확인
        assert group_assignments.shape == (4, 16, 8)
        
        # Softmax 출력 확인 (합이 1)
        assignment_sums = group_assignments.sum(dim=-1)
        assert torch.allclose(assignment_sums, torch.ones_like(assignment_sums), atol=1e-5)
    
    def test_schema_generation(self, schema_learner, sample_patterns):
        """스키마 생성 테스트"""
        schema, _ = schema_learner(sample_patterns, return_metadata=False)
        
        # Shape 확인
        assert schema.shape == (4, 128)
        
        # 스키마가 유효한 값인지 확인
        assert not torch.isnan(schema).any()
        assert not torch.isinf(schema).any()
    
    def test_visualize_schema(self, schema_learner, sample_patterns):
        """스키마 시각화 테스트"""
        vis_data = schema_learner.visualize_schema(sample_patterns)
        
        # 시각화 데이터 확인
        assert 'schema' in vis_data
        assert 'concepts' in vis_data
        assert 'hierarchy' in vis_data
        assert 'rules' in vis_data
        assert 'group_assignments' in vis_data
        
        # Shape 확인
        assert vis_data['schema'].shape == (4, 128)
        assert vis_data['concepts'].shape == (4, 8, 200)
        assert vis_data['hierarchy'].shape == (4, 4, 200)
        assert vis_data['rules'].shape == (4, 8, 8)
    
    def test_parameter_count(self, schema_learner):
        """파라미터 수 테스트"""
        param_count = schema_learner.count_parameters()
        
        # 목표 파라미터 수: ~1.5M (±10%)
        target_params = 1_500_000
        lower_bound = target_params * 0.9
        upper_bound = target_params * 1.1
        
        assert lower_bound <= param_count <= upper_bound, \
            f"Parameter count {param_count} is outside target range [{lower_bound}, {upper_bound}]"
    
    def test_different_input_sizes(self, schema_learner):
        """다양한 입력 크기 테스트"""
        # 다양한 패턴 수
        for num_patterns in [8, 16, 32]:
            patterns = torch.randn(2, num_patterns, 128)
            schema, metadata = schema_learner(patterns, return_metadata=True)
            
            assert schema.shape == (2, 128)
            assert metadata['group_assignments'].shape == (2, num_patterns, 8)
    
    def test_gradient_flow(self, schema_learner, sample_patterns):
        """그래디언트 흐름 테스트"""
        schema, _ = schema_learner(sample_patterns, return_metadata=False)
        
        # 손실 계산
        loss = schema.sum()
        loss.backward()
        
        # 그래디언트 확인
        has_grad = False
        for param in schema_learner.parameters():
            if param.grad is not None:
                has_grad = True
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()
        
        assert has_grad, "No gradients computed"
    
    def test_deterministic_output(self, schema_learner, sample_patterns):
        """결정론적 출력 테스트"""
        # 평가 모드
        schema_learner.eval()
        
        with torch.no_grad():
            schema1, _ = schema_learner(sample_patterns, return_metadata=False)
            schema2, _ = schema_learner(sample_patterns, return_metadata=False)
        
        # 동일한 입력에 대해 동일한 출력
        assert torch.allclose(schema1, schema2, atol=1e-6)
    
    def test_batch_independence(self, schema_learner):
        """배치 독립성 테스트"""
        # 개별 처리
        pattern1 = torch.randn(1, 16, 128)
        pattern2 = torch.randn(1, 16, 128)
        
        schema_learner.eval()
        with torch.no_grad():
            schema1, _ = schema_learner(pattern1, return_metadata=False)
            schema2, _ = schema_learner(pattern2, return_metadata=False)
        
        # 배치 처리
        patterns_batch = torch.cat([pattern1, pattern2], dim=0)
        with torch.no_grad():
            schemas_batch, _ = schema_learner(patterns_batch, return_metadata=False)
        
        # 결과 비교
        assert torch.allclose(schemas_batch[0], schema1[0], atol=1e-5)
        assert torch.allclose(schemas_batch[1], schema2[0], atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
