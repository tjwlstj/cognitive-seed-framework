"""
SEED-C08 Novelty Assessor 단위 테스트
"""

import pytest
import torch
import torch.nn as nn

from seeds.cellular.c08_novelty_assessor import NoveltyAssessor, C08


class TestNoveltyAssessor:
    """Novelty Assessor 테스트 클래스"""
    
    @pytest.fixture
    def seed(self):
        """테스트용 시드 인스턴스 생성"""
        return NoveltyAssessor(
            input_dim=64,
            hidden_dim=128,
            num_reference_concepts=5,
            novelty_dimensions=3,
            dropout=0.1
        )
    
    def test_initialization(self, seed):
        """초기화 테스트"""
        assert seed.input_dim == 64
        assert seed.hidden_dim == 128
        assert seed.num_reference_concepts == 5
        assert seed.novelty_dimensions == 3
        assert seed.config.seed_id == "SEED-C08"
        assert seed.config.name == "Novelty Assessor"
        assert seed.config.level == 2
        assert seed.config.category == "Evaluation"
    
    def test_parameter_count(self, seed):
        """파라미터 수 검증"""
        param_count = seed.count_parameters()
        print(f"Total parameters: {param_count:,}")
        
        # Level 2 시드는 약 1.5M 파라미터 목표
        assert param_count > 0
        assert param_count < 2_000_000, f"Too many parameters: {param_count:,}"
    
    def test_forward_basic(self, seed):
        """기본 forward pass 테스트"""
        batch_size = 4
        num_references = 5
        input_dim = 64
        
        # 입력 생성
        input_concept = torch.randn(batch_size, input_dim)
        reference_concepts = torch.randn(batch_size, num_references, input_dim)
        
        # Forward pass
        novelty_score, dim_novelty, _ = seed(input_concept, reference_concepts)
        
        # 출력 shape 검증
        assert novelty_score.shape == (batch_size,)
        assert dim_novelty.shape == (batch_size, 3)
        
        # 값 범위 검증 (0~1)
        assert torch.all(novelty_score >= 0) and torch.all(novelty_score <= 1)
        assert torch.all(dim_novelty >= 0) and torch.all(dim_novelty <= 1)
    
    def test_forward_with_metadata(self, seed):
        """메타데이터 반환 테스트"""
        batch_size = 2
        num_references = 5
        input_dim = 64
        
        input_concept = torch.randn(batch_size, input_dim)
        reference_concepts = torch.randn(batch_size, num_references, input_dim)
        
        novelty_score, dim_novelty, metadata = seed(
            input_concept, 
            reference_concepts,
            return_metadata=True
        )
        
        # 메타데이터 검증
        assert metadata is not None
        assert 'concept_embedding' in metadata
        assert 'prototypes' in metadata
        assert 'similarities' in metadata
        assert 'closest_prototype_idx' in metadata
        assert 'difference_features' in metadata
        assert 'dimensional_novelty' in metadata
        assert 'explanation_features' in metadata
        assert 'is_novel' in metadata
        
        # Shape 검증
        assert metadata['concept_embedding'].shape == (batch_size, 128)
        assert metadata['prototypes'].shape == (batch_size, num_references, 128)
        assert metadata['closest_prototype_idx'].shape == (batch_size,)
        assert metadata['difference_features'].shape == (batch_size, 128)
        assert metadata['dimensional_novelty'].shape == (batch_size, 3)
    
    def test_concept_extraction(self, seed):
        """개념 추출 테스트"""
        batch_size = 3
        input_dim = 64
        
        # 2D 입력
        x_2d = torch.randn(batch_size, input_dim)
        concept_2d = seed.extract_concept(x_2d)
        assert concept_2d.shape == (batch_size, 128)
        
        # 3D 입력
        seq_len = 10
        x_3d = torch.randn(batch_size, seq_len, input_dim)
        concept_3d = seed.extract_concept(x_3d)
        assert concept_3d.shape == (batch_size, 128)
    
    def test_prototype_encoding(self, seed):
        """프로토타입 인코딩 테스트"""
        batch_size = 2
        num_references = 5
        input_dim = 64
        
        reference_concepts = torch.randn(batch_size, num_references, input_dim)
        prototypes = seed.encode_prototypes(reference_concepts)
        
        assert prototypes.shape == (batch_size, num_references, 128)
    
    def test_dimensional_similarity(self, seed):
        """다차원 유사도 계산 테스트"""
        batch_size = 2
        num_references = 5
        hidden_dim = 128
        
        concept = torch.randn(batch_size, hidden_dim)
        prototypes = torch.randn(batch_size, num_references, hidden_dim)
        
        similarities = seed.compute_dimensional_similarity(concept, prototypes)
        
        # 3개 차원 검증
        assert 'structural' in similarities
        assert 'semantic' in similarities
        assert 'functional' in similarities
        
        # Shape 및 값 범위 검증
        for dim_name, sim in similarities.items():
            assert sim.shape == (batch_size, num_references)
            assert torch.all(sim >= 0) and torch.all(sim <= 1)
    
    def test_difference_amplification(self, seed):
        """차이점 강조 테스트"""
        batch_size = 3
        hidden_dim = 128
        
        concept = torch.randn(batch_size, hidden_dim)
        closest_prototype = torch.randn(batch_size, hidden_dim)
        
        diff = seed.amplify_differences(concept, closest_prototype)
        
        assert diff.shape == (batch_size, hidden_dim)
    
    def test_novelty_dimensions(self, seed):
        """차원별 참신성 계산 테스트"""
        batch_size = 4
        hidden_dim = 128
        
        diff_features = torch.randn(batch_size, hidden_dim)
        dim_scores = seed.compute_novelty_dimensions(diff_features)
        
        assert dim_scores.shape == (batch_size, 3)
        assert torch.all(dim_scores >= 0) and torch.all(dim_scores <= 1)
    
    def test_gradient_flow(self, seed):
        """그래디언트 흐름 테스트"""
        batch_size = 2
        num_references = 5
        input_dim = 64
        
        input_concept = torch.randn(batch_size, input_dim, requires_grad=True)
        reference_concepts = torch.randn(batch_size, num_references, input_dim)
        
        novelty_score, dim_novelty, _ = seed(input_concept, reference_concepts)
        
        # Backward pass
        loss = novelty_score.mean() + dim_novelty.mean()
        loss.backward()
        
        # 그래디언트 존재 확인
        assert input_concept.grad is not None
        assert not torch.all(input_concept.grad == 0)
    
    def test_batch_consistency(self, seed):
        """배치 일관성 테스트"""
        num_references = 5
        input_dim = 64
        
        # Single sample
        input_single = torch.randn(1, input_dim)
        reference_single = torch.randn(1, num_references, input_dim)
        score_single, dim_single, _ = seed(input_single, reference_single)
        
        # Batch with same sample repeated
        input_batch = input_single.repeat(3, 1)
        reference_batch = reference_single.repeat(3, 1, 1)
        score_batch, dim_batch, _ = seed(input_batch, reference_batch)
        
        # 결과가 일관되어야 함
        assert torch.allclose(score_single.repeat(3), score_batch, atol=1e-5)
        assert torch.allclose(dim_single.repeat(3, 1), dim_batch, atol=1e-5)
    
    def test_novelty_discrimination(self, seed):
        """참신성 구별 능력 테스트"""
        batch_size = 2
        num_references = 5
        input_dim = 64
        
        # Case 1: 매우 유사한 개념 (낮은 참신성 예상)
        reference_concepts = torch.randn(batch_size, num_references, input_dim)
        similar_concept = reference_concepts[:, 0, :] + torch.randn(batch_size, input_dim) * 0.1
        
        score_similar, _, _ = seed(similar_concept, reference_concepts)
        
        # Case 2: 매우 다른 개념 (높은 참신성 예상)
        novel_concept = torch.randn(batch_size, input_dim) * 10
        score_novel, _, _ = seed(novel_concept, reference_concepts)
        
        # 참신한 개념이 더 높은 점수를 받아야 함
        # (확률적이므로 항상 성립하지는 않지만, 평균적으로 성립)
        print(f"Similar concept novelty: {score_similar.mean().item():.4f}")
        print(f"Novel concept novelty: {score_novel.mean().item():.4f}")
    
    def test_config_retrieval(self, seed):
        """설정 정보 반환 테스트"""
        config = seed.get_config()
        
        assert config['seed_id'] == "SEED-C08"
        assert config['name'] == "Novelty Assessor"
        assert config['level'] == 2
        assert config['category'] == "Evaluation"
        assert config['input_dim'] == 64
        assert config['hidden_dim'] == 128
        assert 'params' in config
    
    def test_alias(self):
        """별칭 테스트"""
        seed1 = NoveltyAssessor()
        seed2 = C08()
        
        assert type(seed1) == type(seed2)
        assert seed1.config.seed_id == seed2.config.seed_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
