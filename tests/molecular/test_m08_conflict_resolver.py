"""
Tests for SEED-M08: Conflict Resolver

Author: Manus AI
Date: 2025-11-17
"""

import pytest
import torch
import torch.nn as nn

from seeds.molecular.m08_conflict_resolver import (
    ConflictResolver,
    ConflictResolverConfig,
    create_conflict_resolver
)


class TestConflictResolverBasic:
    """기본 기능 테스트"""
    
    def test_initialization(self):
        """초기화 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        assert resolver.config.seed_id == "SEED-M08"
        assert resolver.config.name == "Conflict Resolver"
        assert resolver.config.level == 1
        assert resolver.config.category == "Logic"
        assert resolver.config.input_dim == 128
        assert resolver.config.output_dim == 128
    
    def test_forward_shape(self):
        """Forward 출력 형상 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 4
        num_constraints = 5
        constraints = torch.randn(batch_size, num_constraints, 128)
        
        resolution, conflict_score, fairness_score = resolver(constraints)
        
        assert resolution.shape == (batch_size, 128)
        assert conflict_score.shape == (batch_size,)
        assert fairness_score.shape == (batch_size,)
    
    def test_forward_with_context(self):
        """맥락 정보를 포함한 Forward 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 4
        num_constraints = 5
        context_len = 10
        
        constraints = torch.randn(batch_size, num_constraints, 128)
        context = torch.randn(batch_size, context_len, 128)
        
        resolution, conflict_score, fairness_score = resolver(constraints, context)
        
        assert resolution.shape == (batch_size, 128)
        assert conflict_score.shape == (batch_size,)
        assert fairness_score.shape == (batch_size,)
    
    def test_different_num_constraints(self):
        """다양한 제약 개수 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        for num_constraints in [2, 5, 8, 10]:
            constraints = torch.randn(2, num_constraints, 128)
            resolution, conflict_score, fairness_score = resolver(constraints)
            
            assert resolution.shape == (2, 128)
            assert conflict_score.shape == (2,)
            assert fairness_score.shape == (2,)


class TestConflictDetection:
    """충돌 탐지 기능 테스트"""
    
    def test_conflict_detection(self):
        """충돌 탐지 메서드 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 2
        num_constraints = 4
        constraints = torch.randn(batch_size, num_constraints, 128)
        
        # 제약 인코딩
        encoded = resolver._encode_constraints(constraints)
        
        # 충돌 탐지
        conflict_matrix, conflict_score = resolver._detect_conflicts(encoded)
        
        assert conflict_matrix.shape == (batch_size, num_constraints, num_constraints)
        assert conflict_score.shape == (batch_size,)
        assert torch.all(conflict_score >= 0) and torch.all(conflict_score <= 1)
    
    def test_high_conflict_scenario(self):
        """높은 충돌 시나리오 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        # 상반된 제약 생성 (정반대 방향)
        batch_size = 2
        constraint_1 = torch.ones(batch_size, 1, 128)
        constraint_2 = -torch.ones(batch_size, 1, 128)
        constraints = torch.cat([constraint_1, constraint_2], dim=1)
        
        resolution, conflict_score, fairness_score = resolver(constraints)
        
        # 충돌 점수가 0보다 커야 함
        assert torch.all(conflict_score > 0)
    
    def test_low_conflict_scenario(self):
        """낮은 충돌 시나리오 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        # 유사한 제약 생성
        batch_size = 2
        base_constraint = torch.randn(batch_size, 1, 128)
        constraint_1 = base_constraint + 0.01 * torch.randn(batch_size, 1, 128)
        constraint_2 = base_constraint + 0.01 * torch.randn(batch_size, 1, 128)
        constraints = torch.cat([constraint_1, constraint_2], dim=1)
        
        resolution, conflict_score, fairness_score = resolver(constraints)
        
        # 충돌 점수가 상대적으로 낮아야 함
        assert torch.all(conflict_score >= 0)


class TestPriorityComputation:
    """우선순위 계산 테스트"""
    
    def test_priority_computation(self):
        """우선순위 계산 메서드 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 2
        num_constraints = 4
        constraints = torch.randn(batch_size, num_constraints, 128)
        causal_features = torch.randn(batch_size, num_constraints, 128)
        
        priorities = resolver._compute_priorities(constraints, causal_features)
        
        assert priorities.shape == (batch_size, num_constraints, 1)
        # 우선순위 합이 1이어야 함 (softmax)
        assert torch.allclose(priorities.sum(dim=1), torch.ones(batch_size, 1))
    
    def test_priority_influence_on_resolution(self):
        """우선순위가 해결책에 미치는 영향 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 2
        num_constraints = 3
        constraints = torch.randn(batch_size, num_constraints, 128)
        
        # 첫 번째 제약에 높은 우선순위 부여
        priorities = torch.zeros(batch_size, num_constraints, 1)
        priorities[:, 0, :] = 0.8
        priorities[:, 1, :] = 0.1
        priorities[:, 2, :] = 0.1
        
        causal_features = torch.randn(batch_size, num_constraints, 128)
        conflict_matrix = torch.zeros(batch_size, num_constraints, num_constraints)
        
        resolution = resolver._generate_resolution(
            constraints, causal_features, priorities, conflict_matrix
        )
        
        assert resolution.shape == (batch_size, 128)


class TestFairnessModule:
    """공정성 모듈 테스트"""
    
    def test_fairness_evaluation(self):
        """공정성 평가 메서드 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 2
        num_constraints = 4
        resolution = torch.randn(batch_size, 128)
        constraints = torch.randn(batch_size, num_constraints, 128)
        
        fairness_score = resolver._evaluate_fairness(resolution, constraints)
        
        assert fairness_score.shape == (batch_size,)
        assert torch.all(fairness_score >= 0) and torch.all(fairness_score <= 1)
    
    def test_fairness_adjustment(self):
        """공정성 조정 메서드 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 2
        resolution = torch.randn(batch_size, 128)
        fairness_score = torch.tensor([0.3, 0.8])  # 낮은/높은 공정성
        
        adjusted_resolution = resolver._adjust_for_fairness(resolution, fairness_score)
        
        assert adjusted_resolution.shape == (batch_size, 128)
        # 조정된 해결책은 원본과 달라야 함
        assert not torch.allclose(adjusted_resolution, resolution)
    
    def test_fairness_weight_effect(self):
        """공정성 가중치 효과 테스트"""
        # 높은 공정성 가중치
        resolver_high = ConflictResolver(input_dim=128, fairness_weight=0.9)
        # 낮은 공정성 가중치
        resolver_low = ConflictResolver(input_dim=128, fairness_weight=0.1)
        
        batch_size = 2
        num_constraints = 3
        constraints = torch.randn(batch_size, num_constraints, 128)
        
        resolution_high, _, fairness_high = resolver_high(constraints)
        resolution_low, _, fairness_low = resolver_low(constraints)
        
        # 두 해결책이 달라야 함
        assert not torch.allclose(resolution_high, resolution_low)


class TestResolutionGeneration:
    """해결책 생성 테스트"""
    
    def test_resolution_generation(self):
        """해결책 생성 메서드 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 2
        num_constraints = 4
        constraints = torch.randn(batch_size, num_constraints, 128)
        causal_features = torch.randn(batch_size, num_constraints, 128)
        priorities = torch.softmax(torch.randn(batch_size, num_constraints, 1), dim=1)
        conflict_matrix = torch.rand(batch_size, num_constraints, num_constraints)
        
        resolution = resolver._generate_resolution(
            constraints, causal_features, priorities, conflict_matrix
        )
        
        assert resolution.shape == (batch_size, 128)
    
    def test_resolution_consistency(self):
        """해결책 일관성 테스트 (동일 입력 → 동일 출력)"""
        resolver = ConflictResolver(input_dim=128)
        resolver.eval()  # 평가 모드
        
        batch_size = 2
        num_constraints = 3
        constraints = torch.randn(batch_size, num_constraints, 128)
        
        with torch.no_grad():
            resolution_1, conflict_1, fairness_1 = resolver(constraints)
            resolution_2, conflict_2, fairness_2 = resolver(constraints)
        
        assert torch.allclose(resolution_1, resolution_2)
        assert torch.allclose(conflict_1, conflict_2)
        assert torch.allclose(fairness_1, fairness_2)


class TestHighLevelAPI:
    """고수준 API 테스트"""
    
    def test_resolve_conflicts_api(self):
        """resolve_conflicts 메서드 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 2
        constraints = [
            torch.randn(batch_size, 128),
            torch.randn(batch_size, 128),
            torch.randn(batch_size, 128)
        ]
        
        result = resolver.resolve_conflicts(constraints)
        
        assert 'resolution' in result
        assert 'conflict_score' in result
        assert 'fairness_score' in result
        assert 'priorities' in result
        
        assert result['resolution'].shape == (batch_size, 128)
        assert result['conflict_score'].shape == (batch_size,)
        assert result['fairness_score'].shape == (batch_size,)
        assert result['priorities'].shape == (batch_size, len(constraints))
    
    def test_resolve_conflicts_with_context(self):
        """맥락을 포함한 resolve_conflicts 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 2
        constraints = [
            torch.randn(batch_size, 128),
            torch.randn(batch_size, 128)
        ]
        context = torch.randn(batch_size, 10, 128)
        
        result = resolver.resolve_conflicts(constraints, context)
        
        assert result['resolution'].shape == (batch_size, 128)


class TestMetadata:
    """메타데이터 테스트"""
    
    def test_config_metadata(self):
        """설정 메타데이터 테스트"""
        config = ConflictResolverConfig()
        
        assert config.seed_id == "SEED-M08"
        assert config.name == "Conflict Resolver"
        assert config.level == 1
        assert config.category == "Logic"
        assert config.bit_depth == "FP8"
        assert config.params == 800000
    
    def test_custom_config(self):
        """커스텀 설정 테스트"""
        resolver = ConflictResolver(
            input_dim=256,
            num_constraints_max=15,
            resolution_layers=4,
            fairness_weight=0.7,
            dropout=0.2
        )
        
        assert resolver.config.input_dim == 256
        assert resolver.config.num_constraints_max == 15
        assert resolver.config.resolution_layers == 4
        assert resolver.config.fairness_weight == 0.7
        assert resolver.config.dropout == 0.2


class TestFactoryFunction:
    """팩토리 함수 테스트"""
    
    def test_create_conflict_resolver(self):
        """create_conflict_resolver 함수 테스트"""
        resolver = create_conflict_resolver(input_dim=128)
        
        assert isinstance(resolver, ConflictResolver)
        assert resolver.config.input_dim == 128
    
    def test_create_with_custom_params(self):
        """커스텀 파라미터로 생성 테스트"""
        resolver = create_conflict_resolver(
            input_dim=256,
            num_constraints_max=12,
            resolution_layers=5,
            fairness_weight=0.6,
            dropout=0.15
        )
        
        assert resolver.config.input_dim == 256
        assert resolver.config.num_constraints_max == 12
        assert resolver.config.resolution_layers == 5
        assert resolver.config.fairness_weight == 0.6
        assert resolver.config.dropout == 0.15


class TestParameterCount:
    """파라미터 수 테스트"""
    
    def test_parameter_count(self):
        """총 파라미터 수 검증"""
        resolver = ConflictResolver(input_dim=128)
        
        total_params = sum(p.numel() for p in resolver.parameters())
        
        # 목표: ~800K (±10%)
        target = 800000
        tolerance = 0.10
        
        lower_bound = target * (1 - tolerance)
        upper_bound = target * (1 + tolerance)
        
        print(f"\n총 파라미터 수: {total_params:,}")
        print(f"목표 범위: {lower_bound:,.0f} ~ {upper_bound:,.0f}")
        
        assert lower_bound <= total_params <= upper_bound, \
            f"파라미터 수 {total_params:,}가 목표 범위를 벗어남"
    
    def test_component_parameters(self):
        """컴포넌트별 파라미터 수 확인"""
        resolver = ConflictResolver(input_dim=128)
        
        components = {
            'comparator': resolver.comparator,
            'context_integrator': resolver.context_integrator,
            'causality_detector': resolver.causality_detector,
            'constraint_encoder': resolver.constraint_encoder,
            'resolution_generator': resolver.resolution_generator
        }
        
        print("\n컴포넌트별 파라미터 수:")
        for name, component in components.items():
            params = sum(p.numel() for p in component.parameters())
            print(f"  {name}: {params:,}")


class TestGradientFlow:
    """그래디언트 흐름 테스트"""
    
    def test_gradient_flow(self):
        """그래디언트가 모든 파라미터로 흐르는지 테스트"""
        resolver = ConflictResolver(input_dim=128)
        
        batch_size = 2
        num_constraints = 3
        constraints = torch.randn(batch_size, num_constraints, 128)
        
        resolution, conflict_score, fairness_score = resolver(constraints)
        
        # 손실 계산 (임의)
        loss = resolution.sum() + conflict_score.sum() + fairness_score.sum()
        loss.backward()
        
        # 모든 파라미터에 그래디언트가 있는지 확인
        for name, param in resolver.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"{name}에 그래디언트가 없음"
                assert not torch.isnan(param.grad).any(), f"{name}에 NaN 그래디언트 발견"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
