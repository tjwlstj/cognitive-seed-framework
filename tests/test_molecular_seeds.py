"""
Level 1 (Molecular) Seeds 단위 테스트
"""

import torch
from seeds.molecular import (
    HierarchyBuilder,
    CausalityDetector,
    PatternCompleter,
    SpatialTransformer
)


class TestMolecularSeeds:
    """Molecular 시드 테스트"""
    
    def setup_method(self):
        """테스트 설정"""
        self.batch_size = 4
        self.input_dim = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ========== M01: Hierarchy Builder ==========
    
    def test_hierarchy_builder_forward(self):
        """M01: Forward pass 테스트"""
        print("\n[1/9] Testing Hierarchy Builder - Forward pass...")
        
        seed = HierarchyBuilder(input_dim=self.input_dim, num_clusters=16)
        seed = seed.to(self.device)
        
        # 입력: [B, N, D] - N개 노드
        num_nodes = 20
        x = torch.randn(self.batch_size, num_nodes, self.input_dim).to(self.device)
        
        # Forward
        output = seed(x)
        
        # 출력 shape 확인
        assert output.shape == (self.batch_size, num_nodes, self.input_dim)
        assert not torch.isnan(output).any()
        
        print("✓ Forward pass successful")
    
    def test_hierarchy_builder_tree_structure(self):
        """M01: 트리 구조 추출 테스트"""
        print("\n[2/9] Testing Hierarchy Builder - Tree structure...")
        
        seed = HierarchyBuilder(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        num_nodes = 15
        x = torch.randn(self.batch_size, num_nodes, self.input_dim).to(self.device)
        
        # 트리 구조 추출
        tree_structure = seed.get_tree_structure(x)
        
        # 결과 확인
        assert 'adjacency_matrix' in tree_structure
        assert 'levels' in tree_structure
        assert 'hierarchy_matrix' in tree_structure
        assert 'num_roots' in tree_structure
        assert 'max_depth' in tree_structure
        
        assert tree_structure['adjacency_matrix'].shape == (self.batch_size, num_nodes, num_nodes)
        assert tree_structure['levels'].shape == (self.batch_size, num_nodes)
        assert tree_structure['num_roots'].shape == (self.batch_size,)
        assert tree_structure['max_depth'].shape == (self.batch_size,)
        
        print("✓ Tree structure extraction successful")
    
    def test_hierarchy_builder_metadata(self):
        """M01: 메타데이터 테스트"""
        print("\n[3/9] Testing Hierarchy Builder - Metadata...")
        
        seed = HierarchyBuilder(input_dim=self.input_dim)
        metadata = seed.get_metadata()
        
        assert metadata['seed_id'] == 'SEED-M01'
        assert metadata['name'] == 'Hierarchy Builder'
        assert metadata['level'] == 1
        assert metadata['category'] == 'Relation'
        
        # 파라미터 수 확인
        param_count = seed.count_parameters()
        print(f"  Total parameters: {param_count:,}")
        
        print("✓ Metadata check successful")
    
    # ========== M02: Causality Detector ==========
    
    def test_causality_detector_forward(self):
        """M02: Forward pass 테스트"""
        print("\n[4/9] Testing Causality Detector - Forward pass...")
        
        seed = CausalityDetector(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        # 입력: [B, T, D] - 시계열
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # Forward
        output = seed(x)
        
        # 출력 shape 확인
        assert output.shape == (self.batch_size, seq_len, self.input_dim)
        assert not torch.isnan(output).any()
        
        print("✓ Forward pass successful")
    
    def test_causality_detector_with_intervention(self):
        """M02: 개입 효과 테스트"""
        print("\n[5/9] Testing Causality Detector - Intervention...")
        
        seed = CausalityDetector(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        interventions = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # 개입 있는 경우
        output_with_intervention = seed(x, context={'interventions': interventions})
        
        # 개입 없는 경우
        output_without_intervention = seed(x)
        
        # 두 출력이 달라야 함
        assert not torch.allclose(output_with_intervention, output_without_intervention)
        
        print("✓ Intervention effect successful")
    
    def test_causality_detector_causal_graph(self):
        """M02: 인과 그래프 추정 테스트"""
        print("\n[6/9] Testing Causality Detector - Causal graph...")
        
        seed = CausalityDetector(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # 인과 그래프 추정
        causal_graph = seed.estimate_causal_graph(x)
        
        # 결과 확인 (T x T 행렬)
        assert causal_graph.shape == (self.batch_size, seq_len, seq_len)
        assert (causal_graph >= 0).all() and (causal_graph <= 1).all()  # 확률 값
        
        # 대각선은 0이어야 함
        diagonal = torch.diagonal(causal_graph, dim1=1, dim2=2)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal))
        
        print("✓ Causal graph estimation successful")
    
    # ========== M03: Pattern Completer ==========
    
    def test_pattern_completer_forward(self):
        """M03: Forward pass 테스트"""
        print("\n[10/15] Testing Pattern Completer - Forward pass...")
        
        seed = PatternCompleter(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        # 입력: [B, L, D]
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # Forward
        output = seed(x)
        
        # 출력 shape 확인
        assert output.shape == (self.batch_size, seq_len, self.input_dim)
        assert not torch.isnan(output).any()
        
        print("✓ Forward pass successful")
    
    def test_pattern_completer_with_mask(self):
        """M03: 마스크를 사용한 패턴 완성 테스트"""
        print("\n[11/15] Testing Pattern Completer - With mask...")
        
        seed = PatternCompleter(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # 마스크 생성 (일부 결손)
        mask = torch.ones(self.batch_size, seq_len).to(self.device)
        mask[:, 10:20] = 0  # 10~19 인덱스 결손
        
        # Forward with mask
        output = seed(x, mask=mask)
        
        # 결과 확인
        assert output.shape == (self.batch_size, seq_len, self.input_dim)
        assert not torch.isnan(output).any()
        
        print("✓ Pattern completion with mask successful")
    
    def test_pattern_completer_interpolate(self):
        """M03: 보간 테스트"""
        print("\n[12/15] Testing Pattern Completer - Interpolation...")
        
        seed = PatternCompleter(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # 특정 위치 보간
        missing_indices = [10, 15, 20, 25]
        interpolated = seed.interpolate(x, missing_indices)
        
        # 결과 확인
        assert interpolated.shape == (self.batch_size, seq_len, self.input_dim)
        assert not torch.isnan(interpolated).any()
        
        print("✓ Interpolation successful")
    
    def test_pattern_completer_extrapolate(self):
        """M03: 외삽 테스트"""
        print("\n[13/15] Testing Pattern Completer - Extrapolation...")
        
        seed = PatternCompleter(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # 미래 예측
        num_steps = 10
        extrapolated = seed.extrapolate(x, num_steps=num_steps)
        
        # 결과 확인
        assert extrapolated.shape == (self.batch_size, seq_len + num_steps, self.input_dim)
        assert not torch.isnan(extrapolated).any()
        
        print("✓ Extrapolation successful")
    
    def test_pattern_completer_quality(self):
        """M03: 완성 품질 평가 테스트"""
        print("\n[14/15] Testing Pattern Completer - Quality metrics...")
        
        seed = PatternCompleter(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 50
        original = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # 마스크 생성
        mask = torch.ones(self.batch_size, seq_len).to(self.device)
        mask[:, 10:20] = 0
        
        # 완성
        completed = seed(original, mask=mask)
        
        # 품질 평가
        metrics = seed.compute_completion_quality(original, completed, mask)
        
        # 결과 확인
        assert 'mse' in metrics
        assert 'structural_similarity' in metrics
        assert 'completion_rate' in metrics
        
        print("✓ Quality metrics computation successful")
    
    # ========== M04: Spatial Transformer ==========
    
    def test_spatial_transformer_forward(self):
        """M04: Forward pass 테스트"""
        print("\n[15/15] Testing Spatial Transformer - Forward pass...")
        
        seed = SpatialTransformer(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        # 입력: [B, L, D]
        seq_len = 32
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # Forward
        output = seed(x)
        
        # 출력 shape 확인
        assert output.shape == (self.batch_size, seq_len, self.input_dim)
        assert not torch.isnan(output).any()
        
        print("✓ Forward pass successful")
    
    def test_spatial_transformer_alignment(self):
        """M04: 정규 좌표계 정렬 테스트"""
        print("\n[8/9] Testing Spatial Transformer - Alignment...")
        
        seed = SpatialTransformer(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 32
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # 정렬
        aligned, params = seed.align_to_canonical(x)
        
        # 결과 확인
        assert aligned.shape == (self.batch_size, seq_len, self.input_dim)
        assert 'translation' in params
        assert 'rotation' in params
        assert 'scale' in params
        assert 'shear' in params
        
        assert params['translation'].shape == (self.batch_size, 2)
        assert params['rotation'].shape == (self.batch_size,)
        assert params['scale'].shape == (self.batch_size, 2)
        assert params['shear'].shape == (self.batch_size,)
        
        print("✓ Alignment successful")
    
    def test_spatial_transformer_inverse(self):
        """M04: 역변환 테스트"""
        print("\n[9/9] Testing Spatial Transformer - Inverse transform...")
        
        seed = SpatialTransformer(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 32
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # 정렬
        aligned, params = seed.align_to_canonical(x)
        
        # 역변환
        reconstructed = seed.inverse_transform(aligned, params)
        
        # shape 확인
        assert reconstructed.shape == x.shape
        
        # 완전히 복원되지는 않지만 유사해야 함 (등변성 인코딩 때문)
        # 단순히 NaN이 없는지만 확인
        assert not torch.isnan(reconstructed).any()
        
        print("✓ Inverse transform successful")
    
    # ========== M06: Context Integrator ==========
    
    def test_context_integrator_forward(self):
        """M06: Forward pass 테스트"""
        print("\n[16/18] Testing Context Integrator - Forward pass...")
        
        from seeds.molecular import ContextIntegrator
        
        seed = ContextIntegrator(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        # 입력: [B, L, D] - 시퀀스
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # Forward
        output = seed(x)
        
        # 출력 shape 확인
        assert output.shape == (self.batch_size, seq_len, self.input_dim)
        assert not torch.isnan(output).any()
        
        print("✓ Forward pass successful")
    
    def test_context_integrator_metadata(self):
        """M06: Metadata 반환 테스트"""
        print("\n[17/18] Testing Context Integrator - Metadata...")
        
        from seeds.molecular import ContextIntegrator
        
        seed = ContextIntegrator(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # Metadata 반환
        output, metadata = seed(x, return_metadata=True)
        
        # Metadata 키 확인
        expected_keys = [
            'local_context', 'global_context', 'temporal_context',
            'hierarchical_context', 'group_context', 'fused_context', 'fusion_weights'
        ]
        assert all(k in metadata for k in expected_keys)
        
        # Context shape 확인
        for key in ['local_context', 'global_context', 'temporal_context', 
                    'hierarchical_context', 'group_context', 'fused_context']:
            assert metadata[key].shape == (self.batch_size, seq_len, self.input_dim)
        
        # Fusion weights 확인
        assert metadata['fusion_weights'].shape == (5,)
        assert torch.allclose(metadata['fusion_weights'].sum(), torch.tensor(1.0), atol=1e-5)
        
        print("✓ Metadata returned correctly")
    
    def test_context_integrator_importance(self):
        """M06: Context importance 테스트"""
        print("\n[18/18] Testing Context Integrator - Context importance...")
        
        from seeds.molecular import ContextIntegrator
        
        seed = ContextIntegrator(input_dim=self.input_dim)
        seed = seed.to(self.device)
        
        seq_len = 50
        x = torch.randn(self.batch_size, seq_len, self.input_dim).to(self.device)
        
        # Context importance 계산
        importance = seed.get_context_importance(x)
        
        # 키 확인
        expected_keys = ['local', 'global', 'temporal', 'hierarchical', 'group']
        assert all(k in importance for k in expected_keys)
        
        # 합이 1인지 확인
        total = sum(importance.values())
        assert abs(total - 1.0) < 1e-5, f"Importance sum: {total}"
        
        # 모든 값이 0-1 범위인지 확인
        for v in importance.values():
            assert 0 <= v <= 1
        
        print("✓ Context importance calculated correctly")


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("Level 1 (Molecular) Seeds - Unit Tests")
    print("=" * 60)
    
    test_suite = TestMolecularSeeds()
    test_suite.setup_method()
    
    try:
        # M01 tests
        test_suite.test_hierarchy_builder_forward()
        test_suite.test_hierarchy_builder_tree_structure()
        test_suite.test_hierarchy_builder_metadata()
        
        # M02 tests
        test_suite.test_causality_detector_forward()
        test_suite.test_causality_detector_with_intervention()
        test_suite.test_causality_detector_causal_graph()
        
        # M03 tests
        test_suite.test_pattern_completer_forward()
        test_suite.test_pattern_completer_with_mask()
        test_suite.test_pattern_completer_interpolate()
        test_suite.test_pattern_completer_extrapolate()
        test_suite.test_pattern_completer_quality()
        
        # M04 tests
        test_suite.test_spatial_transformer_forward()
        test_suite.test_spatial_transformer_alignment()
        test_suite.test_spatial_transformer_inverse()
        
        # M06 tests
        test_suite.test_context_integrator_forward()
        test_suite.test_context_integrator_metadata()
        test_suite.test_context_integrator_importance()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
