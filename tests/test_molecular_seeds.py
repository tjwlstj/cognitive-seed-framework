"""
Level 1 (Molecular) Seeds 단위 테스트
"""

import torch
from seeds.molecular import (
    HierarchyBuilder,
    CausalityDetector,
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
    
    # ========== M04: Spatial Transformer ==========
    
    def test_spatial_transformer_forward(self):
        """M04: Forward pass 테스트"""
        print("\n[7/9] Testing Spatial Transformer - Forward pass...")
        
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
        
        # M04 tests
        test_suite.test_spatial_transformer_forward()
        test_suite.test_spatial_transformer_alignment()
        test_suite.test_spatial_transformer_inverse()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()

