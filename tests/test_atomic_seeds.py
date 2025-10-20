"""
Level 0 (Atomic) Seeds 테스트

8개의 원자 시드를 테스트합니다.
"""

import torch
import pytest
from seeds.atomic import (
    EdgeDetector,
    SymmetryDetector,
    RecurrenceSpotter,
    ContrastAmplifier,
    GroupingNucleus,
    SequenceTracker,
    ScaleNormalizer,
    BinaryComparator
)


class TestAtomicSeeds:
    """Atomic Seeds 테스트 클래스"""
    
    @pytest.fixture
    def batch_size(self):
        return 4
    
    @pytest.fixture
    def seq_len(self):
        return 32
    
    @pytest.fixture
    def input_dim(self):
        return 128
    
    @pytest.fixture
    def sample_input(self, batch_size, seq_len, input_dim):
        """테스트용 샘플 입력 생성"""
        return torch.randn(batch_size, seq_len, input_dim)
    
    @pytest.fixture
    def sample_scale(self, batch_size):
        """테스트용 스케일 파라미터 생성"""
        return torch.ones(batch_size, 1)
    
    def test_edge_detector(self, sample_input, sample_scale):
        """SEED-A01: Edge Detector 테스트"""
        seed = EdgeDetector(input_dim=128, hidden_dim=64)
        
        # Forward pass
        output = seed(sample_input, sample_scale)
        assert output.shape == sample_input.shape
        
        # Edge detection
        edge_mask = seed.detect_edges(sample_input, threshold=0.5)
        assert edge_mask.shape == (sample_input.size(0), sample_input.size(1))
        
        # 파라미터 수 확인
        param_count = seed.count_parameters()
        print(f"Edge Detector parameters: {param_count}")
        assert param_count > 0
    
    def test_symmetry_detector(self, sample_input, sample_scale):
        """SEED-A02: Symmetry Detector 테스트"""
        seed = SymmetryDetector(input_dim=128, hidden_dim=128)
        
        # Forward pass
        output = seed(sample_input, sample_scale)
        assert output.shape == sample_input.shape
        
        # Symmetry type detection
        symmetry_types, symmetry_axis = seed.detect_symmetry_type(sample_input)
        assert symmetry_types.shape == (sample_input.size(0), sample_input.size(1), 3)
        assert symmetry_axis.shape == sample_input.shape
        
        # Symmetry score
        symmetry_score = seed.compute_symmetry_score(sample_input)
        assert symmetry_score.shape == (sample_input.size(0), sample_input.size(1))
        
        # 파라미터 수 확인
        param_count = seed.count_parameters()
        print(f"Symmetry Detector parameters: {param_count}")
    
    def test_recurrence_spotter(self, sample_input, sample_scale):
        """SEED-A03: Recurrence Spotter 테스트"""
        seed = RecurrenceSpotter(input_dim=128, hidden_dim=96)
        
        # Forward pass
        output = seed(sample_input, sample_scale)
        assert output.shape == sample_input.shape
        
        # Period detection
        period = seed.detect_period(sample_input)
        assert period.shape == (sample_input.size(0), sample_input.size(1))
        
        # Recurrence score
        recurrence_score = seed.compute_recurrence_score(sample_input, window_size=5)
        assert recurrence_score.shape == (sample_input.size(0), sample_input.size(1))
        
        # 파라미터 수 확인
        param_count = seed.count_parameters()
        print(f"Recurrence Spotter parameters: {param_count}")
    
    def test_contrast_amplifier(self, sample_input, sample_scale):
        """SEED-A04: Contrast Amplifier 테스트"""
        seed = ContrastAmplifier(input_dim=128, hidden_dim=32)
        
        # Forward pass
        output = seed(sample_input, sample_scale)
        assert output.shape == sample_input.shape
        
        # SNR computation
        snr_improvement = seed.compute_snr(sample_input, output)
        assert snr_improvement.shape == (sample_input.size(0),)
        
        # Denoising
        denoised = seed.denoise(sample_input, threshold=0.1)
        assert denoised.shape == sample_input.shape
        
        # 파라미터 수 확인
        param_count = seed.count_parameters()
        print(f"Contrast Amplifier parameters: {param_count}")
    
    def test_grouping_nucleus(self, sample_input, sample_scale):
        """SEED-A05: Grouping Nucleus 테스트"""
        seed = GroupingNucleus(input_dim=128, hidden_dim=128, num_clusters=8)
        
        # Forward pass
        output = seed(sample_input, sample_scale)
        assert output.shape == sample_input.shape
        
        # Cluster assignments
        assignments = seed.get_cluster_assignments(sample_input)
        assert assignments.shape == (sample_input.size(0), sample_input.size(1), 8)
        
        # Hard assignments
        hard_assignments = seed.get_hard_assignments(sample_input)
        assert hard_assignments.shape == (sample_input.size(0), sample_input.size(1))
        
        # Cluster distances
        distances = seed.compute_cluster_distances(sample_input)
        assert distances.shape == (sample_input.size(0), sample_input.size(1), 8)
        
        # 파라미터 수 확인
        param_count = seed.count_parameters()
        print(f"Grouping Nucleus parameters: {param_count}")
    
    def test_sequence_tracker(self, sample_input, sample_scale):
        """SEED-A06: Sequence Tracker 테스트"""
        seed = SequenceTracker(input_dim=128, hidden_dim=160)
        
        # Forward pass
        output = seed(sample_input, sample_scale)
        assert output.shape == sample_input.shape
        
        # Next state prediction
        predictions = seed.predict_next(sample_input, num_steps=3)
        assert predictions.shape == (sample_input.size(0), 3, sample_input.size(2))
        
        # Tracking accuracy
        accuracy = seed.compute_tracking_accuracy(sample_input)
        assert accuracy.shape == (sample_input.size(0),)
        
        # 파라미터 수 확인
        param_count = seed.count_parameters()
        print(f"Sequence Tracker parameters: {param_count}")
    
    def test_scale_normalizer(self, sample_input, sample_scale):
        """SEED-A07: Scale Normalizer 테스트"""
        seed = ScaleNormalizer(input_dim=128, hidden_dim=64)
        
        # Forward pass
        output = seed(sample_input, sample_scale)
        assert output.shape == sample_input.shape
        
        # Scale estimation
        estimated_scale = seed.estimate_scale(sample_input)
        assert estimated_scale.shape == (sample_input.size(0), sample_input.size(1))
        
        # Normalize to specific scale
        normalized = seed.normalize_to_scale(sample_input, target_scale=1.0)
        assert normalized.shape == sample_input.shape
        
        # Variance stability
        stability = seed.compute_variance_stability(sample_input)
        assert stability.shape == (sample_input.size(0),)
        
        # Overflow/underflow check
        risk_stats = seed.check_overflow_underflow(sample_input)
        assert "max_value" in risk_stats
        assert "min_value" in risk_stats
        
        # 파라미터 수 확인
        param_count = seed.count_parameters()
        print(f"Scale Normalizer parameters: {param_count}")
    
    def test_binary_comparator(self, sample_input, sample_scale):
        """SEED-A08: Binary Comparator 테스트"""
        seed = BinaryComparator(input_dim=128, hidden_dim=48)
        
        # Forward pass
        output = seed(sample_input, sample_scale)
        assert output.shape == sample_input.shape
        
        # Pairwise comparison
        a = sample_input[:, 0, :]  # [B, D]
        b = sample_input[:, 1, :]  # [B, D]
        
        comparison = seed.compare(a, b)
        assert comparison.shape == (sample_input.size(0), 3)
        
        # Comparison predicates
        is_less = seed.is_less_than(a, b)
        assert is_less.shape == (sample_input.size(0),)
        
        is_equal = seed.is_equal(a, b)
        assert is_equal.shape == (sample_input.size(0),)
        
        is_greater = seed.is_greater_than(a, b)
        assert is_greater.shape == (sample_input.size(0),)
        
        # Comparison type
        comp_type = seed.get_comparison_type(a, b)
        assert comp_type.shape == (sample_input.size(0),)
        
        # 파라미터 수 확인
        param_count = seed.count_parameters()
        print(f"Binary Comparator parameters: {param_count}")
    
    def test_all_seeds_metadata(self):
        """모든 시드의 메타데이터 테스트"""
        seeds = [
            EdgeDetector(128, 64),
            SymmetryDetector(128, 128),
            RecurrenceSpotter(128, 96),
            ContrastAmplifier(128, 32),
            GroupingNucleus(128, 128, 8),
            SequenceTracker(128, 160),
            ScaleNormalizer(128, 64),
            BinaryComparator(128, 48),
        ]
        
        print("\n" + "="*60)
        print("Level 0 (Atomic) Seeds Metadata")
        print("="*60)
        
        for seed in seeds:
            metadata = seed.get_metadata()
            param_count = seed.count_parameters()
            
            print(f"\n{metadata['seed_id']}: {metadata['name']}")
            print(f"  Category: {metadata['category']}")
            print(f"  Level: {metadata['level']}")
            print(f"  Bit Depth: {metadata['bit_depth']}")
            print(f"  Target Params: {metadata['params']}")
            print(f"  Actual Params: {param_count}")
            print(f"  Geometries: {metadata['geometries']}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # pytest 없이 직접 실행
    test = TestAtomicSeeds()
    
    batch_size = 4
    seq_len = 32
    input_dim = 128
    
    sample_input = torch.randn(batch_size, seq_len, input_dim)
    sample_scale = torch.ones(batch_size, 1)
    
    print("Testing Level 0 (Atomic) Seeds...")
    print("="*60)
    
    try:
        print("\n[1/8] Testing Edge Detector...")
        test.test_edge_detector(sample_input, sample_scale)
        print("✓ Edge Detector passed")
        
        print("\n[2/8] Testing Symmetry Detector...")
        test.test_symmetry_detector(sample_input, sample_scale)
        print("✓ Symmetry Detector passed")
        
        print("\n[3/8] Testing Recurrence Spotter...")
        test.test_recurrence_spotter(sample_input, sample_scale)
        print("✓ Recurrence Spotter passed")
        
        print("\n[4/8] Testing Contrast Amplifier...")
        test.test_contrast_amplifier(sample_input, sample_scale)
        print("✓ Contrast Amplifier passed")
        
        print("\n[5/8] Testing Grouping Nucleus...")
        test.test_grouping_nucleus(sample_input, sample_scale)
        print("✓ Grouping Nucleus passed")
        
        print("\n[6/8] Testing Sequence Tracker...")
        test.test_sequence_tracker(sample_input, sample_scale)
        print("✓ Sequence Tracker passed")
        
        print("\n[7/8] Testing Scale Normalizer...")
        test.test_scale_normalizer(sample_input, sample_scale)
        print("✓ Scale Normalizer passed")
        
        print("\n[8/8] Testing Binary Comparator...")
        test.test_binary_comparator(sample_input, sample_scale)
        print("✓ Binary Comparator passed")
        
        print("\n" + "="*60)
        print("Testing metadata...")
        test.test_all_seeds_metadata()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

