"""
Level 0 (Atomic) Seeds 사용 예제

8개의 원자 시드를 사용하는 방법을 보여줍니다.
"""

import torch
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


def example_edge_detector():
    """SEED-A01: Edge Detector 예제"""
    print("\n" + "="*60)
    print("SEED-A01: Edge Detector")
    print("="*60)
    
    # 시드 생성
    seed = EdgeDetector(input_dim=128, hidden_dim=64)
    
    # 샘플 입력 (배치 크기 2, 시퀀스 길이 50, 특징 차원 128)
    x = torch.randn(2, 50, 128)
    
    # 경계 검출
    output = seed(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 경계 마스크 추출
    edge_mask = seed.detect_edges(x, threshold=0.5)
    print(f"Edge mask shape: {edge_mask.shape}")
    print(f"Detected edges: {edge_mask[0].sum().item():.0f} / {edge_mask.size(1)}")


def example_symmetry_detector():
    """SEED-A02: Symmetry Detector 예제"""
    print("\n" + "="*60)
    print("SEED-A02: Symmetry Detector")
    print("="*60)
    
    # 시드 생성
    seed = SymmetryDetector(input_dim=128, hidden_dim=128)
    
    # 샘플 입력
    x = torch.randn(2, 50, 128)
    
    # 대칭 검출
    output = seed(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 대칭 유형 분석
    symmetry_types, symmetry_axis = seed.detect_symmetry_type(x)
    print(f"Symmetry types shape: {symmetry_types.shape}")
    print(f"Symmetry types (reflection, rotation, translation):")
    print(f"  {symmetry_types[0, 0].tolist()}")
    
    # 대칭성 점수
    symmetry_score = seed.compute_symmetry_score(x)
    print(f"Average symmetry score: {symmetry_score.mean().item():.4f}")


def example_recurrence_spotter():
    """SEED-A03: Recurrence Spotter 예제"""
    print("\n" + "="*60)
    print("SEED-A03: Recurrence Spotter")
    print("="*60)
    
    # 시드 생성
    seed = RecurrenceSpotter(input_dim=128, hidden_dim=96)
    
    # 주기적 패턴이 있는 샘플 생성
    t = torch.linspace(0, 4*3.14159, 50).unsqueeze(0).unsqueeze(-1)
    x = torch.sin(t).expand(2, 50, 128) + torch.randn(2, 50, 128) * 0.1
    
    # 반복 패턴 검출
    output = seed(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 주기 추정
    period = seed.detect_period(x)
    print(f"Estimated period shape: {period.shape}")
    print(f"Average period: {period.mean().item():.4f}")
    
    # 반복성 점수
    recurrence_score = seed.compute_recurrence_score(x, window_size=5)
    print(f"Average recurrence score: {recurrence_score.mean().item():.4f}")


def example_contrast_amplifier():
    """SEED-A04: Contrast Amplifier 예제"""
    print("\n" + "="*60)
    print("SEED-A04: Contrast Amplifier")
    print("="*60)
    
    # 시드 생성
    seed = ContrastAmplifier(input_dim=128, hidden_dim=32)
    
    # 노이즈가 있는 샘플 생성
    signal = torch.randn(2, 50, 128) * 2.0
    noise = torch.randn(2, 50, 128) * 0.5
    x = signal + noise
    
    # 대비 증폭
    output = seed(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # SNR 개선
    snr_improvement = seed.compute_snr(x, output)
    print(f"SNR improvement: {snr_improvement.mean().item():.4f}x")
    
    # 노이즈 제거
    denoised = seed.denoise(x, threshold=0.3)
    print(f"Denoised shape: {denoised.shape}")


def example_grouping_nucleus():
    """SEED-A05: Grouping Nucleus 예제"""
    print("\n" + "="*60)
    print("SEED-A05: Grouping Nucleus")
    print("="*60)
    
    # 시드 생성
    seed = GroupingNucleus(input_dim=128, hidden_dim=128, num_clusters=8)
    
    # 샘플 입력
    x = torch.randn(2, 50, 128)
    
    # 그룹화
    output = seed(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 클러스터 할당
    assignments = seed.get_cluster_assignments(x)
    print(f"Cluster assignments shape: {assignments.shape}")
    
    # 하드 할당
    hard_assignments = seed.get_hard_assignments(x)
    print(f"Hard assignments shape: {hard_assignments.shape}")
    print(f"Cluster distribution: {torch.bincount(hard_assignments[0], minlength=8).tolist()}")


def example_sequence_tracker():
    """SEED-A06: Sequence Tracker 예제"""
    print("\n" + "="*60)
    print("SEED-A06: Sequence Tracker")
    print("="*60)
    
    # 시드 생성
    seed = SequenceTracker(input_dim=128, hidden_dim=160)
    
    # 샘플 입력
    x = torch.randn(2, 50, 128)
    
    # 시퀀스 추적
    output = seed(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 다음 상태 예측
    predictions = seed.predict_next(x, num_steps=5)
    print(f"Predictions shape: {predictions.shape}")
    
    # 추적 정확도
    accuracy = seed.compute_tracking_accuracy(x)
    print(f"Average tracking accuracy: {accuracy.mean().item():.4f}")


def example_scale_normalizer():
    """SEED-A07: Scale Normalizer 예제"""
    print("\n" + "="*60)
    print("SEED-A07: Scale Normalizer")
    print("="*60)
    
    # 시드 생성
    seed = ScaleNormalizer(input_dim=128, hidden_dim=64)
    
    # 다양한 스케일의 샘플 생성
    x = torch.randn(2, 50, 128) * torch.tensor([0.1, 10.0]).view(2, 1, 1)
    
    # 스케일 정규화
    output = seed(x)
    print(f"Input shape: {x.shape}")
    print(f"Input scale range: [{x.abs().min().item():.4f}, {x.abs().max().item():.4f}]")
    print(f"Output shape: {output.shape}")
    print(f"Output scale range: [{output.abs().min().item():.4f}, {output.abs().max().item():.4f}]")
    
    # 스케일 추정
    estimated_scale = seed.estimate_scale(x)
    print(f"Estimated scale shape: {estimated_scale.shape}")
    print(f"Estimated scale: {estimated_scale[0, 0].item():.4f}, {estimated_scale[1, 0].item():.4f}")
    
    # 분산 안정성
    stability = seed.compute_variance_stability(x)
    print(f"Variance stability: {stability.mean().item():.4f}")


def example_binary_comparator():
    """SEED-A08: Binary Comparator 예제"""
    print("\n" + "="*60)
    print("SEED-A08: Binary Comparator")
    print("="*60)
    
    # 시드 생성
    seed = BinaryComparator(input_dim=128, hidden_dim=48)
    
    # 샘플 입력
    x = torch.randn(2, 50, 128)
    
    # 비교 연산
    output = seed(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 두 벡터 비교
    a = torch.randn(4, 128)
    b = torch.randn(4, 128)
    
    comparison = seed.compare(a, b)
    print(f"\nComparison result shape: {comparison.shape}")
    print(f"Comparison probabilities (< = >): {comparison[0].tolist()}")
    
    # 비교 술어
    is_less = seed.is_less_than(a, b)
    is_equal = seed.is_equal(a, b)
    is_greater = seed.is_greater_than(a, b)
    
    print(f"Is less than: {is_less.tolist()}")
    print(f"Is equal: {is_equal.tolist()}")
    print(f"Is greater than: {is_greater.tolist()}")


def main():
    """모든 예제 실행"""
    print("="*60)
    print("Level 0 (Atomic) Seeds Usage Examples")
    print("="*60)
    
    example_edge_detector()
    example_symmetry_detector()
    example_recurrence_spotter()
    example_contrast_amplifier()
    example_grouping_nucleus()
    example_sequence_tracker()
    example_scale_normalizer()
    example_binary_comparator()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()

