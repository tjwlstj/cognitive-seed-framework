"""
M06 Context Integrator - 활용 예제

이 파일은 M06 Context Integrator의 다양한 활용 사례를 보여줍니다.

분류: 활용 예제 (Examples)
관련 문서:
- 정보 자료: docs/M06_RESEARCH_MATERIALS.md
- 구현 가이드: docs/M06_IMPLEMENTATION_GUIDE.md
- 메인 코드: seeds/molecular/m06_context_integrator.py
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from seeds.molecular import ContextIntegrator


# ============================================================
# 예제 1: 기본 사용법
# ============================================================

def example_01_basic_usage():
    """
    예제 1: 기본 사용법
    
    Context Integrator의 가장 기본적인 사용 방법을 보여줍니다.
    """
    print("=" * 60)
    print("예제 1: 기본 사용법")
    print("=" * 60)
    
    # 시드 생성
    integrator = ContextIntegrator(input_dim=128)
    
    # 입력 데이터 생성
    batch_size = 4
    seq_len = 50
    x = torch.randn(batch_size, seq_len, 128)
    
    # Forward pass
    output = integrator(x)
    
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"출력 통계:")
    print(f"  - 평균: {output.mean().item():.4f}")
    print(f"  - 표준편차: {output.std().item():.4f}")
    print(f"  - 최소값: {output.min().item():.4f}")
    print(f"  - 최대값: {output.max().item():.4f}")
    
    print("\n✓ 기본 사용법 완료\n")


# ============================================================
# 예제 2: 메타데이터 활용
# ============================================================

def example_02_metadata_usage():
    """
    예제 2: 메타데이터 활용
    
    중간 결과(메타데이터)를 활용하여 맥락 통합 과정을 분석합니다.
    """
    print("=" * 60)
    print("예제 2: 메타데이터 활용")
    print("=" * 60)
    
    integrator = ContextIntegrator(input_dim=128)
    
    # 입력 데이터
    x = torch.randn(2, 30, 128)
    
    # 메타데이터와 함께 forward
    output, metadata = integrator(x, return_metadata=True)
    
    print("추출된 맥락 정보:")
    for key, value in metadata.items():
        if key != 'fusion_weights':
            print(f"  - {key}: {value.shape}")
    
    print(f"\nFusion weights shape: {metadata['fusion_weights'].shape}")
    print(f"Fusion weights 통계:")
    print(f"  - 평균: {metadata['fusion_weights'].mean(dim=[0, 1])}")
    
    # 각 맥락의 기여도 분석
    fusion_weights = metadata['fusion_weights']  # [B, L, 5]
    avg_weights = fusion_weights.mean(dim=[0, 1])  # [5]
    
    context_names = ['Local', 'Global', 'Temporal', 'Hierarchical', 'Group']
    print("\n맥락별 평균 기여도:")
    for name, weight in zip(context_names, avg_weights):
        print(f"  - {name}: {weight.item():.4f} ({weight.item()*100:.1f}%)")
    
    print("\n✓ 메타데이터 활용 완료\n")


# ============================================================
# 예제 3: 맥락 중요도 분석
# ============================================================

def example_03_context_importance():
    """
    예제 3: 맥락 중요도 분석
    
    다양한 입력에 대해 각 맥락의 중요도를 분석합니다.
    """
    print("=" * 60)
    print("예제 3: 맥락 중요도 분석")
    print("=" * 60)
    
    integrator = ContextIntegrator(input_dim=128)
    
    # 다양한 패턴의 입력 생성
    patterns = {
        'Random': torch.randn(4, 50, 128),
        'Periodic': torch.sin(torch.linspace(0, 10, 50).unsqueeze(0).unsqueeze(-1).expand(4, 50, 128)),
        'Linear': torch.linspace(0, 1, 50).unsqueeze(0).unsqueeze(-1).expand(4, 50, 128),
    }
    
    print("패턴별 맥락 중요도:\n")
    
    for pattern_name, x in patterns.items():
        importance = integrator.get_context_importance(x)
        
        print(f"{pattern_name} 패턴:")
        for context_name, weight in importance.items():
            bar = '█' * int(weight * 50)
            print(f"  {context_name:12s}: {bar} {weight:.4f}")
        print()
    
    print("✓ 맥락 중요도 분석 완료\n")


# ============================================================
# 예제 4: 윈도우 크기 영향 분석
# ============================================================

def example_04_window_size_effect():
    """
    예제 4: 윈도우 크기 영향 분석
    
    Local context 윈도우 크기가 성능에 미치는 영향을 분석합니다.
    """
    print("=" * 60)
    print("예제 4: 윈도우 크기 영향 분석")
    print("=" * 60)
    
    integrator = ContextIntegrator(input_dim=128)
    
    # 입력 데이터
    x = torch.randn(2, 50, 128)
    
    # 다양한 윈도우 크기 테스트
    window_sizes = [3, 5, 7, 9, 11]
    
    print("윈도우 크기별 출력 통계:\n")
    print(f"{'Window Size':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 60)
    
    for window_size in window_sizes:
        output = integrator(x, context_window=window_size)
        
        print(f"{window_size:<15} "
              f"{output.mean().item():<12.4f} "
              f"{output.std().item():<12.4f} "
              f"{output.min().item():<12.4f} "
              f"{output.max().item():<12.4f}")
    
    print("\n✓ 윈도우 크기 영향 분석 완료\n")


# ============================================================
# 예제 5: 시각화
# ============================================================

def example_05_visualization():
    """
    예제 5: 맥락 융합 시각화
    
    Fusion weights를 시각화하여 맥락 통합 과정을 이해합니다.
    """
    print("=" * 60)
    print("예제 5: 맥락 융합 시각화")
    print("=" * 60)
    
    integrator = ContextIntegrator(input_dim=128)
    
    # 입력 데이터 (주기적 패턴)
    t = torch.linspace(0, 4 * 3.14159, 50)
    x = torch.sin(t).unsqueeze(0).unsqueeze(-1).expand(1, 50, 128)
    
    # Forward with metadata
    output, metadata = integrator(x, return_metadata=True)
    
    # Fusion weights 추출
    fusion_weights = metadata['fusion_weights'][0].detach().numpy()  # [L, 5]
    
    # 시각화
    plt.figure(figsize=(12, 6))
    
    context_names = ['Local', 'Global', 'Temporal', 'Hierarchical', 'Group']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for i, (name, color) in enumerate(zip(context_names, colors)):
        plt.plot(fusion_weights[:, i], label=name, color=color, linewidth=2)
    
    plt.xlabel('Sequence Position', fontsize=12)
    plt.ylabel('Fusion Weight', fontsize=12)
    plt.title('Context Fusion Weights over Sequence', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 저장
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'm06_fusion_weights.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n시각화 저장: {output_path}")
    
    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        fusion_weights.T,
        cmap='YlOrRd',
        xticklabels=range(0, 50, 5),
        yticklabels=context_names,
        cbar_kws={'label': 'Weight'}
    )
    plt.xlabel('Sequence Position', fontsize=12)
    plt.ylabel('Context Type', fontsize=12)
    plt.title('Context Fusion Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    heatmap_path = output_dir / 'm06_fusion_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"히트맵 저장: {heatmap_path}")
    
    print("\n✓ 시각화 완료\n")


# ============================================================
# 예제 6: 다른 시드와의 연계
# ============================================================

def example_06_integration_with_other_seeds():
    """
    예제 6: 다른 시드와의 연계
    
    M06을 다른 Molecular 시드와 함께 사용하는 방법을 보여줍니다.
    """
    print("=" * 60)
    print("예제 6: 다른 시드와의 연계")
    print("=" * 60)
    
    from seeds.molecular import PatternCompleter
    
    # 시드 생성
    completer = PatternCompleter(input_dim=128)
    integrator = ContextIntegrator(input_dim=128)
    
    # 입력 데이터 (일부 결손)
    x = torch.randn(2, 50, 128)
    mask = torch.ones(2, 50)
    mask[:, 20:30] = 0  # 20~29 인덱스 결손
    
    print("파이프라인: Pattern Completer -> Context Integrator\n")
    
    # Step 1: Pattern completion
    print("Step 1: 패턴 완성...")
    completed = completer(x, mask=mask)
    print(f"  완성된 시퀀스 shape: {completed.shape}")
    
    # Step 2: Context integration
    print("Step 2: 맥락 통합...")
    integrated = integrator(completed)
    print(f"  통합된 시퀀스 shape: {integrated.shape}")
    
    # 결과 비교
    print("\n결과 비교:")
    print(f"  원본 평균: {x.mean().item():.4f}")
    print(f"  완성 후 평균: {completed.mean().item():.4f}")
    print(f"  통합 후 평균: {integrated.mean().item():.4f}")
    
    print("\n✓ 다른 시드와의 연계 완료\n")


# ============================================================
# 예제 7: 실제 응용 - 텍스트 중의성 해소
# ============================================================

def example_07_text_disambiguation():
    """
    예제 7: 텍스트 중의성 해소 (시뮬레이션)
    
    텍스트의 중의성을 맥락을 통해 해소하는 시나리오를 시뮬레이션합니다.
    """
    print("=" * 60)
    print("예제 7: 텍스트 중의성 해소 (시뮬레이션)")
    print("=" * 60)
    
    integrator = ContextIntegrator(input_dim=128)
    
    # 시뮬레이션 시나리오
    # "bank"라는 단어가 "강둑"인지 "은행"인지 맥락으로 판단
    
    # 시나리오 1: "강둑" 맥락
    print("\n시나리오 1: '강둑' 맥락")
    print("문장: 'We sat by the bank and watched the river flow.'")
    
    # 강둑 관련 맥락 임베딩 (시뮬레이션)
    river_context = torch.randn(1, 10, 128) * 0.5 + torch.tensor([1.0, 0.0, 0.5]).view(1, 1, -1).expand(1, 10, 128)[:, :, :128]
    
    output1, meta1 = integrator(river_context, return_metadata=True)
    importance1 = integrator.get_context_importance(river_context)
    
    print("맥락 중요도:")
    for name, weight in importance1.items():
        print(f"  {name}: {weight:.4f}")
    
    # 시나리오 2: "은행" 맥락
    print("\n시나리오 2: '은행' 맥락")
    print("문장: 'I need to go to the bank to withdraw money.'")
    
    # 은행 관련 맥락 임베딩 (시뮬레이션)
    finance_context = torch.randn(1, 10, 128) * 0.5 + torch.tensor([0.0, 1.0, 0.5]).view(1, 1, -1).expand(1, 10, 128)[:, :, :128]
    
    output2, meta2 = integrator(finance_context, return_metadata=True)
    importance2 = integrator.get_context_importance(finance_context)
    
    print("맥락 중요도:")
    for name, weight in importance2.items():
        print(f"  {name}: {weight:.4f}")
    
    # 비교
    print("\n맥락 차이 분석:")
    for name in importance1.keys():
        diff = importance2[name] - importance1[name]
        direction = "↑" if diff > 0 else "↓"
        print(f"  {name}: {direction} {abs(diff):.4f}")
    
    print("\n✓ 텍스트 중의성 해소 완료\n")


# ============================================================
# 예제 8: 성능 벤치마크
# ============================================================

def example_08_performance_benchmark():
    """
    예제 8: 성능 벤치마크
    
    다양한 시퀀스 길이에서의 처리 속도를 측정합니다.
    """
    print("=" * 60)
    print("예제 8: 성능 벤치마크")
    print("=" * 60)
    
    import time
    
    integrator = ContextIntegrator(input_dim=128)
    
    sequence_lengths = [10, 20, 50, 100, 200]
    batch_size = 4
    
    print(f"\n배치 크기: {batch_size}")
    print(f"{'Seq Length':<15} {'Time (ms)':<15} {'Throughput (seq/s)':<20}")
    print("-" * 50)
    
    for seq_len in sequence_lengths:
        x = torch.randn(batch_size, seq_len, 128)
        
        # Warmup
        _ = integrator(x)
        
        # 측정
        num_runs = 10
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = integrator(x)
        
        elapsed_time = (time.time() - start_time) / num_runs
        throughput = batch_size / elapsed_time
        
        print(f"{seq_len:<15} {elapsed_time*1000:<15.2f} {throughput:<20.2f}")
    
    print("\n✓ 성능 벤치마크 완료\n")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """모든 예제 실행"""
    
    print("\n" + "=" * 60)
    print("M06 Context Integrator - 활용 예제")
    print("=" * 60 + "\n")
    
    # 예제 실행
    example_01_basic_usage()
    example_02_metadata_usage()
    example_03_context_importance()
    example_04_window_size_effect()
    example_05_visualization()
    example_06_integration_with_other_seeds()
    example_07_text_disambiguation()
    example_08_performance_benchmark()
    
    print("=" * 60)
    print("모든 예제 완료!")
    print("=" * 60)


if __name__ == "__main__":
    # 개별 예제 실행 (선택)
    # example_01_basic_usage()
    # example_02_metadata_usage()
    # example_03_context_importance()
    # example_04_window_size_effect()
    # example_05_visualization()
    # example_06_integration_with_other_seeds()
    # example_07_text_disambiguation()
    # example_08_performance_benchmark()
    
    # 전체 예제 실행
    main()

