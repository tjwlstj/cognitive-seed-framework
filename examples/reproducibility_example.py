"""
Reproducibility Example

PyTorch 재현성 보장 방법을 보여주는 예제입니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core import (
    set_seed,
    seed_worker,
    get_reproducible_dataloader_config,
    check_reproducibility,
    ReproducibleContext,
    enable_reproducibility
)


class SimpleModel(nn.Module):
    """테스트용 간단한 모델"""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def example_1_basic_seed():
    """예제 1: 기본 시드 설정"""
    print("\n" + "="*60)
    print("Example 1: Basic Seed Setting")
    print("="*60)
    
    # 시드 설정
    set_seed(42, deterministic=True)
    
    # 모델 생성 및 초기화
    model = SimpleModel()
    input_data = torch.randn(5, 10)
    
    # 첫 번째 실행
    output1 = model(input_data)
    print(f"First run output:\n{output1}")
    
    # 시드 재설정 후 두 번째 실행
    set_seed(42, deterministic=True)
    model = SimpleModel()
    output2 = model(input_data)
    print(f"\nSecond run output:\n{output2}")
    
    # 결과 비교
    is_same = torch.allclose(output1, output2)
    print(f"\nOutputs are identical: {is_same}")


def example_2_dataloader():
    """예제 2: DataLoader 재현성"""
    print("\n" + "="*60)
    print("Example 2: Reproducible DataLoader")
    print("="*60)
    
    # 데이터셋 생성
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    
    # 재현 가능한 DataLoader 설정
    config = get_reproducible_dataloader_config()
    
    # DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=2,
        **config
    )
    
    # 첫 번째 epoch
    print("First epoch batches:")
    batches1 = []
    for i, (batch_x, batch_y) in enumerate(dataloader):
        if i < 3:  # 처음 3개 배치만 출력
            print(f"  Batch {i}: mean={batch_x.mean():.4f}, std={batch_x.std():.4f}")
        batches1.append(batch_x.clone())
    
    # DataLoader 재생성 (동일한 설정)
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=2,
        **config
    )
    
    # 두 번째 epoch
    print("\nSecond epoch batches (should be identical):")
    batches2 = []
    for i, (batch_x, batch_y) in enumerate(dataloader):
        if i < 3:
            print(f"  Batch {i}: mean={batch_x.mean():.4f}, std={batch_x.std():.4f}")
        batches2.append(batch_x.clone())
    
    # 배치 순서 비교
    all_same = all(torch.allclose(b1, b2) for b1, b2 in zip(batches1, batches2))
    print(f"\nAll batches are identical: {all_same}")


def example_3_check_reproducibility():
    """예제 3: 재현성 자동 체크"""
    print("\n" + "="*60)
    print("Example 3: Automatic Reproducibility Check")
    print("="*60)
    
    model = SimpleModel()
    input_data = torch.randn(1, 10)
    
    # 재현성 체크 (3번 실행하여 모두 동일한지 확인)
    is_reproducible = check_reproducibility(
        model,
        input_data,
        seed=42,
        num_runs=5
    )
    
    print(f"\nModel is reproducible: {is_reproducible}")


def example_4_context_manager():
    """예제 4: 컨텍스트 매니저 사용"""
    print("\n" + "="*60)
    print("Example 4: Reproducible Context Manager")
    print("="*60)
    
    model = SimpleModel()
    input_data = torch.randn(1, 10)
    
    # 재현 가능한 컨텍스트 내에서 실행
    print("Running in reproducible context...")
    with ReproducibleContext(seed=42):
        output1 = model(input_data)
        print(f"Output 1: {output1.item():.6f}")
    
    # 컨텍스트 외부에서는 랜덤
    output_random = model(input_data)
    print(f"Output (random): {output_random.item():.6f}")
    
    # 다시 재현 가능한 컨텍스트
    with ReproducibleContext(seed=42):
        output2 = model(input_data)
        print(f"Output 2: {output2.item():.6f}")
    
    # 비교
    is_same = torch.allclose(output1, output2)
    print(f"\nOutput 1 and Output 2 are identical: {is_same}")


def example_5_magic_seed():
    """예제 5: Magic Seed 3407 사용"""
    print("\n" + "="*60)
    print("Example 5: Magic Seed 3407")
    print("="*60)
    print("Using seed 3407 from 'Torch.manual_seed(3407) is all you need' paper")
    print("Reference: https://arxiv.org/abs/2109.08203")
    
    # Magic seed로 재현성 활성화
    enable_reproducibility()  # 기본값으로 3407 사용
    
    model = SimpleModel()
    input_data = torch.randn(1, 10)
    
    output = model(input_data)
    print(f"\nOutput with magic seed: {output.item():.6f}")
    
    # 재현성 확인
    is_reproducible = check_reproducibility(model, input_data, seed=3407, num_runs=3)
    print(f"Reproducibility verified: {is_reproducible}")


def main():
    print("="*60)
    print("Cognitive Seed Framework - Reproducibility Examples")
    print("="*60)
    
    # 모든 예제 실행
    example_1_basic_seed()
    example_2_dataloader()
    example_3_check_reproducibility()
    example_4_context_manager()
    example_5_magic_seed()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

