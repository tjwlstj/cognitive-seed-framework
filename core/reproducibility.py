"""
Reproducibility Utilities

PyTorch 모델의 재현성을 보장하기 위한 유틸리티 함수들입니다.
DataLoader worker seed 초기화 및 deterministic 설정을 제공합니다.
"""

import os
import random
import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    전역 랜덤 시드를 설정하여 재현성을 보장합니다.
    
    Args:
        seed: 랜덤 시드 값
        deterministic: True일 경우 완전한 재현성 보장 (성능 저하 가능)
    
    Note:
        - deterministic=True는 일부 연산에서 10-20% 성능 저하 가능
        - 완전한 재현성은 동일 하드웨어, 동일 CUDA 버전에서만 보장
    
    References:
        - https://pytorch.org/docs/stable/notes/randomness.html
        - https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    """
    # Python random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    
    if deterministic:
        # CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Deterministic algorithms (PyTorch 1.8+)
        try:
            torch.use_deterministic_algorithms(True)
            # Required for some operations
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            logger.info(f"Set seed={seed} with full deterministic mode")
        except AttributeError:
            logger.warning(
                "torch.use_deterministic_algorithms not available. "
                "Upgrade to PyTorch 1.8+ for full reproducibility."
            )
    else:
        logger.info(f"Set seed={seed} (deterministic mode disabled)")


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker의 랜덤 시드를 초기화합니다.
    
    PyTorch의 DataLoader는 num_workers > 1일 때 각 worker가 동일한 
    NumPy random seed를 사용하는 버그가 있습니다. 이 함수는 각 worker에게 
    고유한 시드를 할당하여 문제를 해결합니다.
    
    Args:
        worker_id: DataLoader worker ID
    
    Usage:
        >>> from torch.utils.data import DataLoader
        >>> g = torch.Generator()
        >>> g.manual_seed(0)
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     worker_init_fn=seed_worker,
        ...     generator=g
        ... )
    
    References:
        - https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        - https://github.com/pytorch/pytorch/issues/5059
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_reproducible_dataloader_config() -> dict:
    """
    재현 가능한 DataLoader 설정을 반환합니다.
    
    Returns:
        DataLoader에 전달할 설정 딕셔너리
    
    Usage:
        >>> from torch.utils.data import DataLoader
        >>> config = get_reproducible_dataloader_config()
        >>> dataloader = DataLoader(dataset, batch_size=32, **config)
    """
    g = torch.Generator()
    g.manual_seed(0)
    
    return {
        'worker_init_fn': seed_worker,
        'generator': g,
    }


def check_reproducibility(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    seed: int = 42,
    num_runs: int = 3
) -> bool:
    """
    모델의 재현성을 테스트합니다.
    
    Args:
        model: 테스트할 PyTorch 모델
        input_data: 테스트 입력 데이터
        seed: 랜덤 시드
        num_runs: 테스트 실행 횟수
    
    Returns:
        모든 실행 결과가 동일하면 True, 아니면 False
    
    Example:
        >>> model = MyModel()
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> is_reproducible = check_reproducibility(model, input_tensor)
        >>> print(f"Model is reproducible: {is_reproducible}")
    """
    model.eval()
    outputs = []
    
    for i in range(num_runs):
        # 시드 재설정
        set_seed(seed, deterministic=True)
        
        with torch.no_grad():
            output = model(input_data.clone())
            outputs.append(output.cpu().numpy())
    
    # 모든 출력이 동일한지 확인
    for i in range(1, num_runs):
        if not np.allclose(outputs[0], outputs[i], rtol=1e-5, atol=1e-7):
            logger.warning(
                f"Reproducibility check failed: "
                f"Run 0 vs Run {i} differ by max {np.max(np.abs(outputs[0] - outputs[i]))}"
            )
            return False
    
    logger.info(f"Reproducibility check passed ({num_runs} runs)")
    return True


class ReproducibleContext:
    """
    재현 가능한 컨텍스트 매니저
    
    with 블록 내에서 재현성을 보장하고, 블록 종료 시 원래 상태로 복원합니다.
    
    Usage:
        >>> with ReproducibleContext(seed=42):
        ...     output = model(input_data)
        ...     # 이 블록 내에서는 재현성 보장
    """
    
    def __init__(self, seed: int = 42, deterministic: bool = True):
        """
        Args:
            seed: 랜덤 시드
            deterministic: 완전한 재현성 모드 사용 여부
        """
        self.seed = seed
        self.deterministic = deterministic
        
        # 원래 상태 저장
        self.original_python_state = None
        self.original_numpy_state = None
        self.original_torch_state = None
        self.original_cuda_state = None
        self.original_cudnn_deterministic = None
        self.original_cudnn_benchmark = None
    
    def __enter__(self):
        """컨텍스트 진입 시 현재 상태 저장 및 시드 설정"""
        # 현재 상태 저장
        self.original_python_state = random.getstate()
        self.original_numpy_state = np.random.get_state()
        self.original_torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.original_cuda_state = torch.cuda.get_rng_state_all()
        self.original_cudnn_deterministic = torch.backends.cudnn.deterministic
        self.original_cudnn_benchmark = torch.backends.cudnn.benchmark
        
        # 재현성 설정
        set_seed(self.seed, self.deterministic)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료 시 원래 상태 복원"""
        # 상태 복원
        random.setstate(self.original_python_state)
        np.random.set_state(self.original_numpy_state)
        torch.set_rng_state(self.original_torch_state)
        if torch.cuda.is_available() and self.original_cuda_state is not None:
            torch.cuda.set_rng_state_all(self.original_cuda_state)
        torch.backends.cudnn.deterministic = self.original_cudnn_deterministic
        torch.backends.cudnn.benchmark = self.original_cudnn_benchmark
        
        return False


# 편의 함수: 기본 시드로 재현성 설정
def enable_reproducibility(seed: Optional[int] = None) -> None:
    """
    기본 설정으로 재현성을 활성화합니다.
    
    Args:
        seed: 랜덤 시드 (None일 경우 3407 사용)
    
    Note:
        시드 3407은 "Torch.manual_seed(3407) is all you need" 논문에서 
        제안된 최적의 시드입니다.
    
    Reference:
        - https://arxiv.org/abs/2109.08203
    """
    if seed is None:
        seed = 3407  # "Magic seed" from research
    
    set_seed(seed, deterministic=True)
    logger.info(f"Reproducibility enabled with seed={seed}")

