"""
Cognitive Seed Framework - Core Module

32개의 인지 시드를 동적으로 조합하여 복잡한 태스크를 해결하는 코어 아키텍처입니다.
"""

__version__ = "1.1.0"
__author__ = "Manus AI Team"

from .registry import SeedRegistry, SeedMetadata
from .router import SeedRouter
from .composition import CompositionEngine, CompositionGraph
from .cache import CacheManager
from .metrics import MetricsCollector
from .reproducibility import (
    set_seed,
    seed_worker,
    get_reproducible_dataloader_config,
    check_reproducibility,
    ReproducibleContext,
    enable_reproducibility
)

__all__ = [
    "SeedRegistry",
    "SeedMetadata",
    "SeedRouter",
    "CompositionEngine",
    "CompositionGraph",
    "CacheManager",
    "MetricsCollector",
    # Reproducibility
    "set_seed",
    "seed_worker",
    "get_reproducible_dataloader_config",
    "check_reproducibility",
    "ReproducibleContext",
    "enable_reproducibility",
]

