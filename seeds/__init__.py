"""
Cognitive Seed Framework - Seeds Module

32개의 인지 시드 구현을 포함합니다.
- Level 0 (Atomic): 8개의 기본 인지 시드
- Level 1 (Molecular): 8개의 조합 시드
- Level 2 (Cellular): 8개의 복합 시드
- Level 3 (Tissue): 8개의 고급 시드
"""

__version__ = "1.1.0"

from .base import BaseSeed, SeedConfig

# Atomic seeds
from .atomic.a01_edge_detector import EdgeDetector as A01_EdgeDetector
from .atomic.a02_symmetry_detector import SymmetryDetector as A02_SymmetryDetector
from .atomic.a03_recurrence_spotter import RecurrenceSpotter as A03_RecurrenceSpotter
from .atomic.a04_contrast_amplifier import ContrastAmplifier as A04_ContrastAmplifier
from .atomic.a05_grouping_nucleus import GroupingNucleus as A05_GroupingNucleus
from .atomic.a06_sequence_tracker import SequenceTracker as A06_SequenceTracker
from .atomic.a07_scale_normalizer import ScaleNormalizer as A07_ScaleNormalizer
from .atomic.a08_binary_comparator import BinaryComparator as A08_BinaryComparator

# Molecular seeds
from .molecular.m01_hierarchy_builder import HierarchyBuilder as M01_HierarchyBuilder
from .molecular.m02_causality_detector import CausalityDetector as M02_CausalityDetector
from .molecular.m03_pattern_completer import PatternCompleter as M03_PatternCompleter
from .molecular.m04_spatial_transformer import SpatialTransformer as M04_SpatialTransformer
from .molecular.m05_concept_crystallizer import ConceptCrystallizer as M05_ConceptCrystallizer
from .molecular.m06_context_integrator import ContextIntegrator as M06_ContextIntegrator
from .molecular.m07_analogy_mapper import AnalogyMapper as M07_AnalogyMapper
from .molecular.m08_conflict_resolver import ConflictResolver as M08_ConflictResolver

# Cellular seeds
from .cellular.c01_metaphor_engine import MetaphorEngine as C01_MetaphorEngine
from .cellular.c02_counterfactual_reasoner import CounterfactualReasoner as C02_CounterfactualReasoner
from .cellular.c03_schema_learner import SchemaLearner as C03_SchemaLearner
from .cellular.c04_perspective_shifter import PerspectiveShifter as C04_PerspectiveShifter
from .cellular.c05_narrative_constructor import NarrativeConstructor as C05_NarrativeConstructor
from .cellular.c06_attention_director import AttentionDirector as C06_AttentionDirector
from .cellular.c07_boundary_detector import BoundaryDetector as C07_BoundaryDetector
from .cellular.c08_novelty_assessor import NoveltyAssessor as C08_NoveltyAssessor

# Tissue seeds
from .tissue.t01_abductive_reasoner import AbductiveReasoner as T01_AbductiveReasoner

# Seed ID to class mapping
_SEED_REGISTRY = {
    # Atomic seeds - support multiple naming conventions
    "A01": A01_EdgeDetector,
    "SEED-A01": A01_EdgeDetector,
    "A01_Edge_Detector": A01_EdgeDetector,
    "A01_EdgeDetector": A01_EdgeDetector,
    
    "A02": A02_SymmetryDetector,
    "SEED-A02": A02_SymmetryDetector,
    "A02_Symmetry_Detector": A02_SymmetryDetector,
    "A02_SymmetryDetector": A02_SymmetryDetector,
    
    "A03": A03_RecurrenceSpotter,
    "SEED-A03": A03_RecurrenceSpotter,
    "A03_Recurrence_Spotter": A03_RecurrenceSpotter,
    "A03_RecurrenceSpotter": A03_RecurrenceSpotter,
    
    "A04": A04_ContrastAmplifier,
    "SEED-A04": A04_ContrastAmplifier,
    "A04_Contrast_Amplifier": A04_ContrastAmplifier,
    "A04_ContrastAmplifier": A04_ContrastAmplifier,
    
    "A05": A05_GroupingNucleus,
    "SEED-A05": A05_GroupingNucleus,
    "A05_Grouping_Nucleus": A05_GroupingNucleus,
    "A05_GroupingNucleus": A05_GroupingNucleus,
    
    "A06": A06_SequenceTracker,
    "SEED-A06": A06_SequenceTracker,
    "A06_Sequence_Tracker": A06_SequenceTracker,
    "A06_SequenceTracker": A06_SequenceTracker,
    
    "A07": A07_ScaleNormalizer,
    "SEED-A07": A07_ScaleNormalizer,
    "A07_Scale_Normalizer": A07_ScaleNormalizer,
    "A07_ScaleNormalizer": A07_ScaleNormalizer,
    
    "A08": A08_BinaryComparator,
    "SEED-A08": A08_BinaryComparator,
    "A08_Binary_Comparator": A08_BinaryComparator,
    "A08_BinaryComparator": A08_BinaryComparator,
    
    # Molecular seeds
    "M01": M01_HierarchyBuilder,
    "SEED-M01": M01_HierarchyBuilder,
    "M01_Hierarchy_Builder": M01_HierarchyBuilder,
    "M01_HierarchyBuilder": M01_HierarchyBuilder,
    
    "M02": M02_CausalityDetector,
    "SEED-M02": M02_CausalityDetector,
    "M02_Causality_Detector": M02_CausalityDetector,
    "M02_CausalityDetector": M02_CausalityDetector,
    
    "M03": M03_PatternCompleter,
    "SEED-M03": M03_PatternCompleter,
    "M03_Pattern_Completer": M03_PatternCompleter,
    "M03_PatternCompleter": M03_PatternCompleter,
    
    "M04": M04_SpatialTransformer,
    "SEED-M04": M04_SpatialTransformer,
    "M04_Spatial_Transformer": M04_SpatialTransformer,
    "M04_SpatialTransformer": M04_SpatialTransformer,
    
    "M05": M05_ConceptCrystallizer,
    "SEED-M05": M05_ConceptCrystallizer,
    "M05_Concept_Crystallizer": M05_ConceptCrystallizer,
    "M05_ConceptCrystallizer": M05_ConceptCrystallizer,
    
    "M06": M06_ContextIntegrator,
    "SEED-M06": M06_ContextIntegrator,
    "M06_Context_Integrator": M06_ContextIntegrator,
    "M06_ContextIntegrator": M06_ContextIntegrator,
    
    "M07": M07_AnalogyMapper,
    "SEED-M07": M07_AnalogyMapper,
    "M07_Analogy_Mapper": M07_AnalogyMapper,
    "M07_AnalogyMapper": M07_AnalogyMapper,
    
    "M08": M08_ConflictResolver,
    "SEED-M08": M08_ConflictResolver,
    "M08_Conflict_Resolver": M08_ConflictResolver,
    "M08_ConflictResolver": M08_ConflictResolver,
    
    # Cellular seeds
    "C01": C01_MetaphorEngine,
    "SEED-C01": C01_MetaphorEngine,
    "C01_Metaphor_Engine": C01_MetaphorEngine,
    "C01_MetaphorEngine": C01_MetaphorEngine,
    
    "C02": C02_CounterfactualReasoner,
    "SEED-C02": C02_CounterfactualReasoner,
    "C02_Counterfactual_Reasoner": C02_CounterfactualReasoner,
    "C02_CounterfactualReasoner": C02_CounterfactualReasoner,
    
    "C03": C03_SchemaLearner,
    "SEED-C03": C03_SchemaLearner,
    "C03_Schema_Learner": C03_SchemaLearner,
    "C03_SchemaLearner": C03_SchemaLearner,
    
    "C04": C04_PerspectiveShifter,
    "SEED-C04": C04_PerspectiveShifter,
    "C04_Perspective_Shifter": C04_PerspectiveShifter,
    "C04_PerspectiveShifter": C04_PerspectiveShifter,
    
    "C05": C05_NarrativeConstructor,
    "SEED-C05": C05_NarrativeConstructor,
    "C05_Narrative_Constructor": C05_NarrativeConstructor,
    "C05_NarrativeConstructor": C05_NarrativeConstructor,
    
    "C06": C06_AttentionDirector,
    "SEED-C06": C06_AttentionDirector,
    "C06_Attention_Director": C06_AttentionDirector,
    "C06_AttentionDirector": C06_AttentionDirector,
    
    "C07": C07_BoundaryDetector,
    "SEED-C07": C07_BoundaryDetector,
    "C07_Boundary_Detector": C07_BoundaryDetector,
    "C07_BoundaryDetector": C07_BoundaryDetector,
    
    "C08": C08_NoveltyAssessor,
    "SEED-C08": C08_NoveltyAssessor,
    "C08_Novelty_Assessor": C08_NoveltyAssessor,
    "C08_NoveltyAssessor": C08_NoveltyAssessor,
    
    # Tissue seeds
    "T01": T01_AbductiveReasoner,
    "SEED-T01": T01_AbductiveReasoner,
    "T01_Abductive_Reasoner": T01_AbductiveReasoner,
    "T01_AbductiveReasoner": T01_AbductiveReasoner,
}


def load_seed(seed_id: str, **kwargs):
    """
    시드 ID로 시드 인스턴스를 로드합니다.
    
    다양한 명명 규칙을 지원합니다:
    - "A01", "SEED-A01", "A01_Edge_Detector", "A01_EdgeDetector" 모두 동일한 시드를 반환
    
    Args:
        seed_id: 시드 식별자 (예: "A01", "SEED-A01", "A01_Edge_Detector")
        **kwargs: 시드 초기화 파라미터
    
    Returns:
        초기화된 시드 인스턴스
    
    Raises:
        KeyError: 등록되지 않은 시드 ID인 경우
    
    Examples:
        >>> edge_detector = load_seed("SEED-A01")
        >>> edge_detector = load_seed("A01")  # 동일한 결과
        >>> output = edge_detector(input_tensor)
    """
    if seed_id not in _SEED_REGISTRY:
        available_ids = sorted(set([k.split('_')[0] for k in _SEED_REGISTRY.keys() if k.startswith(('A', 'M', 'C', 'T'))]))
        raise KeyError(
            f"Seed '{seed_id}' not found. "
            f"Available seed IDs: {', '.join(available_ids)}"
        )
    
    seed_class = _SEED_REGISTRY[seed_id]
    return seed_class(**kwargs)


def list_available_seeds():
    """
    사용 가능한 모든 시드 목록을 반환합니다.
    
    Returns:
        시드 ID 리스트 (canonical 형식)
    """
    canonical_ids = sorted(set([
        k for k in _SEED_REGISTRY.keys() 
        if len(k) <= 3 and k[0] in ['A', 'M', 'C', 'T']
    ]))
    return canonical_ids


__all__ = [
    "BaseSeed",
    "SeedConfig",
    "load_seed",
    "list_available_seeds",
    # Atomic
    "A01_EdgeDetector",
    "A02_SymmetryDetector",
    "A03_RecurrenceSpotter",
    "A04_ContrastAmplifier",
    "A05_GroupingNucleus",
    "A06_SequenceTracker",
    "A07_ScaleNormalizer",
    "A08_BinaryComparator",
    # Molecular
    "M01_HierarchyBuilder",
    "M02_CausalityDetector",
    "M03_PatternCompleter",
    "M04_SpatialTransformer",
    "M05_ConceptCrystallizer",
    "M06_ContextIntegrator",
    "M07_AnalogyMapper",
    "M08_ConflictResolver",
    # Cellular
    "C01_MetaphorEngine",
    "C02_CounterfactualReasoner",
    "C03_SchemaLearner",
    "C04_PerspectiveShifter",
    "C05_NarrativeConstructor",
    "C06_AttentionDirector",
    "C07_BoundaryDetector",
    "C08_NoveltyAssessor",
    # Tissue
    "T01_AbductiveReasoner",
]

