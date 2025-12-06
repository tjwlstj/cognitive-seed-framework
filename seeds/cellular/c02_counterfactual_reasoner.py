"""
SEED-C02: Counterfactual Reasoner

반사실 시뮬레이션을 통해 "만약 ~했다면" 시나리오를 추론하는 Cellular 레벨 시드입니다.

구성 시드:
- M02: Causality Detector (인과 구조 파악)
- M08: Conflict Resolver (대안 시나리오 간 충돌 해소)
- A08: Binary Comparator (사실/반사실 비교)

주요 기능:
- 인과 구조 기반 반사실 시나리오 생성
- 개입 효과 시뮬레이션
- 대안 시나리오 간 비교 및 평가
- 일관성 있는 반사실 추론

Author: Manus AI
Date: 2025-12-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from seeds.base import BaseSeed, SeedConfig
from seeds.molecular.m02_causality_detector import CausalityDetector
from seeds.molecular.m08_conflict_resolver import ConflictResolver
from seeds.atomic.a08_binary_comparator import BinaryComparator


@dataclass
class CounterfactualReasonerConfig(SeedConfig):
    """Counterfactual Reasoner 설정"""
    seed_id: str = "SEED-C02"
    name: str = "Counterfactual Reasoner"
    level: int = 2
    category: str = "Logic"
    bit_depth: str = "FP8"
    params: int = 6200000  # ~6.2M
    input_dim: int = 128
    output_dim: int = 128
    
    # C02 특화 설정
    num_scenarios: int = 5  # 생성할 반사실 시나리오 수
    intervention_strength: float = 0.5
    consistency_weight: float = 0.3
    dropout: float = 0.1


class CounterfactualReasoner(BaseSeed):
    """
    SEED-C02: Counterfactual Reasoner
    
    반사실 추론을 통해 "만약 ~했다면" 시나리오를 생성하고 평가합니다.
    
    주요 기능:
    - 인과 구조 파악 및 개입 지점 식별
    - 반사실 시나리오 생성
    - 개입 효과 시뮬레이션
    - 시나리오 간 비교 및 평가
    - 일관성 검증
    
    Examples:
        >>> reasoner = CounterfactualReasoner(input_dim=128)
        >>> factual = torch.randn(4, 10, 128)  # 4 batches, 10 time steps
        >>> intervention = {'time_step': 3, 'value': torch.randn(4, 1, 128)}
        >>> counterfactual, comparison = reasoner(factual, intervention=intervention)
        >>> counterfactual.shape
        torch.Size([4, 10, 128])
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_scenarios: int = 5,
        intervention_strength: float = 0.5,
        consistency_weight: float = 0.3,
        dropout: float = 0.1
    ):
        config = CounterfactualReasonerConfig(
            input_dim=input_dim,
            output_dim=input_dim,
            num_scenarios=num_scenarios,
            intervention_strength=intervention_strength,
            consistency_weight=consistency_weight,
            dropout=dropout
        )
        super().__init__(config)
        
        self.config = config
        
        # 컴포넌트 초기화
        self._init_molecular_atomic_seeds()
        self._init_scenario_generator()
        self._init_intervention_module()
        self._init_consistency_checker()
    
    def _init_molecular_atomic_seeds(self):
        """Molecular/Atomic seeds 초기화"""
        # M02: Causality Detector - 인과 구조 파악
        self.causality_detector = CausalityDetector(
            input_dim=self.config.input_dim
        )
        
        # M08: Conflict Resolver - 대안 시나리오 간 충돌 해소
        self.conflict_resolver = ConflictResolver(
            input_dim=self.config.input_dim,
            num_constraints_max=self.config.num_scenarios
        )
        
        # A08: Binary Comparator - 사실/반사실 비교
        self.comparator = BinaryComparator(
            input_dim=self.config.input_dim,
            hidden_dim=64
        )
    
    def _init_scenario_generator(self):
        """반사실 시나리오 생성기 초기화"""
        # 시나리오 인코더
        self.scenario_encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, self.config.input_dim)
        )
        
        # 시나리오 생성 네트워크 (GRU 기반)
        self.scenario_generator = nn.GRU(
            input_size=self.config.input_dim,
            hidden_size=self.config.input_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.config.dropout if self.config.dropout > 0 else 0
        )
        
        # 다중 시나리오 생성을 위한 프로젝션
        self.scenario_projections = nn.ModuleList([
            nn.Linear(self.config.input_dim, self.config.input_dim)
            for _ in range(self.config.num_scenarios)
        ])
    
    def _init_intervention_module(self):
        """개입 모듈 초기화"""
        # 개입 지점 식별 네트워크
        self.intervention_locator = nn.Sequential(
            nn.Linear(self.config.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 개입 적합도 [0, 1]
        )
        
        # 개입 효과 추정기
        self.intervention_effect_estimator = nn.Sequential(
            nn.Linear(self.config.input_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(256, self.config.input_dim)
        )
        
        # 개입 강도 조절기
        self.intervention_modulator = nn.Sequential(
            nn.Linear(self.config.input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _init_consistency_checker(self):
        """일관성 검증 모듈 초기화"""
        # 시나리오 일관성 검증
        self.consistency_checker = nn.Sequential(
            nn.Linear(self.config.input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 일관성 점수 [0, 1]
        )
        
        # 인과 일관성 검증
        self.causal_consistency_net = nn.Sequential(
            nn.Linear(self.config.input_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        intervention: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        반사실 추론 수행
        
        Args:
            x: [B, T, D] - 사실적 시계열 데이터 (factual scenario)
            scale: [B, 1] - 스케일 매개변수
            intervention: 개입 정보 딕셔너리
                - 'time_step': int - 개입 시점
                - 'value': [B, 1, D] - 개입 값
                - 'strength': float (optional) - 개입 강도
        
        Returns:
            counterfactual: [B, T, D] - 반사실 시나리오
            metadata: 추가 정보 딕셔너리
                - 'causal_graph': 인과 그래프
                - 'intervention_effects': 개입 효과
                - 'consistency_scores': 일관성 점수
                - 'comparison': 사실/반사실 비교 결과
        """
        batch_size, seq_len, dim = x.shape
        
        # CSE: 스케일 조건부 정규화
        if self.config.use_cse:
            x = self.cse(x, scale)
        
        # MGP: 기하학적 투영
        if self.config.use_mgp:
            x_proj = self.mgp(x)
        else:
            x_proj = x
        
        # 1. 인과 구조 파악 (M02)
        causal_features = self.causality_detector(x_proj)
        causal_graph = self.causality_detector.estimate_causal_graph(x_proj)
        
        # 2. 개입 지점 식별 및 개입 적용
        if intervention is None:
            # 자동으로 최적 개입 지점 찾기
            intervention_scores = self.intervention_locator(causal_features)  # [B, T, 1]
            intervention_time = torch.argmax(intervention_scores.squeeze(-1), dim=1)  # [B]
            
            # 자동 개입 값 생성
            intervention_value = self._generate_intervention_value(
                x_proj, intervention_time, causal_features
            )
        else:
            intervention_time = intervention.get('time_step', seq_len // 2)
            intervention_value = intervention.get('value')
            if isinstance(intervention_time, int):
                intervention_time = torch.full((batch_size,), intervention_time, device=x.device)
        
        # 3. 반사실 시나리오 생성
        counterfactual = self._generate_counterfactual_scenario(
            x_proj, causal_features, intervention_time, intervention_value
        )
        
        # 4. 개입 효과 추정
        intervention_effects = self._estimate_intervention_effects(
            x_proj, counterfactual, intervention_time
        )
        
        # 5. 일관성 검증
        consistency_scores = self._check_consistency(
            x_proj, counterfactual, causal_features
        )
        
        # 6. 사실/반사실 비교 (A08)
        comparison = self._compare_scenarios(x_proj, counterfactual)
        
        # 메타데이터 구성
        metadata = {
            'causal_graph': causal_graph,
            'intervention_effects': intervention_effects,
            'consistency_scores': consistency_scores,
            'comparison': comparison,
            'intervention_time': intervention_time
        }
        
        return counterfactual, metadata
    
    def _generate_intervention_value(
        self,
        x: torch.Tensor,
        intervention_time: torch.Tensor,
        causal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        자동으로 개입 값 생성
        
        Args:
            x: [B, T, D]
            intervention_time: [B]
            causal_features: [B, T, D]
        
        Returns:
            intervention_value: [B, 1, D]
        """
        batch_size = x.shape[0]
        
        # 개입 시점의 특징 추출
        intervention_features = torch.stack([
            causal_features[i, intervention_time[i], :]
            for i in range(batch_size)
        ])  # [B, D]
        
        # 개입 값 생성 (노이즈 추가)
        noise = torch.randn_like(intervention_features) * 0.1
        intervention_value = intervention_features + noise
        
        return intervention_value.unsqueeze(1)  # [B, 1, D]
    
    def _generate_counterfactual_scenario(
        self,
        factual: torch.Tensor,
        causal_features: torch.Tensor,
        intervention_time: torch.Tensor,
        intervention_value: torch.Tensor
    ) -> torch.Tensor:
        """
        반사실 시나리오 생성
        
        Args:
            factual: [B, T, D] - 사실적 시나리오
            causal_features: [B, T, D] - 인과 특징
            intervention_time: [B] - 개입 시점
            intervention_value: [B, 1, D] - 개입 값
        
        Returns:
            counterfactual: [B, T, D]
        """
        batch_size, seq_len, dim = factual.shape
        
        # 시나리오 인코딩
        encoded = self.scenario_encoder(factual)
        
        # 개입 시점까지는 사실적 시나리오 유지
        counterfactual = factual.detach().clone()
        
        # 개입 적용 (베터화된 연산으로 inplace 피하기)
        intervention_mask = torch.zeros(batch_size, seq_len, 1, device=factual.device)
        intervention_values_expanded = torch.zeros(batch_size, seq_len, dim, device=factual.device)
        
        for i in range(batch_size):
            t = intervention_time[i].item()
            if 0 <= t < seq_len:
                # 개입 강도 계산
                strength = self.intervention_modulator(encoded[i, t:t+1, :])  # [1, 1]
                strength = strength * self.config.intervention_strength
                
                intervention_mask[i, t, 0] = strength
                intervention_values_expanded[i, t, :] = intervention_value[i, 0, :]
        
        # 개입 적용 (벡터 연산)
        counterfactual = (1 - intervention_mask) * counterfactual + intervention_mask * intervention_values_expanded
        
        # 개입 이후 시점 재생성 (새 텐서 생성)
        counterfactual_list = []
        for i in range(batch_size):
            t = intervention_time[i].item()
            if t < seq_len - 1:
                # 개입 이후 시퀀스 재생성
                prefix = counterfactual[i:i+1, :t+1, :]  # [1, t+1, D]
                
                # GRU로 이후 시퀀스 생성
                _, hidden = self.scenario_generator(prefix)
                
                # 나머지 시점 생성
                generated_steps = []
                current_input = counterfactual[i:i+1, t:t+1, :]
                for step in range(t+1, seq_len):
                    output, hidden = self.scenario_generator(current_input, hidden)
                    generated_steps.append(output)
                    current_input = output
                
                # 전체 시퀀스 결합
                if generated_steps:
                    generated = torch.cat(generated_steps, dim=1)  # [1, seq_len-t-1, D]
                    full_sequence = torch.cat([
                        counterfactual[i:i+1, :t+1, :],
                        generated
                    ], dim=1)  # [1, seq_len, D]
                else:
                    full_sequence = counterfactual[i:i+1, :, :]
                
                counterfactual_list.append(full_sequence)
            else:
                counterfactual_list.append(counterfactual[i:i+1, :, :])
        
        # 배치 결합
        counterfactual = torch.cat(counterfactual_list, dim=0)  # [B, T, D]
        
        return counterfactual
    
    def _estimate_intervention_effects(
        self,
        factual: torch.Tensor,
        counterfactual: torch.Tensor,
        intervention_time: torch.Tensor
    ) -> torch.Tensor:
        """
        개입 효과 추정
        
        Args:
            factual: [B, T, D]
            counterfactual: [B, T, D]
            intervention_time: [B]
        
        Returns:
            effects: [B, T, D] - 시간에 따른 개입 효과
        """
        # 사실/반사실 차이 계산
        difference = counterfactual - factual
        
        # 개입 효과 추정
        combined = torch.cat([factual, counterfactual], dim=-1)
        effects = self.intervention_effect_estimator(combined)
        
        # 개입 시점 이전은 효과 없음
        batch_size, seq_len, dim = factual.shape
        mask = torch.zeros(batch_size, seq_len, 1, device=factual.device)
        
        for i in range(batch_size):
            t = intervention_time[i].item()
            if 0 <= t < seq_len:
                mask[i, t:, :] = 1.0
        
        effects = effects * mask
        
        return effects
    
    def _check_consistency(
        self,
        factual: torch.Tensor,
        counterfactual: torch.Tensor,
        causal_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        반사실 시나리오의 일관성 검증
        
        Args:
            factual: [B, T, D]
            counterfactual: [B, T, D]
            causal_features: [B, T, D]
        
        Returns:
            consistency_scores: 일관성 점수 딕셔너리
        """
        batch_size, seq_len, dim = factual.shape
        
        # 1. 시간적 일관성 (인접 시점 간)
        temporal_consistency = []
        for t in range(seq_len - 1):
            current = counterfactual[:, t, :]
            next_step = counterfactual[:, t+1, :]
            pair = torch.cat([current, next_step], dim=-1)
            score = self.consistency_checker(pair)  # [B, 1]
            temporal_consistency.append(score)
        
        temporal_consistency = torch.stack(temporal_consistency, dim=1)  # [B, T-1, 1]
        temporal_consistency = torch.cat([
            temporal_consistency,
            torch.ones(batch_size, 1, 1, device=factual.device)
        ], dim=1)  # [B, T, 1]
        
        # 2. 인과 일관성 (인과 구조 보존)
        causal_consistency = []
        for t in range(seq_len):
            fact = factual[:, t, :]
            counter = counterfactual[:, t, :]
            causal = causal_features[:, t, :]
            triple = torch.cat([fact, counter, causal], dim=-1)
            score = self.causal_consistency_net(triple)  # [B, 1]
            causal_consistency.append(score)
        
        causal_consistency = torch.stack(causal_consistency, dim=1)  # [B, T, 1]
        
        # 3. 전체 일관성 점수
        overall_consistency = (
            temporal_consistency.mean(dim=1) * 0.5 +
            causal_consistency.mean(dim=1) * 0.5
        )  # [B, 1]
        
        return {
            'temporal': temporal_consistency.squeeze(-1),  # [B, T]
            'causal': causal_consistency.squeeze(-1),      # [B, T]
            'overall': overall_consistency.squeeze(-1)     # [B]
        }
    
    def _compare_scenarios(
        self,
        factual: torch.Tensor,
        counterfactual: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        사실/반사실 시나리오 비교 (A08 활용)
        
        Args:
            factual: [B, T, D]
            counterfactual: [B, T, D]
        
        Returns:
            comparison: 비교 결과 딕셔너리
        """
        # A08 Binary Comparator로 비교
        # 사실과 반사실을 결합하여 비교
        combined = torch.cat([factual, counterfactual], dim=1)  # [B, 2T, D]
        comparison_features = self.comparator(combined)
        
        # 차이 분석
        difference = torch.abs(factual - counterfactual)
        difference_magnitude = difference.norm(dim=-1)  # [B, T]
        
        # 유사도 계산
        similarity = F.cosine_similarity(
            factual.reshape(factual.shape[0], -1),
            counterfactual.reshape(counterfactual.shape[0], -1),
            dim=-1
        )  # [B]
        
        return {
            'features': comparison_features,
            'difference_magnitude': difference_magnitude,
            'similarity': similarity
        }
    
    def generate_multiple_scenarios(
        self,
        x: torch.Tensor,
        num_scenarios: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        """
        다중 반사실 시나리오 생성
        
        Args:
            x: [B, T, D] - 사실적 시나리오
            num_scenarios: 생성할 시나리오 수 (기본값: config.num_scenarios)
        
        Returns:
            scenarios: 반사실 시나리오 리스트
            metadata_list: 각 시나리오의 메타데이터 리스트
        """
        if num_scenarios is None:
            num_scenarios = self.config.num_scenarios
        
        scenarios = []
        metadata_list = []
        
        batch_size, seq_len, dim = x.shape
        
        # 다양한 개입 시점과 값으로 시나리오 생성
        for i in range(num_scenarios):
            # 개입 시점 다양화
            intervention_time = int(seq_len * (i + 1) / (num_scenarios + 1))
            
            # 개입 값 다양화
            intervention_value = torch.randn(batch_size, 1, dim, device=x.device) * 0.5
            
            intervention = {
                'time_step': intervention_time,
                'value': intervention_value
            }
            
            scenario, metadata = self.forward(x, intervention=intervention)
            scenarios.append(scenario)
            metadata_list.append(metadata)
        
        # 시나리오 간 충돌 해소 (M08)
        if len(scenarios) > 1:
            scenarios_tensor = torch.stack(scenarios, dim=1)  # [B, num_scenarios, T, D]
            B, N, T, D = scenarios_tensor.shape
            
            # 각 시간 단계별로 충돌 해소
            resolved_scenarios = []
            for t in range(T):
                constraints = scenarios_tensor[:, :, t, :]  # [B, N, D]
                context = x[:, :t+1, :] if t > 0 else x[:, :1, :]  # [B, t+1, D]
                
                # M08은 constraints만 받음 (context는 내부에서 처리)
                resolved, _, _ = self.conflict_resolver(constraints)
                resolved_scenarios.append(resolved)
            
            # 최종 조화된 시나리오
            harmonized_scenario = torch.stack(resolved_scenarios, dim=1)  # [B, T, D]
            scenarios.append(harmonized_scenario)
            
            metadata_list.append({
                'type': 'harmonized',
                'source_scenarios': len(scenarios) - 1
            })
        
        return scenarios, metadata_list


def create_counterfactual_reasoner(
    input_dim: int = 128,
    num_scenarios: int = 5
) -> CounterfactualReasoner:
    """Counterfactual Reasoner 시드 생성 함수"""
    return CounterfactualReasoner(
        input_dim=input_dim,
        num_scenarios=num_scenarios
    )
