"""
Seed Router Module

입력과 태스크를 분석하여 실행할 시드를 동적으로 선택하는 라우터입니다.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .registry import SeedRegistry

logger = logging.getLogger(__name__)


class TaskEncoder(nn.Module):
    """태스크 설명을 벡터로 인코딩하는 모듈"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, hidden_dim: int = 512):
        """
        Args:
            vocab_size: 어휘 크기
            embed_dim: 임베딩 차원
            hidden_dim: 은닉층 차원
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, task_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_tokens: 토큰화된 태스크 설명 [batch, seq_len]
        
        Returns:
            태스크 임베딩 [batch, hidden_dim]
        """
        # 임베딩
        embedded = self.embedding(task_tokens)  # [batch, seq_len, embed_dim]
        
        # LSTM 인코딩
        _, (hidden, _) = self.lstm(embedded)  # hidden: [2, batch, hidden_dim]
        
        # 양방향 은닉 상태 결합
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # [batch, hidden_dim*2]
        
        # 선형 변환
        task_embedding = F.relu(self.fc(hidden))  # [batch, hidden_dim]
        
        return task_embedding


class InputAnalyzer(nn.Module):
    """입력 데이터의 특징을 추출하는 모듈"""
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512):
        """
        Args:
            input_dim: 입력 특징 차원
            hidden_dim: 은닉층 차원
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_features: 입력 특징 [batch, input_dim]
        
        Returns:
            입력 임베딩 [batch, hidden_dim]
        """
        x = F.relu(self.fc1(input_features))
        x = F.relu(self.fc2(x))
        return x


class GatingNetwork(nn.Module):
    """시드 선택을 위한 게이팅 네트워크"""
    
    def __init__(self, hidden_dim: int = 512, num_seeds: int = 32):
        """
        Args:
            hidden_dim: 은닉층 차원
            num_seeds: 총 시드 개수
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_seeds)
    
    def forward(
        self,
        task_embedding: torch.Tensor,
        input_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            task_embedding: 태스크 임베딩 [batch, hidden_dim]
            input_embedding: 입력 임베딩 [batch, hidden_dim]
        
        Returns:
            시드 활성화 확률 [batch, num_seeds]
        """
        # 태스크와 입력 결합
        combined = torch.cat([task_embedding, input_embedding], dim=-1)  # [batch, hidden_dim*2]
        
        # 은닉층
        hidden = F.relu(self.fc1(combined))  # [batch, hidden_dim]
        
        # 시드 점수 계산
        logits = self.fc2(hidden)  # [batch, num_seeds]
        
        # Sigmoid로 독립적 확률 계산 (다중 시드 선택 가능)
        probabilities = torch.sigmoid(logits)
        
        return probabilities


class SeedRouter(nn.Module):
    """시드 라우터 메인 클래스"""
    
    def __init__(
        self,
        registry: SeedRegistry,
        hidden_dim: int = 512,
        vocab_size: int = 10000,
        input_dim: int = 2048
    ):
        """
        Args:
            registry: 시드 레지스트리
            hidden_dim: 은닉층 차원
            vocab_size: 어휘 크기
            input_dim: 입력 특징 차원
        """
        super().__init__()
        self.registry = registry
        self.num_seeds = len(registry)
        
        # 서브모듈 초기화
        self.task_encoder = TaskEncoder(vocab_size, hidden_dim // 2, hidden_dim)
        self.input_analyzer = InputAnalyzer(input_dim, hidden_dim)
        self.gating_network = GatingNetwork(hidden_dim, self.num_seeds)
        
        # 시드 이름 순서 고정 (인덱스 매핑용)
        self.seed_names = sorted(registry.list_all())
        
        logger.info(f"Seed Router initialized with {self.num_seeds} seeds")
    
    def forward(
        self,
        task_tokens: torch.Tensor,
        input_features: torch.Tensor,
        threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> Tuple[List[str], torch.Tensor]:
        """
        태스크와 입력을 분석하여 실행할 시드 선택
        
        Args:
            task_tokens: 토큰화된 태스크 설명 [batch, seq_len]
            input_features: 입력 특징 [batch, input_dim]
            threshold: 시드 선택 임계값 (0~1)
            top_k: 상위 k개 시드만 선택 (None이면 threshold 사용)
        
        Returns:
            selected_seeds: 선택된 시드 이름 리스트
            probabilities: 각 시드의 활성화 확률 [batch, num_seeds]
        """
        # 1. 태스크 인코딩
        task_embedding = self.task_encoder(task_tokens)
        
        # 2. 입력 분석
        input_embedding = self.input_analyzer(input_features)
        
        # 3. 게이팅 네트워크로 시드 확률 계산
        probabilities = self.gating_network(task_embedding, input_embedding)
        
        # 4. 시드 선택
        if top_k is not None:
            # Top-k 선택
            _, indices = torch.topk(probabilities[0], k=top_k)
            selected_indices = indices.tolist()
        else:
            # Threshold 기반 선택
            selected_indices = (probabilities[0] > threshold).nonzero(as_tuple=True)[0].tolist()
        
        # 인덱스를 시드 이름으로 변환
        selected_seeds = [self.seed_names[idx] for idx in selected_indices]
        
        logger.info(f"Selected {len(selected_seeds)} seeds: {selected_seeds}")
        
        return selected_seeds, probabilities
    
    def select_with_dependencies(
        self,
        task_tokens: torch.Tensor,
        input_features: torch.Tensor,
        threshold: float = 0.5,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        시드 선택 후 의존성까지 포함하여 반환
        
        Args:
            task_tokens: 토큰화된 태스크 설명
            input_features: 입력 특징
            threshold: 시드 선택 임계값
            top_k: 상위 k개 시드만 선택
        
        Returns:
            의존성을 포함한 시드 이름 리스트
        """
        # 기본 시드 선택
        selected_seeds, _ = self.forward(task_tokens, input_features, threshold, top_k)
        
        # 의존성 추가
        all_seeds = set(selected_seeds)
        for seed_name in selected_seeds:
            dependencies = self.registry.get_dependencies(seed_name, recursive=True)
            all_seeds.update(dependencies)
        
        result = list(all_seeds)
        logger.info(f"Seeds with dependencies: {len(result)} total")
        
        return result
    
    def explain_selection(
        self,
        task_tokens: torch.Tensor,
        input_features: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """
        시드 선택 이유를 확률과 함께 설명
        
        Args:
            task_tokens: 토큰화된 태스크 설명
            input_features: 입력 특징
        
        Returns:
            (시드 이름, 확률) 튜플의 리스트 (확률 내림차순)
        """
        _, probabilities = self.forward(task_tokens, input_features, threshold=0.0)
        
        # 확률 내림차순 정렬
        probs = probabilities[0].detach().cpu().tolist()
        seed_probs = list(zip(self.seed_names, probs))
        seed_probs.sort(key=lambda x: x[1], reverse=True)
        
        return seed_probs

