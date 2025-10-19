# Neural Module Networks - 핵심 내용

**출처**: Jacob Andreas, Marcus Rohrbach, Trevor Darrell, Dan Klein  
**발행**: arXiv:1511.02799, Nov 2015 (최종 수정: Jul 2017)  
**인용**: 1584회  
**분야**: Computer Vision, Natural Language Processing, Machine Learning

## 핵심 개념

### Visual Question Answering의 조합적 본질

**문제 인식**: 시각적 질문 답변(VQA)은 본질적으로 조합적(compositional)

**예시**:
- "where is the dog?" (개가 어디에 있나요?)
- "what color is the dog?" (개는 무슨 색인가요?)
- "where is the cat?" (고양이가 어디에 있나요?)

이 질문들은 **공통 하위구조**를 공유:
- "개 인식" 모듈
- "위치 찾기" 모듈
- "색상 분류" 모듈

### Neural Module Networks (NMN)

**정의**: 공동 학습된 신경 "모듈"들의 컬렉션을 조합하여 질문 답변을 위한 딥 네트워크를 구성하는 방법

**핵심 아이디어**:
1. 질문을 언어적 하위구조로 분해
2. 이 구조를 사용하여 모듈식 네트워크를 동적으로 인스턴스화
3. 재사용 가능한 컴포넌트 (개 인식, 색상 분류 등)
4. 결과 복합 네트워크를 공동 학습

## 주요 기여

### 1. 조합적 구조 활용

**언어 구조 → 네트워크 구조 매핑**:
- 질문의 구문 분석 → 모듈 조합 레이아웃
- 동사/명사 → 특정 모듈 선택
- 의존 관계 → 모듈 연결 방식

### 2. 모듈 재사용성

**모듈 라이브러리**:
- `find[dog]`: 개를 찾는 모듈
- `find[cat]`: 고양이를 찾는 모듈
- `classify[color]`: 색상 분류 모듈
- `locate`: 위치 파악 모듈
- `combine`: 여러 입력 결합 모듈

**장점**:
- 새로운 질문 조합에 즉시 대응
- 학습 데이터 효율성 향상
- 일반화 능력 강화

### 3. 공동 학습 (Joint Training)

**모든 모듈을 동시에 학습**:
- End-to-end 학습
- 모듈 간 상호작용 최적화
- 전체 시스템 성능 극대화

## 아키텍처 상세

### 모듈 타입

#### 1. Attention Modules (어텐션 모듈)

**find[c]**: 개념 c를 찾는 모듈
- 입력: 이미지
- 출력: 어텐션 맵 (개념이 있는 위치)

**relate[r]**: 관계 r에 따라 어텐션 이동
- 입력: 어텐션 맵
- 출력: 변환된 어텐션 맵

#### 2. Re-attention Modules (재어텐션 모듈)

**filter[c]**: 개념 c로 어텐션 필터링
- 입력: 어텐션 맵
- 출력: 정제된 어텐션 맵

#### 3. Combination Modules (조합 모듈)

**and**: 두 어텐션 맵의 교집합
**or**: 두 어텐션 맵의 합집합

#### 4. Measurement Modules (측정 모듈)

**classify[a]**: 속성 a 분류
- 입력: 이미지 + 어텐션 맵
- 출력: 속성 값

**exists**: 객체 존재 여부
- 입력: 어텐션 맵
- 출력: yes/no

**count**: 객체 개수
- 입력: 어텐션 맵
- 출력: 숫자

### 동적 네트워크 인스턴스화

**질문 파싱 → 레이아웃 생성**:

```
질문: "What color is the dog?"

파싱 트리:
  classify[color]
    └─ find[dog]

네트워크 인스턴스화:
  Image → find[dog] → classify[color] → Answer
```

**복잡한 예시**:

```
질문: "Is there a red shape above the blue circle?"

파싱 트리:
  exists
    └─ and
        ├─ find[red]
        └─ relate[above]
            └─ and
                ├─ find[blue]
                └─ find[circle]

네트워크 인스턴스화:
  Image → find[blue] ┐
  Image → find[circle] → and → relate[above] ┐
  Image → find[red] ────────────────────────→ and → exists → Answer
```

## 학습 방법

### 1. 모듈 파라미터 학습

각 모듈은 신경망으로 구현:
- CNN 기반 어텐션 모듈
- MLP 기반 분류 모듈
- 파라미터는 역전파로 학습

### 2. 레이아웃 파서 학습

질문 → 모듈 레이아웃 매핑:
- 구문 파서 사용 (Stanford Parser 등)
- 또는 학습 가능한 레이아웃 생성기

### 3. End-to-End 최적화

전체 시스템을 질문-답변 쌍으로 학습:
- 손실 함수: Cross-entropy (분류), MSE (회귀)
- 최적화: Adam, SGD

## 인지 시드 코어와의 연관성

### 직접적 유사성

본 논문의 NMN과 인지 시드 프레임워크는 놀라울 정도로 유사:

| NMN | 인지 시드 프레임워크 |
|---|---|
| 모듈 (Module) | 시드 (Seed) |
| 모듈 라이브러리 | 32개 시드 카탈로그 |
| 동적 레이아웃 | Seed Router |
| 공동 학습 | 계층적 학습 |
| 재사용성 | 조합 가능성 |

### 코어 설계 시사점

#### 1. 모듈 인터페이스 표준화

**NMN의 교훈**: 모든 모듈이 일관된 입출력 형식

**시드 인터페이스 설계**:
```python
class SeedInterface:
    def forward(self, 
                input_features: Tensor,
                attention_map: Optional[Tensor] = None,
                context: Dict = None) -> Tuple[Tensor, Dict]:
        """
        Args:
            input_features: 입력 특징 (이미지, 텍스트 등)
            attention_map: 선택적 어텐션 맵
            context: 컨텍스트 정보 (메타데이터, 중간 결과 등)
        
        Returns:
            output_features: 출력 특징
            updated_context: 업데이트된 컨텍스트
        """
        pass
```

#### 2. 동적 조합 메커니즘

**NMN의 레이아웃 생성 → 시드 조합 계획**:

```python
class CompositionPlanner:
    def plan(self, task_description: str) -> CompositionGraph:
        # 1. 태스크 분석
        subtasks = self.parse_task(task_description)
        
        # 2. 필요한 시드 선택
        required_seeds = []
        for subtask in subtasks:
            seed = self.seed_registry.query(subtask)
            required_seeds.append(seed)
        
        # 3. 의존성 해결
        graph = self.resolve_dependencies(required_seeds)
        
        # 4. 실행 순서 결정
        execution_order = topological_sort(graph)
        
        return CompositionGraph(execution_order, graph)
```

#### 3. 공동 학습 전략

**NMN의 joint training → 시드 공동 학습**:

```python
class JointTrainer:
    def train_epoch(self, dataloader):
        for batch in dataloader:
            # 1. 태스크별 시드 조합 생성
            composition = self.planner.plan(batch.task)
            
            # 2. Forward pass
            output = self.execute_composition(composition, batch.input)
            
            # 3. 손실 계산
            loss = self.criterion(output, batch.target)
            
            # 4. Backward pass (모든 시드 동시 업데이트)
            loss.backward()
            self.optimizer.step()
```

## 실험 결과 및 성능

### VQA 데이터셋

**State-of-the-art 달성** (2015-2016년 기준):
- 정확도: 58.7% (이전 최고: 57.2%)
- 특히 조합적 질문에서 큰 향상

### SHAPES 데이터셋

**복잡한 추론 질문**:
- 정확도: 96.6% (이전 최고: 92.3%)
- 공간 관계, 논리 연산 등에서 우수

### 일반화 능력

**보지 못한 조합에 대한 강건성**:
- 학습 시 보지 못한 질문 구조에도 답변 가능
- 모듈 재사용으로 zero-shot 일반화

## 한계 및 개선 방향

### 1. 레이아웃 파싱 의존성

**문제**: 정확한 구문 파서 필요
**해결**: End-to-end 레이아웃 학습 (후속 연구)

### 2. 모듈 수 제한

**문제**: 사전 정의된 모듈만 사용 가능
**해결**: 동적 모듈 생성 메커니즘

### 3. 계산 효율성

**문제**: 복잡한 질문은 많은 모듈 필요
**해결**: 모듈 공유, 조기 종료 등

## 인지 시드 코어 구현 권장사항

### 1. 시드 타입 분류

**NMN의 모듈 타입 → 시드 레벨**:
- Attention Modules → Level 0 (Atomic)
- Combination Modules → Level 1 (Molecular)
- Measurement Modules → Level 2 (Cellular)
- Complex Reasoning → Level 3 (Tissue)

### 2. 인터페이스 설계

**일관된 입출력**:
- 모든 시드가 동일한 인터페이스 구현
- 타입 힌팅으로 호환성 보장
- 컨텍스트 전달로 상태 공유

### 3. 조합 그래프

**DAG (Directed Acyclic Graph) 구조**:
- 시드 간 의존성 명시
- 병렬 실행 가능 시드 식별
- 최적 실행 순서 자동 결정

### 4. 학습 전략

**단계별 학습**:
1. 각 시드 독립 사전학습
2. 조합 레이아웃 학습
3. End-to-end 미세조정

## 코드 예시: 시드 조합 실행

```python
class SeedCompositionExecutor:
    def __init__(self, seed_registry):
        self.registry = seed_registry
        self.cache = {}
    
    def execute(self, composition_graph, input_data):
        # 위상 정렬로 실행 순서 결정
        execution_order = composition_graph.topological_sort()
        
        # 중간 결과 저장
        intermediate_results = {}
        
        for seed_node in execution_order:
            # 의존성 확인
            dependencies = composition_graph.get_dependencies(seed_node)
            
            # 입력 준비
            seed_inputs = []
            for dep in dependencies:
                seed_inputs.append(intermediate_results[dep.id])
            
            # 시드 실행
            seed = self.registry.get(seed_node.name)
            output = seed.forward(*seed_inputs, context=input_data.context)
            
            # 결과 저장
            intermediate_results[seed_node.id] = output
            
            # 캐싱 (필요시)
            if seed_node.cacheable:
                cache_key = self.compute_cache_key(seed_node, seed_inputs)
                self.cache[cache_key] = output
        
        # 최종 출력 반환
        final_node = composition_graph.get_output_node()
        return intermediate_results[final_node.id]
```

## 추가 읽을거리

### 후속 연구

- **End-to-End Module Networks** (2016): 레이아웃 학습
- **Learning to Reason** (2017): 복잡한 추론
- **Compositional Attention Networks** (2018): 어텐션 조합

### 관련 프레임워크

- **TorchScript**: PyTorch 동적 그래프
- **ONNX**: 모델 교환 포맷
- **Modular Networks 라이브러리**: 구현 참조

---

**Note**: 이 논문은 1584회 인용된 영향력 있는 연구로, 모듈식 신경망의 조합 및 공동 학습 방법론을 제시하여 인지 시드 코어의 설계 철학에 직접적 영향을 줌.

