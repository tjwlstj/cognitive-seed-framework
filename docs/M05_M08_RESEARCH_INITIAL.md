# M05 & M08 연구 자료 초기 수집

**작성일**: 2025-11-02  
**작성자**: Manus AI

---

## M05: Concept Crystallizer 연구 방향

### 핵심 개념: Prototypical Networks

**주요 논문**: Snell et al., "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
- **인용 수**: 11,956회
- **핵심 아이디어**: Few-shot 분류 문제에서 각 클래스의 프로토타입 표현을 학습하여 거리 기반 분류 수행
- **장점**: 간단한 inductive bias로 limited-data regime에서 우수한 성능 달성

### 관련 연구

1. **Improved Prototypical Networks** (Ji et al., 2020, 인용 136회)
   - 기존 Prototypical Networks의 개선 버전

2. **Deep Meta-Learning** (Zhou et al., 2018, 인용 177회)
   - Concept space에서의 meta-learning

3. **Meta-learning in Neural Networks Survey** (Hospedales et al., 2021, 인용 3,262회)
   - Meta-learning 전반에 대한 포괄적 서베이

### M05 구현 방향

**구성 시드**: A05 (Grouping Nucleus) + M03 (Pattern Completer) + M01 (Hierarchy Builder)

**핵심 기능**:
- 프로토타입 학습 및 추상화
- Few-shot 학습 능력
- 개념 결정화 (Concept Crystallization)

---

## M08: Conflict Resolver 연구 방향

### 핵심 개념: Constraint Satisfaction & Conflict Resolution

**주요 연구 영역**:

1. **Constraint Satisfaction Problems (CSP) with Neural Networks**
   - Graph Neural Networks for Maximum Constraint Satisfaction (Tönshoff et al., 2021, 인용 80회)
   - DeepSaDe: Learning Neural Networks that Guarantee Constraints (Goyal et al., 2023, 인용 7회)

2. **Conflict Resolution with Machine Learning**
   - Machine Learning for Intelligent Support of Conflict Resolution (Sycara, 1993, 인용 137회)
   - A Machine Learning Approach for Conflict Resolution in Dense Traffic Scenarios (Pham et al., 2019, 인용 69회)

3. **Multi-Objective Optimization**
   - Multi-Task Learning as Multi-Objective Optimization (Sener et al., NeurIPS, 인용 1,840회)
   - Multi-objective Deep Learning Survey (2024)

### M08 구현 방향

**구성 시드**: A08 (Binary Comparator) + M06 (Context Integrator) + M02 (Causality Detector)

**핵심 기능**:
- 제약 충돌 감지
- 다목적 최적화
- 충돌 해소 전략 학습

---

## 다음 단계

1. Prototypical Networks 논문 PDF 상세 분석
2. Constraint Satisfaction 관련 논문 수집 및 분석
3. M05, M08 구현 가이드 작성
4. 아키텍처 설계 및 구현

---

**참고 URL**:
- Prototypical Networks: https://arxiv.org/abs/1703.05175
- Graph Neural Networks for CSP: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.580607/full
- Multi-Task Learning as MOO: http://papers.neurips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf
