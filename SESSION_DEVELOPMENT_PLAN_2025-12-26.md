# Cognitive Seed Framework - 분할 개발 계획 (2025-12-26)

**작성일**: 2025-12-26  
**작성자**: Manus AI  
**목적**: 토큰 효율을 고려한 Level 3 (Tissue) 분할 개발 계획 수립

---

## 1. 프로젝트 현황 (2025-12-26 기준)

### 1.1 전체 진행 상황

| 항목 | 완료 | 전체 | 진행률 | 상태 |
|---|---|---|---|---|
| **시드 구현** | 24 | 32 | 75.0% | 🔄 |
| **프로젝트 버전** | 2.0.0 | - | - | ✅ |
| **현재 Phase** | Phase 3 완료 | - | - | ✅ |

### 1.2 레벨별 진행 상황

#### Level 0 (Atomic) - 100% 완료 ✅
- **시드**: A01~A08 (8개)
- **파일 수**: 9개 Python 파일
- **상태**: 모든 시드 구현 및 테스트 완료

#### Level 1 (Molecular) - 100% 완료 ✅
- **시드**: M01~M08 (8개)
- **파일 수**: 9개 Python 파일
- **상태**: 모든 시드 구현 및 테스트 완료

#### Level 2 (Cellular) - 100% 완료 ✅
- **시드**: C01~C08 (8개)
- **파일 수**: 9개 Python 파일
- **상태**: 모든 시드 구현 및 테스트 완료
- **최근 완성**: C04 Perspective Shifter, C05 Narrative Constructor (2025-12-15)

#### Level 3 (Tissue) - 0% 예정 📅
- **시드**: T01~T08 (8개) 미구현
- **목표**: 2026년 6월 완료

---

## 2. 보안 및 의존성 검사 결과

### 2.1 보안 검사 요약

**검사일**: 2025-12-26  
**보안 등급**: ✅ **우수 (Excellent)**

| 항목 | 상태 | 비고 |
|---|---|---|
| 코드 보안 취약점 | ✅ 없음 | eval() 사용은 PyTorch 정상 사용 |
| 의존성 보안 | ✅ 안전 | 21개 패키지 모두 안전 |
| 민감정보 노출 | ✅ 없음 | .gitignore 적절히 설정 |
| Dependabot | ✅ 활성화 | 11개 PR 생성됨 |
| SECURITY.md | ✅ 존재 | 보안 정책 완비 |
| CodeQL | ❌ 미설정 | 설정 권장 (P1) |
| Secret Scanning | ❌ 미설정 | 활성화 권장 (P1) |

### 2.2 의존성 현황

**총 의존성**: 21개  
**보안 상태**: ✅ 모든 패키지 안전  
**Dependabot PR**: 11개 (미병합)

**핵심 라이브러리**:
- torch ≥2.0.0 → 2.5.1 (최신)
- numpy ≥1.24.0 → 1.26.4 (최신)
- scipy ≥1.10.0 → 1.14.1 (최신)

**결론**: Level 3 개발 진행에 문제 없음

---

## 3. 로드맵 분석 (ROADMAP.md v5.0)

### 3.1 현재 로드맵 구조

**작성일**: 2025-12-26  
**버전**: v5.0

**핵심 Phase**:
- **Phase 4 (진행중)**: Level 3 (Tissue) 구현
- **Phase 5 (진행중)**: 프로젝트 안정화 및 자동화
- **Phase 6 (예정)**: 최종 최적화, 배포 및 연구

**주요 특징**: Phase 4와 Phase 5를 병렬로 진행하여 개발 기간 단축

### 3.2 Phase 4: Level 3 (Tissue) 구현 계획

**전략**: 각 시드 구현을 독립적인 개발 세션으로 분리

| 세션 ID | 대상 시드 | 우선순위 | 복잡도 | 예상 기간 |
|---|---|---|---|---|
| **S4.1** | T01: Abductive Reasoner | **P0** | 높음 | 7-10일 |
| **S4.2** | T04: Strategic Planner | **P0** | 높음 | 7-10일 |
| **S4.3** | T02: Analogical Transfer Engine | P1 | 높음 | 7-10일 |
| **S4.4** | T05: Social Modeler | P1 | 높음 | 7-10일 |
| **S4.5** | T07: Ethical Reasoner | P1 | 높음 | 7-10일 |
| **S4.6** | T03: Theory Builder | P2 | 매우 높음 | 10-14일 |
| **S4.7** | T06: Meta-Learner | P2 | 매우 높음 | 10-14일 |
| **S4.8** | T08: Creative Synthesizer | P2 | 매우 높음 | 10-14일 |
| **S4.9** | Level 3 통합 및 벤치마크 | **P0** | 높음 | 10-14일 |

### 3.3 Phase 5: 프로젝트 안정화 계획

**전략**: Level 3 구현과 병행하여 진행

| 세션 ID | 작업 내용 | 우선순위 | 예상 기간 |
|---|---|---|---|
| **S5.1** | **보안 강화** | **P0** | 1-2일 |
| **S5.2** | **CI/CD 파이프라인 구축** | P1 | 3-5일 |
| **S5.3** | **문서 관리 자동화** | P1 | 2-3일 |
| **S5.4** | **기여 가이드라인 마련** | P2 | 1-2일 |
| **S5.5** | **코드 리팩토링** | P3 | 3-5일 |

---

## 4. 분할 개발 전략

### 4.1 토큰 효율 최적화 원칙

**목표**: 단일 세션의 토큰 사용량을 최소화하고 개발 효율성을 극대화

**핵심 전략**:
1. **세션 분리**: 각 시드 구현을 독립적인 세션으로 분리
2. **우선순위 기반 개발**: P0 → P1 → P2 순서로 개발
3. **병행 개발**: Phase 4 (시드 구현)와 Phase 5 (안정화)를 동시 진행
4. **점진적 통합**: 각 시드 완성 후 즉시 통합 테스트
5. **문서 자동화**: 반복 작업을 스크립트로 자동화

### 4.2 세션별 표준 구조

각 개발 세션은 다음 표준 구조를 따릅니다:

```
세션 시작
  ↓
1. 의존성 확인 (5-10분)
  - 하위 레벨 시드 구현 상태 확인
  - 참고 문서 확인
  ↓
2. 아키텍처 설계 (1-2시간)
  - 시드 설계 문서 작성
  - 클래스 구조 정의
  - 입출력 규격 정의
  ↓
3. 코어 로직 구현 (2-3일)
  - 기본 구조 구현
  - 핵심 로직 구현
  - 의존 시드 통합
  ↓
4. 테스트 작성 (1-2일)
  - 단위 테스트 작성 (최소 5개)
  - 통합 테스트 작성
  - 벤치마크 테스트 추가
  ↓
5. 문서화 (0.5-1일)
  - 구현 완료 보고서 작성
  - README, CHANGELOG 업데이트
  - VERSION 업데이트
  ↓
6. Git 커밋 및 푸시 (10분)
  - 커밋 메시지 작성
  - 푸시 및 PR 생성 (선택)
  ↓
세션 종료
```

### 4.3 세션별 예상 토큰 사용량

| 작업 단계 | 예상 토큰 | 비율 |
|---|---|---|
| 의존성 확인 및 설계 | ~10,000 | 12% |
| 코어 로직 구현 | ~40,000 | 48% |
| 테스트 작성 | ~20,000 | 24% |
| 문서화 | ~8,000 | 10% |
| 디버깅 및 수정 | ~5,000 | 6% |
| **총합** | **~83,000** | **100%** |

**참고**: Level 3 시드는 복잡도가 높아 Level 2 대비 약 20% 더 많은 토큰 사용 예상

---

## 5. 세션 S4.1: T01 Abductive Reasoner 구현 (최우선)

### 5.1 시드 개요

**ID**: T01  
**이름**: Abductive Reasoner  
**카테고리**: Logic  
**핵심 용도**: 최선 설명 추론 (Abductive Reasoning)

**설명**: 주어진 관찰 결과에 대해 가장 그럴듯한 설명을 추론하는 시드입니다. 불완전한 정보로부터 최선의 가설을 생성하고 평가합니다.

### 5.2 의존성 분석

T01은 Level 3 시드로, 하위 레벨 시드들을 조합하여 구현됩니다.

**예상 구성 시드**:
- **M02 Causality Detector**: 인과 관계 추정
- **M08 Conflict Resolver**: 제약 충돌 해소
- **M05 Concept Crystallizer**: 개념 추상화
- **C02 Counterfactual Reasoner**: 반사실 추론 (Level 2)

**의존성 상태**:
- ✅ M02: 완료 (`seeds/molecular/m02_causality_detector.py`)
- ✅ M08: 완료 (`seeds/molecular/m08_conflict_resolver.py`)
- ✅ M05: 완료 (`seeds/molecular/m05_concept_crystallizer.py`)
- ✅ C02: 완료 (`seeds/cellular/c02_counterfactual_reasoner.py`)

**결론**: 모든 의존성 해결됨, 즉시 구현 가능

### 5.3 참고 문서

1. `docs/표준_인지_시드_설계_가이드_v_1.md` - 표준 설계 가이드
2. `docs/CORE_ARCHITECTURE.md` - 코어 아키텍처 가이드
3. `C01_IMPLEMENTATION_COMPLETE.md` - Level 2 구현 사례
4. `M02_IMPLEMENTATION_COMPLETE.md` - M02 구현 사례
5. `M08_IMPLEMENTATION_COMPLETE.md` - M08 구현 사례

### 5.4 아키텍처 설계 (초안)

#### 핵심 컴포넌트

```python
class T01AbductiveReasoner(BaseSeed):
    """
    SEED-T01: Abductive Reasoner
    최선 설명 추론 (Abduction)
    
    조합: M02 (Causality Detector) + M08 (Conflict Resolver) 
          + M05 (Concept Crystallizer) + C02 (Counterfactual Reasoner)
    """
    
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()
        
        # 1. Observation Encoder
        self.observation_encoder = nn.Sequential(...)
        
        # 2. Hypothesis Generator
        self.hypothesis_generator = nn.Sequential(...)
        
        # 3. Causality Detector (M02)
        self.causality_detector = M02CausalityDetector(...)
        
        # 4. Counterfactual Reasoner (C02)
        self.counterfactual_reasoner = C02CounterfactualReasoner(...)
        
        # 5. Concept Crystallizer (M05)
        self.concept_crystallizer = M05ConceptCrystallizer(...)
        
        # 6. Conflict Resolver (M08)
        self.conflict_resolver = M08ConflictResolver(...)
        
        # 7. Explanation Scorer
        self.explanation_scorer = nn.Sequential(...)
        
        # 8. Best Explanation Selector
        self.best_explanation_selector = nn.Sequential(...)
```

#### 핵심 기능

1. **관찰 인코딩**
   - 입력 관찰 데이터 인코딩
   - 관찰 패턴 추출

2. **가설 생성**
   - 다수의 후보 가설 생성
   - 인과 관계 기반 가설 구성 (M02)

3. **반사실 추론**
   - 각 가설에 대한 반사실 시나리오 생성 (C02)
   - 가설 검증

4. **개념 추상화**
   - 가설의 핵심 개념 추출 (M05)
   - 일반화된 설명 생성

5. **충돌 해소**
   - 모순되는 가설 간 충돌 해소 (M08)
   - 일관성 있는 설명 도출

6. **최선 설명 선택**
   - 각 가설의 그럴듯함(plausibility) 점수 계산
   - 최선의 설명 선택 및 반환

### 5.5 입출력 규격

**입력**:
- `observations`: 관찰 데이터 `(batch, num_obs, input_dim)`
- `context`: 맥락 정보 `(batch, context_len, input_dim)` (선택)
- `constraints`: 제약 조건 `(batch, num_constraints, input_dim)` (선택)

**출력**:
- `best_explanation`: 최선의 설명 `(batch, output_dim)`
- `hypotheses`: 생성된 가설들 `(batch, num_hypotheses, hidden_dim)`
- `plausibility_scores`: 각 가설의 그럴듯함 점수 `(batch, num_hypotheses)`
- `causal_graph`: 인과 그래프 `(batch, num_nodes, num_nodes)` (선택)

### 5.6 구현 단계

#### Phase 1: 기본 구조 (2-3일)
1. ✅ 의존 시드 코드 리뷰 (M02, M08, M05, C02)
2. ✅ 아키텍처 설계 및 클래스 구조 정의
3. 🔄 파일 생성: `seeds/tissue/t01_abductive_reasoner.py`
4. 🔄 클래스 정의: `T01AbductiveReasoner` 구현
5. 🔄 의존 시드 통합 및 기본 forward 메서드

#### Phase 2: 핵심 로직 (3-4일)
1. 🔄 Observation Encoder 구현
2. 🔄 Hypothesis Generator 구현
3. 🔄 Causality Detector (M02) 통합
4. 🔄 Counterfactual Reasoner (C02) 통합
5. 🔄 Concept Crystallizer (M05) 통합
6. 🔄 Conflict Resolver (M08) 통합
7. 🔄 Explanation Scorer 구현
8. 🔄 Best Explanation Selector 구현

#### Phase 3: 테스트 및 문서화 (2-3일)
1. 🔄 단위 테스트 작성: `tests/tissue/test_t01_abductive_reasoner.py`
   - 최소 5개 테스트 케이스
   - 추론 시나리오 테스트
   - 가설 생성 검증
   - 메타데이터 검증

2. 🔄 파라미터 분석
   - 총 파라미터 수 계산
   - 목표: ~3.0M (±15%)

3. 🔄 구현 완료 보고서: `T01_IMPLEMENTATION_COMPLETE.md`
   - 아키텍처 설명
   - 파라미터 분석
   - 테스트 결과
   - 사용 예제

4. 🔄 `seeds/tissue/__init__.py` 생성 및 업데이트
   - T01 import 추가
   - load_seed() 함수 업데이트

5. 🔄 버전 업데이트
   - `VERSION`: 2.0.0 → 2.1.0
   - `CHANGELOG.md`: v2.1.0 항목 추가
   - `README.md`: Level 3 진행 상황 업데이트

### 5.7 체크리스트

- [ ] 의존 시드 (M02, M08, M05, C02) 코드 리뷰
- [ ] T01 설계 문서 작성
- [ ] `seeds/tissue/` 디렉토리 생성
- [ ] `seeds/tissue/t01_abductive_reasoner.py` 작성
- [ ] Observation Encoder 구현
- [ ] Hypothesis Generator 구현
- [ ] Causality Detector (M02) 통합
- [ ] Counterfactual Reasoner (C02) 통합
- [ ] Concept Crystallizer (M05) 통합
- [ ] Conflict Resolver (M08) 통합
- [ ] Explanation Scorer 구현
- [ ] Best Explanation Selector 구현
- [ ] `tests/tissue/` 디렉토리 생성
- [ ] `tests/tissue/test_t01_abductive_reasoner.py` 작성
- [ ] 단위 테스트 5개 이상 작성 및 통과
- [ ] 파라미터 수 검증 (~3.0M 목표)
- [ ] `T01_IMPLEMENTATION_COMPLETE.md` 작성
- [ ] `seeds/tissue/__init__.py` 생성
- [ ] `VERSION` 2.1.0으로 업데이트
- [ ] `CHANGELOG.md` 업데이트
- [ ] `README.md` 업데이트
- [ ] Git 커밋 및 푸시

### 5.8 예상 토큰 사용량

| 작업 | 예상 토큰 |
|---|---|
| 의존성 확인 및 설계 | ~12,000 |
| 코드 구현 | ~45,000 |
| 테스트 작성 | ~20,000 |
| 문서화 | ~10,000 |
| 디버깅 | ~8,000 |
| **총합** | **~95,000** |

**참고**: Level 3 첫 번째 시드로 디렉토리 생성 등 초기 설정 작업이 포함되어 토큰 사용량이 다소 높습니다.

### 5.9 산출물

1. `seeds/tissue/t01_abductive_reasoner.py`
2. `seeds/tissue/__init__.py`
3. `tests/tissue/test_t01_abductive_reasoner.py`
4. `T01_IMPLEMENTATION_COMPLETE.md`
5. 업데이트된 `VERSION` (2.1.0)
6. 업데이트된 `CHANGELOG.md`
7. 업데이트된 `README.md`

---

## 6. 세션 S5.1: 보안 강화 (병행 작업)

### 6.1 목표

프로젝트 보안 수준을 향상시키고 자동화된 보안 스캔을 구축합니다.

### 6.2 작업 범위

#### 1. CodeQL 설정 (P0)

**파일**: `.github/workflows/codeql.yml`

**내용**:
- GitHub CodeQL 분석 설정
- Python 코드 보안 스캔
- 주간 자동 실행

**예상 시간**: 30분

#### 2. Secret Scanning 활성화 (P0)

**작업**:
- GitHub 저장소 설정에서 Secret Scanning 활성화
- 민감정보 자동 검출 설정

**예상 시간**: 10분

#### 3. Dependabot PR 검토 및 병합 (P1)

**작업**:
- 11개의 열린 Dependabot PR 검토
- 테스트 통과 확인
- 안전한 PR 병합

**예상 시간**: 1-2시간

#### 4. 보안 정책 업데이트 (P2)

**파일**: `SECURITY.md`

**내용**:
- 최신 보안 감사 이력 추가 (2025-12-26)
- 자동화된 보안 스캔 도구 목록 업데이트

**예상 시간**: 30분

### 6.3 예상 토큰 사용량

| 작업 | 예상 토큰 |
|---|---|
| CodeQL 설정 | ~8,000 |
| Dependabot PR 검토 | ~5,000 |
| 문서 업데이트 | ~3,000 |
| **총합** | **~16,000** |

### 6.4 산출물

1. `.github/workflows/codeql.yml`
2. 업데이트된 `SECURITY.md`
3. 병합된 Dependabot PR (11개)

---

## 7. 전체 개발 일정 및 토큰 예산

### 7.1 Phase 4 세션별 일정

| 세션 | 작업 | 우선순위 | 예상 기간 | 예상 토큰 |
|---|---|---|---|---|
| **S4.1** | T01 Abductive Reasoner | P0 | 7-10일 | ~95,000 |
| **S4.2** | T04 Strategic Planner | P0 | 7-10일 | ~85,000 |
| **S4.3** | T02 Analogical Transfer Engine | P1 | 7-10일 | ~85,000 |
| **S4.4** | T05 Social Modeler | P1 | 7-10일 | ~85,000 |
| **S4.5** | T07 Ethical Reasoner | P1 | 7-10일 | ~85,000 |
| **S4.6** | T03 Theory Builder | P2 | 10-14일 | ~95,000 |
| **S4.7** | T06 Meta-Learner | P2 | 10-14일 | ~95,000 |
| **S4.8** | T08 Creative Synthesizer | P2 | 10-14일 | ~95,000 |
| **S4.9** | Level 3 통합 및 벤치마크 | P0 | 10-14일 | ~70,000 |
| **총합** | **Phase 4 완료** | - | **75-102일** | **~790,000** |

### 7.2 Phase 5 세션별 일정

| 세션 | 작업 | 우선순위 | 예상 기간 | 예상 토큰 |
|---|---|---|---|---|
| **S5.1** | 보안 강화 | P0 | 1-2일 | ~16,000 |
| **S5.2** | CI/CD 파이프라인 구축 | P1 | 3-5일 | ~25,000 |
| **S5.3** | 문서 관리 자동화 | P1 | 2-3일 | ~15,000 |
| **S5.4** | 기여 가이드라인 마련 | P2 | 1-2일 | ~12,000 |
| **S5.5** | 코드 리팩토링 | P3 | 3-5일 | ~20,000 |
| **총합** | **Phase 5 완료** | - | **10-17일** | **~88,000** |

### 7.3 통합 개발 일정 (병행)

| 주차 | Phase 4 작업 | Phase 5 작업 | 총 토큰 |
|---|---|---|---|
| **1-2주** | S4.1: T01 구현 | S5.1: 보안 강화 | ~111,000 |
| **3-4주** | S4.2: T04 구현 | S5.2: CI/CD 구축 | ~110,000 |
| **5-6주** | S4.3: T02 구현 | S5.3: 문서 자동화 | ~100,000 |
| **7-8주** | S4.4: T05 구현 | - | ~85,000 |
| **9-10주** | S4.5: T07 구현 | S5.4: 기여 가이드 | ~97,000 |
| **11-13주**| S4.6: T03 구현 | - | ~95,000 |
| **14-16주**| S4.7: T06 구현 | - | ~95,000 |
| **17-19주**| S4.8: T08 구현 | - | ~95,000 |
| **20-22주**| S4.9: Level 3 통합 | S5.5: 리팩토링 | ~90,000 |
| **총합** | **~790,000** | **~88,000** | **~878,000** |

**예상 완료 시점**: 2026년 6월 (약 5.5개월 후)

### 7.4 마일스톤

| 마일스톤 | 목표 | 예상 완료일 |
|---|---|---|
| **M1: Level 3 핵심 시드 완성** | T01, T04 구현 | 2026-02-10 |
| **M2: Level 3 주요 시드 완성** | T02, T05, T07 구현 | 2026-04-10 |
| **M3: Level 3 전체 완성** | T01-T08 구현 | 2026-05-20 |
| **M4: 프로젝트 최종 완성** | 통합 테스트 및 안정화 | 2026-06-15 |

---

## 8. 즉시 실행 가능한 다음 단계

### 8.1 현재 세션에서 실행 가능한 작업

**세션 S4.1: T01 Abductive Reasoner 구현**을 즉시 시작할 수 있습니다.

#### 준비 완료 사항
- ✅ 의존 시드 (M02, M08, M05, C02) 모두 구현 완료
- ✅ 보안 검사 통과 (보안 등급: 우수)
- ✅ 의존성 검사 통과 (21개 패키지 모두 안전)
- ✅ 프로젝트 구조 확인 완료
- ✅ 로드맵 v5.0 수립 완료

#### 다음 단계
1. 의존 시드 코드 리뷰 (M02, M08, M05, C02)
2. T01 아키텍처 상세 설계
3. `seeds/tissue/` 디렉토리 생성
4. `seeds/tissue/t01_abductive_reasoner.py` 구현 시작
5. 단위 테스트 작성
6. 문서화 및 커밋

### 8.2 병행 작업

**세션 S5.1: 보안 강화**를 T01 구현과 병행하여 진행할 수 있습니다.

#### 작업 내용
1. CodeQL 설정 (30분)
2. Secret Scanning 활성화 (10분)
3. Dependabot PR 검토 및 병합 (1-2시간)
4. SECURITY.md 업데이트 (30분)

### 8.3 권장 실행 순서

**옵션 A: 즉시 개발 시작 (권장)**
1. S4.1: T01 Abductive Reasoner 구현 시작
2. S5.1: 보안 강화 (병행)

**옵션 B: 보안 강화 우선**
1. S5.1: 보안 강화 완료
2. S4.1: T01 Abductive Reasoner 구현 시작

**옵션 C: 전체 계획 검토 후 진행**
1. 분할 개발 계획 검토 및 수정
2. 사용자 승인 후 개발 시작

---

## 9. 결론

### 9.1 준비 상태

**✅ Level 3 개발 즉시 시작 가능**

프로젝트는 다음과 같은 이유로 Level 3 개발을 즉시 시작할 수 있는 최적의 상태입니다:

1. **보안 우수**: 코드 보안 취약점 없음, 의존성 안전
2. **의존성 해결**: 모든 하위 레벨 시드 구현 완료
3. **로드맵 명확**: v5.0 로드맵 수립 완료
4. **분할 계획 수립**: 토큰 효율적인 세션 분할 완료
5. **문서 완비**: 참고 문서 및 가이드 충분

### 9.2 핵심 성공 요인

1. **세션 분리**: 각 시드를 독립적인 세션으로 분리하여 토큰 효율 극대화
2. **병행 개발**: Phase 4와 Phase 5를 동시 진행하여 개발 기간 단축
3. **우선순위 기반**: P0 → P1 → P2 순서로 체계적 개발
4. **점진적 통합**: 각 시드 완성 후 즉시 통합 테스트
5. **문서 자동화**: 반복 작업 자동화로 효율성 향상

### 9.3 예상 성과

**2026년 6월 완료 시**:
- ✅ 32개 인지 시드 전체 완성
- ✅ 세계 최초의 완전한 모듈식 인지 프레임워크 구축
- ✅ 보안 및 CI/CD 완비
- ✅ 프로젝트 버전 3.0.0 릴리스

---

**작성일**: 2025-12-26  
**작성자**: Manus AI  
**다음 세션**: S4.1 (T01 Abductive Reasoner 구현) 또는 S5.1 (보안 강화)
