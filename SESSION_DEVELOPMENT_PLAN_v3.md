# Cognitive Seed Framework - 분할 개발 계획 v3.0

**작성일**: 2025-11-29  
**작성자**: Manus AI  
**목적**: 최신 프로젝트 현황 반영 및 토큰 효율적 분할 개발 계획 수립

---

## 1. 프로젝트 현황 종합 분석 (2025-11-29 기준)

### 1.1 전체 진행 상황

| 항목 | 완료 | 전체 | 진행률 | 상태 |
|---|---|---|---|---|
| **시드 구현** | 17 | 32 | 53.1% | 🔄 |
| **파라미터** | ~7.46M | ~19.69M | 37.9% | 🔄 |
| **Phase** | 2.1 | 6 | 35.0% | 🔄 |

### 1.2 레벨별 상세 현황

#### Level 0 (Atomic) - 100% 완료 ✅
- **시드**: A01~A08 (8개)
- **파라미터**: ~1.09M
- **상태**: 모든 시드 구현 및 테스트 완료

#### Level 1 (Molecular) - 100% 완료 ✅
| ID | Name | 상태 | 파라미터 | 비고 |
|---|---|---|---|---|
| M01 | Hierarchy Builder | ✅ | ~426K | 완료 |
| M02 | Causality Detector | ✅ | ~600K | 완료 |
| M03 | Pattern Completer | ✅ | ~550K | 완료 |
| M04 | Spatial Transformer | ✅ | ~450K | 완료 |
| M05 | Concept Crystallizer | ✅ | ~660K | 완료 |
| M06 | Context Integrator | ✅ | ~2,092K | 완료 |
| M07 | Analogy Mapper | ✅ | ~750K | 완료 |
| M08 | Conflict Resolver | ✅ | ~800K | **완료** |

**완료 파라미터**: ~6.37M / ~6.37M (100%)

#### Level 2 (Cellular) - 12.5% 진행 중 🟡
| ID | Name | 상태 | 구성 시드 | 비고 |
|---|---|---|---|---|
| C01 | Metaphor Engine | ✅ | M01+M07+M05 | 2025-11-26 완료 |
| C02 | Counterfactual Reasoner | ❌ | M02+M08+A08 | 예정 |
| C03 | Schema Learner | ❌ | M01+M05+A05 | 예정 |
| C04 | Perspective Shifter | ❌ | M04+M07+A02 | 예정 |
| C05 | Narrative Constructor | ❌ | M06+M03+A06 | 예정 |
| C06 | Attention Director | ❌ | M06+M01+A05 | 예정 |
| C07 | Boundary Detector | ❌ | A01+M03+M06 | 예정 |
| C08 | Novelty Assessor | ❌ | M05+M07+A04 | 예정 |

#### Level 3 (Tissue) - 0% 예정 📅
- T01~T08 미구현

---

## 2. 보안 및 의존성 검사 결과 (2025-11-29)

### 2.1 보안 검사 (Bandit)
- **실행일**: 2025-11-29
- **검사 코드**: 4,503 LOC
- **검사 결과**: ✅ **0개 이슈 발견**
  - HIGH: 0
  - MEDIUM: 0
  - LOW: 0
- **결론**: 코드베이스는 보안 취약점 없음

### 2.2 의존성 보안 검사 (pip-audit)
- **실행일**: 2025-11-29
- **검사 대상**: requirements.txt (19개 패키지)
- **검사 결과**: ✅ **알려진 취약점 없음** (pip, setuptools 업그레이드 완료)
- **결론**: 모든 의존성 패키지가 안전한 버전 사용

### 2.3 의존성 버전 현황
| 패키지 | 최소 버전 | 상태 |
|---|---|---|
| torch | ≥2.0.0 | ✅ 안전 |
| numpy | ≥1.24.0 | ✅ 안전 |
| scipy | ≥1.10.0 | ✅ 안전 |
| geoopt | ≥0.5.0 | ✅ 안전 |
| scikit-learn | ≥1.3.0 | ✅ 안전 |

### 2.4 GitHub 저장소 보안 설정

| 항목 | 현재 상태 | 권장 사항 |
|---|---|---|
| 보안 정책 (SECURITY.md) | ❌ 미설정 | 추가 필요 (P1) |
| 취약점 알림 | 확인 불가 | 활성화 권장 |
| Dependabot | 확인 불가 | 활성화 권장 |
| CodeQL 스캔 | ❌ 미설정 | 추가 권장 (P2) |
| CI/CD 파이프라인 | ❌ 미설정 | 추가 필요 (P1) |

---

## 3. 로드맵 및 개발 우선순위

### 3.1 현재 Phase (Phase 3)
**목표**: Level 2 (Cellular) 완성 (8개 시드)  
**현재 상태**: 1/8 완료 (12.5%)  
**남은 작업**: C02~C08 구현 (7개 시드)

### 3.2 주요 발견 사항

#### ✅ Level 1 완성 확인
- **M01~M08**: 전체 구현 완료
- **VERSION**: 1.3.0 (최신)
- **CHANGELOG**: v1.3.0에 M07, M08, C01 구현 기록됨
- **상태**: Level 1 완전 완성 ✅

#### 🟡 Level 2 진행 중
- **C01 Metaphor Engine**: 완료 (2025-11-26)
- **C02~C08**: 미구현 (7개 시드)
- **의존성**: Level 0-1 시드만 필요하므로 병렬 개발 가능

### 3.3 개발 우선순위 재조정

#### P0 (최우선 - 즉시 실행 가능)
Level 2 시드는 모두 Level 0-1 시드에만 의존하므로 **병렬 개발 가능**합니다. 다음 우선순위로 개발을 권장합니다:

| 우선순위 | 시드 ID | 시드 이름 | 구성 시드 | 예상 기간 | 이유 |
|---|---|---|---|---|---|
| **P0-1** | C03 | Schema Learner | M01+M05+A05 | 5-7일 | 추상화 핵심 기능 |
| **P0-2** | C02 | Counterfactual Reasoner | M02+M08+A08 | 5-7일 | 논리 추론 핵심 |
| **P0-3** | C06 | Attention Director | M06+M01+A05 | 5-7일 | 조합 핵심 기능 |

#### P1 (높음 - P0 완료 후)
| 우선순위 | 시드 ID | 시드 이름 | 구성 시드 | 예상 기간 |
|---|---|---|---|---|
| P1-1 | C07 | Boundary Detector | A01+M03+M06 | 5-7일 |
| P1-2 | C08 | Novelty Assessor | M05+M07+A04 | 5-7일 |

#### P2 (중간 - P1 완료 후)
| 우선순위 | 시드 ID | 시드 이름 | 구성 시드 | 예상 기간 |
|---|---|---|---|---|
| P2-1 | C04 | Perspective Shifter | M04+M07+A02 | 5-7일 |
| P2-2 | C05 | Narrative Constructor | M06+M03+A06 | 5-7일 |

#### P3 (Level 2 완료 후)
- **Level 2 통합 및 벤치마크**: 7-10일
- **보안 강화 및 CI/CD**: 5-7일

---

## 4. 분할 개발 계획 (토큰 효율 최적화)

### 4.1 세션 구조 개요

```
세션 1: C03 Schema Learner 구현 (P0-1)
   ↓
세션 2: C02 Counterfactual Reasoner 구현 (P0-2)
   ↓
세션 3: C06 Attention Director 구현 (P0-3)
   ↓
세션 4: C07 Boundary Detector 구현 (P1-1)
   ↓
세션 5: C08 Novelty Assessor 구현 (P1-2)
   ↓
세션 6: C04 Perspective Shifter 구현 (P2-1)
   ↓
세션 7: C05 Narrative Constructor 구현 (P2-2)
   ↓
세션 8: Level 2 통합 및 벤치마크 (P3)
   ↓
세션 9: 보안 강화 및 CI/CD (P3)
```

---

## 5. 세션 1: C03 Schema Learner 구현 (최우선)

### 5.1 목표
M01, M05, A05를 조합하여 스키마 구조 학습 시드 구현

### 5.2 의존성 확인
- **M01 Hierarchy Builder**: ✅ 완료 (`seeds/molecular/m01_hierarchy_builder.py`)
- **M05 Concept Crystallizer**: ✅ 완료 (`seeds/molecular/m05_concept_crystallizer.py`)
- **A05 Grouping Nucleus**: ✅ 완료 (`seeds/atomic/a05_grouping_nucleus.py`)

**결론**: 모든 의존성 해결됨, 즉시 구현 가능

### 5.3 참고 문서
1. `docs/CORE_ARCHITECTURE.md` - 코어 아키텍처 가이드
2. `docs/표준_인지_시드_설계_가이드_v_1.md` - 표준 설계 가이드
3. `C01_IMPLEMENTATION_COMPLETE.md` - C01 구현 사례 (최신)
4. `M05_IMPLEMENTATION_COMPLETE.md` - M05 구현 사례
5. `M01_IMPLEMENTATION_COMPLETE.md` - M01 구현 사례

### 5.4 아키텍처 설계

#### 핵심 컴포넌트
```python
class C03SchemaLearner(BaseSeed):
    """
    SEED-C03: Schema Learner
    스키마 구조 학습 및 추상화
    
    조합: M01 (Hierarchy Builder) + M05 (Concept Crystallizer) + A05 (Grouping Nucleus)
    """
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        # 1. Pattern Encoder
        self.pattern_encoder = nn.Sequential(...)
        
        # 2. Grouping Module (A05 활용)
        self.grouping_module = A05GroupingNucleus(...)
        
        # 3. Concept Crystallizer (M05 활용)
        self.concept_crystallizer = M05ConceptCrystallizer(...)
        
        # 4. Hierarchy Builder (M01 활용)
        self.hierarchy_builder = M01HierarchyBuilder(...)
        
        # 5. Schema Generator
        self.schema_generator = nn.Sequential(...)
```

#### 핵심 기능
1. **패턴 인식 및 그룹화**
   - 입력 패턴 인코딩
   - 유사 패턴 그룹화 (A05)

2. **개념 추출**
   - 그룹별 프로토타입 학습 (M05)
   - 개념 정제 및 일반화

3. **계층 구조 학습**
   - 개념 간 상하 관계 구축 (M01)
   - 스키마 트리 생성

4. **스키마 생성**
   - 추상화된 스키마 표현 생성
   - 구조적 규칙 추출

### 5.5 입출력 규격

**입력**:
- `patterns`: 패턴 텐서 `(batch, num_patterns, input_dim)`
- `context`: 맥락 정보 `(batch, context_len, input_dim)` (선택)

**출력**:
- `schema`: 스키마 표현 `(batch, output_dim)`
- `hierarchy`: 계층 구조 `(batch, num_levels, hidden_dim)`
- `concepts`: 추출된 개념 `(batch, num_concepts, hidden_dim)`

### 5.6 구현 단계

#### Phase 1: 기본 구조 (2-3일)
1. ✅ 의존 시드 코드 리뷰 (M01, M05, A05)
2. ✅ 아키텍처 설계 및 클래스 구조 정의
3. 🔄 파일 생성: `seeds/cellular/c03_schema_learner.py`
4. 🔄 클래스 정의: `C03SchemaLearner` 구현
5. 🔄 의존 시드 통합 및 기본 forward 메서드

#### Phase 2: 핵심 로직 (2-3일)
1. 🔄 Pattern Encoder 구현
2. 🔄 Grouping Module (A05 기반)
3. 🔄 Concept Crystallizer (M05 기반)
4. 🔄 Hierarchy Builder (M01 기반)
5. 🔄 Schema Generator 구현

#### Phase 3: 테스트 및 문서화 (1-2일)
1. 🔄 단위 테스트 작성: `tests/cellular/test_c03_schema_learner.py`
   - 최소 5개 테스트 케이스
   - 스키마 학습 시나리오 테스트
   - 계층 구조 검증
   - 메타데이터 검증

2. 🔄 파라미터 분석
   - 총 파라미터 수 계산
   - 목표: ~1.5M (±10%)

3. 🔄 구현 완료 보고서: `C03_IMPLEMENTATION_COMPLETE.md`
   - 아키텍처 설명
   - 파라미터 분석
   - 테스트 결과
   - 사용 예제

4. 🔄 `seeds/cellular/__init__.py` 업데이트
   - C03 import 추가
   - load_seed() 함수 업데이트

### 5.7 체크리스트

- [ ] 의존 시드 (M01, M05, A05) 코드 리뷰
- [ ] `seeds/cellular/c03_schema_learner.py` 작성
- [ ] Pattern Encoder 구현
- [ ] Grouping Module 구현 (A05 통합)
- [ ] Concept Crystallizer 구현 (M05 통합)
- [ ] Hierarchy Builder 구현 (M01 통합)
- [ ] Schema Generator 구현
- [ ] `tests/cellular/test_c03_schema_learner.py` 작성
- [ ] 단위 테스트 5개 이상 작성 및 통과
- [ ] 파라미터 수 검증 (~1.5M 목표)
- [ ] `C03_IMPLEMENTATION_COMPLETE.md` 작성
- [ ] `seeds/cellular/__init__.py` 업데이트
- [ ] Git 커밋 및 푸시

### 5.8 예상 토큰 사용량

| 작업 | 예상 토큰 |
|---|---|
| 코드 구현 | ~35,000 |
| 테스트 작성 | ~15,000 |
| 문서화 | ~10,000 |
| 디버깅 | ~10,000 |
| **총합** | **~70,000** |

### 5.9 산출물

1. `seeds/cellular/c03_schema_learner.py`
2. `tests/cellular/test_c03_schema_learner.py`
3. `C03_IMPLEMENTATION_COMPLETE.md`
4. 업데이트된 `seeds/cellular/__init__.py`

---

## 6. 세션 2-7: 나머지 Level 2 시드 구현

각 세션은 세션 1과 동일한 구조를 따릅니다:
- 의존성 확인
- 아키텍처 설계
- 구현 (기본 구조 → 핵심 로직 → 테스트)
- 문서화
- Git 커밋

**예상 토큰**: 각 세션당 ~60,000-70,000 토큰

---

## 7. 세션 8: Level 2 통합 및 벤치마크

### 7.1 목표
Level 2 전체 시드 (C01~C08) 통합 테스트 및 벤치마크 구축

### 7.2 의존성
- C01~C08 전체 완료 필요 (세션 1-7 완료 후)

### 7.3 구현 범위

#### 1. 통합 테스트
**파일**: `tests/test_level2_integration.py`

1. **전체 시드 로드 테스트**
   - C01~C08 전체 로드 검증
   - 메타데이터 일관성 검증

2. **조합 패턴 테스트**
   - 시드 간 조합 실행
   - DAG 실행 순서 검증

3. **성능 프로파일링**
   - 각 시드별 실행 시간 측정
   - 메모리 사용량 분석

#### 2. 벤치마크 구축
**파일**: `benchmarks/level2_benchmark.py`

1. **데이터셋 준비**
   - 합성 데이터 생성
   - 실제 태스크 데이터 준비

2. **평가 메트릭**
   - Few-shot 학습 지표
   - Latency (ms)

3. **수용 기준 검증**
   - Few-shot 지표 ≥ 0.80
   - Latency < 100ms

4. **결과 저장**
   - JSON 형식으로 결과 저장
   - 시각화 (matplotlib)

#### 3. 문서 업데이트

1. **README.md**
   - Level 2 완성 표시
   - 벤치마크 결과 추가

2. **CHANGELOG.md**
   - v1.4.0 변경 사항 기록
   - C02~C08 추가 명시

3. **ROADMAP.md**
   - Phase 3 완료 표시
   - Phase 4 계획 업데이트

### 7.4 예상 토큰 사용량

| 작업 | 예상 토큰 |
|---|---|
| 통합 테스트 | ~25,000 |
| 벤치마크 | ~30,000 |
| 문서 업데이트 | ~10,000 |
| **총합** | **~65,000** |

---

## 8. 세션 9: 보안 강화 및 CI/CD

### 8.1 목표
보안 정책 수립 및 CI/CD 파이프라인 구축

### 8.2 구현 범위

#### 1. 보안 정책
**파일**: `SECURITY.md`

- 지원 버전 명시
- 취약점 보고 절차
- 보안 연락처 정보

#### 2. CI/CD 파이프라인
**파일**: `.github/workflows/ci.yml`

- 자동 테스트 실행
- 보안 스캔 (Bandit, pip-audit)
- 코드 품질 검사

#### 3. 커뮤니티 문서

1. **CONTRIBUTING.md**
   - 기여 가이드라인
   - 코드 스타일 가이드
   - PR 프로세스

2. **CODE_OF_CONDUCT.md**
   - 행동 강령
   - 커뮤니티 규칙

3. **Issue/PR 템플릿**
   - `.github/ISSUE_TEMPLATE/bug_report.md`
   - `.github/ISSUE_TEMPLATE/feature_request.md`
   - `.github/PULL_REQUEST_TEMPLATE.md`

### 8.3 예상 토큰 사용량

| 작업 | 예상 토큰 |
|---|---|
| 보안 정책 | ~10,000 |
| CI/CD | ~15,000 |
| 커뮤니티 문서 | ~10,000 |
| **총합** | **~35,000** |

---

## 9. 전체 개발 일정 및 토큰 예산

### 9.1 세션별 일정

| 세션 | 작업 | 예상 기간 | 예상 토큰 |
|---|---|---|---|
| 세션 1 | C03 Schema Learner | 5-7일 | ~70,000 |
| 세션 2 | C02 Counterfactual Reasoner | 5-7일 | ~70,000 |
| 세션 3 | C06 Attention Director | 5-7일 | ~70,000 |
| 세션 4 | C07 Boundary Detector | 5-7일 | ~65,000 |
| 세션 5 | C08 Novelty Assessor | 5-7일 | ~65,000 |
| 세션 6 | C04 Perspective Shifter | 5-7일 | ~65,000 |
| 세션 7 | C05 Narrative Constructor | 5-7일 | ~65,000 |
| 세션 8 | Level 2 통합 및 벤치마크 | 7-10일 | ~65,000 |
| 세션 9 | 보안 강화 및 CI/CD | 5-7일 | ~35,000 |
| **총합** | **Phase 3 완료** | **47-63일** | **~570,000** |

### 9.2 마일스톤

| 마일스톤 | 목표 | 예상 완료일 |
|---|---|---|
| **M1: Level 2 핵심 시드 완성** | C02, C03, C06 구현 | 2026-01-15 |
| **M2: Level 2 전체 완성** | C01-C08 구현 | 2026-02-15 |
| **M3: Level 2 안정화** | 통합 테스트 및 벤치마크 | 2026-02-28 |
| **M4: 프로젝트 강화** | 보안 및 CI/CD | 2026-03-15 |

---

## 10. 즉시 실행 가능한 다음 단계

### 10.1 현재 세션에서 실행 가능한 작업

**세션 1: C03 Schema Learner 구현**을 즉시 시작할 수 있습니다.

#### 준비 완료 사항
- ✅ 의존 시드 (M01, M05, A05) 모두 구현 완료
- ✅ 보안 검사 통과
- ✅ 의존성 검사 통과
- ✅ 프로젝트 구조 확인 완료

#### 다음 단계
1. 의존 시드 코드 리뷰 (M01, M05, A05)
2. C03 아키텍처 상세 설계
3. `seeds/cellular/c03_schema_learner.py` 구현 시작
4. 단위 테스트 작성
5. 문서화 및 커밋

### 10.2 권장 실행 순서

사용자의 요청에 따라 다음 중 하나를 선택할 수 있습니다:

**옵션 A: 즉시 개발 시작**
- 세션 1 (C03 Schema Learner) 구현 시작

**옵션 B: 계획 검토 후 진행**
- 분할 개발 계획 검토 및 피드백
- 우선순위 조정 (필요시)
- 세션 1 구현 시작

**옵션 C: 다른 시드 우선 구현**
- C02 또는 C06 등 다른 시드 선택
- 해당 시드 구현 계획 수립
- 구현 시작

---

## 11. 결론

본 계획은 Cognitive Seed Framework 프로젝트의 최신 현황(2025-11-29)을 반영하여 작성되었습니다. Level 1이 완전히 완성되었으므로, Level 2 (Cellular) 시드 7개를 순차적으로 구현하는 것이 다음 목표입니다.

각 세션은 독립적으로 실행 가능하며, 토큰 효율을 고려하여 설계되었습니다. 모든 의존성이 해결되었으므로 즉시 개발을 시작할 수 있습니다.

**다음 단계**: 사용자의 선택에 따라 세션 1 (C03 Schema Learner) 또는 다른 우선순위 시드 구현을 시작합니다.

---

**작성일**: 2025-11-29  
**버전**: 3.0  
**상태**: 즉시 실행 가능 ✅
