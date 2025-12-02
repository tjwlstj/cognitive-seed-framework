# Cognitive Seed Framework - 프로젝트 로드맵 (v4.0)

**버전**: 4.0  
**작성일**: 2025-12-02  
**작성자**: Manus AI  
**GitHub**: https://github.com/tjwlstj/cognitive-seed-framework

---

## 1. 로드맵 개요

### 1.1. 프로젝트 비전

32개의 표준 인지 시드를 완전히 구현하고, 동적 조합을 통해 복잡한 AI 태스크를 해결하는 **세계 최초의 모듈식 인지 프레임워크**를 구축합니다.

### 1.2. 핵심 목표 (v4.0)

- **완전성**: 32개 시드 전체 구현 (Level 0-3)
- **안정성**: Level 2, 3 통합 벤치마크 구축 및 프로젝트 안정화
- **효율성**: 토큰 최적화를 위한 **소규모 분산 개발 세션** 적용
- **체계성**: 각 세션의 산출물을 명확히 하고, 진행 상황을 체계적으로 관리
- **자동화**: CI/CD 파이프라인 및 문서 관리 자동화 도입

### 1.3. 로드맵 구조 (v4.0)

본 로드맵은 기존 v3.0을 계승하고, 최신 구현 상태(C03 완료)를 반영하여 최신화되었습니다. 특히, 사용자 요구사항에 따라 개발 과정을 **토큰 효율적인 소규모 세션**으로 분산하고, **선택적 집중**이 가능하도록 체계화했습니다.

- **Phase 1**: Level 0-1 기반 구축 (완료 ✅)
- **Phase 2**: Level 1 완성 및 Level 2 시작 (완료 ✅)
- **Phase 3**: Level 2 (Cellular) 완성 (진행 중 🟡)
- **Phase 4**: 프로젝트 안정화 및 자동화 (예정 📅)
- **Phase 5**: Level 3 (Tissue) 구현 (예정 📅)
- **Phase 6**: 최적화, 배포 및 연구 (예정 📅)

---

## 2. 현재 상태 (2025-12-02 기준)

### 2.1. 전체 진행 상황

| 항목 | 완료 | 전체 | 진행률 |
|---|---|---|---|
| **시드 구현** | 18 | 32 | 56.3% |
| **Phase** | 2.2 | 6 | 36.7% |

### 2.2. 레벨별 진행 상황

#### Level 0 (Atomic) - 100% 완료 ✅
- **시드**: A01~A08 (8개) 전체 구현 및 테스트 완료

#### Level 1 (Molecular) - 100% 완료 ✅
- **시드**: M01~M08 (8개) 전체 구현 및 테스트 완료
- **벤치마크**: `benchmarks/level1_benchmark.py` 구축 완료

#### Level 2 (Cellular) - 25.0% 진행 중 🟡
| ID | Name | 상태 | 구성 시드 | 비고 |
|---|---|---|---|---|
| C01 | Metaphor Engine | ✅ | M01+M07+M05 | 2025-11-26 완료 |
| C02 | Counterfactual Reasoner | ❌ | M02+M08+A08 | 예정 |
| C03 | Schema Learner | ✅ | M01+M05+A05 | 2025-12-01 완료 |
| C04 | Perspective Shifter | ❌ | M04+M07+A02 | 예정 |
| C05 | Narrative Constructor | ❌ | M06+M03+A06 | 예정 |
| C06 | Attention Director | ❌ | M06+M01+A05 | 예정 |
| C07 | Boundary Detector | ❌ | A01+M03+M06 | 예정 |
| C08 | Novelty Assessor | ❌ | M05+M07+A04 | 예정 |

#### Level 3 (Tissue) - 0% 예정 📅
- T01~T08 미구현

---

## 3. Phase 3: Level 2 (Cellular) 완성 (진행 중 🟡)

**목표**: Level 2 시드 6개 추가 구현 및 Level 2 통합 벤치마크 구축

### 3.1. 개발 전략: 소규모 분산 세션 (Selective Focus)

- 각 시드 구현을 독립적인 **개발 세션**으로 분리하여 토큰 사용량을 최적화하고, 선택적 집중 개발을 지원합니다.
- 각 세션은 **코드 구현, 단위 테스트, 문서 작성**을 포함하는 완결된 단위입니다.
- Level 2 시드는 Level 0-1 시드에만 의존하므로, 모든 잔여 시드는 **병렬 개발이 가능**합니다.

### 3.2. Level 2 개발 세션 계획

아래는 권장 개발 순서이며, 필요에 따라 순서를 조정하거나 동시 진행할 수 있습니다.

| 세션 | 대상 시드 | 우선순위 | 예상 기간 | 주요 의존성 |
|---|---|---|---|---|
| **S3.1** | C02 Counterfactual Reasoner | **P0** | 5-7일 | M02, M08, A08 |
| **S3.2** | C06 Attention Director | P1 | 5-7일 | M06, M01, A05 |
| **S3.3** | C07 Boundary Detector | P1 | 5-7일 | A01, M03, M06 |
| **S3.4** | C08 Novelty Assessor | P2 | 5-7일 | M05, M07, A04 |
| **S3.5** | C04 Perspective Shifter | P2 | 5-7일 | M04, M07, A02 |
| **S3.6** | C05 Narrative Constructor | P3 | 5-7일 | M06, M03, A06 |
| **S3.7** | Level 2 통합 및 벤치마크 | **P0** | 7-10일 | C01-C08 완료 |

### 3.3. 세션별 산출물 가이드

각 개발 세션(S3.1 ~ S3.6)은 다음 산출물을 생성해야 합니다.

1.  **시드 구현 코드**: `seeds/cellular/<seed_name>.py`
2.  **단위 테스트 코드**: `tests/cellular/test_<seed_name>.py` (최소 10개 케이스)
3.  **구현 완료 보고서**: `<SEED_ID>_IMPLEMENTATION_COMPLETE.md`
4.  **문서 업데이트**: `CHANGELOG.md`, `ROADMAP.md`, `VERSION` 파일 최신화
5.  **Git 커밋**: `feat: Implement <SEED_ID> <Seed_Name>` 형식

---

## 4. Phase 4: 프로젝트 안정화 및 자동화 (예정 📅)

**목표**: 프로젝트의 장기적인 유지보수성, 보안, 사용성을 강화하고, 문서 관리 프로세스를 자동화합니다.

| 세션 | 작업 내용 | 우선순위 | 비고 |
|---|---|---|---|
| **S4.1** | **보안 강화** | **P0** | SECURITY.md 작성, Dependabot/CodeQL 설정 |
| **S4.2** | **CI/CD 파이프라인 구축** | P1 | GitHub Actions 기반 자동 테스트 및 빌드 |
| **S4.3** | **문서 관리 자동화** | P1 | 버전/로드맵/상태 보고서 자동 업데이트 스크립트 |
| **S4.4** | **기여 가이드라인 마련** | P2 | CONTRIBUTING.md, Issue/PR 템플릿 작성 |
| **S4.5** | **코드 리팩토링** | P3 | 기술 부채 해소 및 코드 품질 개선 |

---

## 5. Phase 5: Level 3 (Tissue) 구현 (예정 📅)

**목표**: Level 3 시드 8개 전체 구현 및 Level 3 통합 벤치마크 구축

- Phase 4 완료 후, Level 2 시드와의 의존성을 분석하여 세부 개발 세션을 계획합니다.
- Level 2와 동일하게 소규모 분산 세션 전략을 유지합니다.

| ID | Name | Category | 핵심 용도 |
|---|---|---|---|
| T01 | Abductive Reasoner | Logic | 최선 설명 추론 |
| T02 | Analogical Transfer Engine | Analogy | 구조 전이·적응 |
| T03 | Theory Builder | Abstraction | 이론화 |
| T04 | Strategic Planner | Composition | 목표 분해·계획 |
| T05 | Social Modeler | Relation | 신념/욕구/의도 추론 |
| T06 | Meta-Learner | Abstraction | 메타학습·신속 적응 |
| T07 | Ethical Reasoner | Logic | 윤리 판단 |
| T08 | Creative Synthesizer | Composition | 창의적 결합 |

---

## 6. Phase 6: 최적화, 배포 및 연구 (예정 📅)

**목표**: 프레임워크의 성능을 극대화하고, 실제 적용 사례를 제시하며, 연구를 확장합니다.

- **양자화 (Quantization)**: FP8, INT8 등 양자화 기법 적용 및 성능 평가
- **백본 네트워크 통합**: 다양한 백본(e.g., ResNet, ViT)과의 통합 예제 제공
- **배포 자동화**: PyPI 패키지 배포 자동화
- **허깅페이스 허브 연동**: 모델 및 데모를 허깅페이스 허브에 공개
- **신규 시드 연구**: 신경과학 등 최신 연구 기반의 신규 시드 설계

---

## 7. 마일스톤 및 조정된 일정

| 마일스톤 | 목표 | 조정된 예상 완료일 | 상태 |
|---|---|---|---|
| **M1: Level 2 완성** | C01-C08 구현 및 벤치마크 통과 | 2026-01-15 | 🟡 진행 중 |
| **M2: 프로젝트 안정화** | CI/CD 구축 및 보안 강화 | 2026-02-28 | 📅 예정 |
| **M3: Level 3 완성** | T01-T08 구현 및 벤치마크 통과 | 2026-07-31 | 📅 예정 |
| **M4: 프로젝트 최종 완성** | 32개 시드 전체 구현 및 최적화 | 2026-09-30 | 📅 예정 |

---

## 8. 버전 관리 정책

- **MAJOR (x.0.0)**: Level 전체 완성 (예: Level 2 완성 시 2.0.0)
- **MINOR (x.y.0)**: 시드 1개 구현 완료
- **PATCH (x.y.z)**: 버그 수정, 문서 업데이트, 리팩토링

**적용 예시**:
- 현재 버전: 1.4.0 (C03 구현)
- C02 구현 후: 1.5.0
- Level 2 완성 후: 2.0.0
