# Cognitive Seed Framework - 프로젝트 로드맵 (v3.0)

**버전**: 3.0  
**작성일**: 2025-11-27  
**작성자**: Manus AI (NouseAI)
**GitHub**: https://github.com/tjwlstj/cognitive-seed-framework

---

## 1. 로드맵 개요

### 1.1. 프로젝트 비전

32개의 표준 인지 시드를 완전히 구현하고, 동적 조합을 통해 복잡한 AI 태스크를 해결하는 **세계 최초의 모듈식 인지 프레임워크**를 구축합니다.

### 1.2. 핵심 목표

- **완전성**: 4개 레벨(Atomic, Molecular, Cellular, Tissue) 32개 시드 전체 구현
- **안정성**: Level 1, 2, 3에 대한 통합 테스트 및 벤치마크 구축
- **효율성**: 토큰 최적화를 고려한 소규모 분산 개발 세션 적용
- **체계성**: 각 세션의 산출물을 명확히 하고, 진행 상황을 체계적으로 관리
- **보안 및 CI/CD**: 보안 강화 및 지속적인 통합/배포 파이프라인 구축

### 1.3. 로드맵 구조 (v3.0)

본 로드맵은 기존 v2.1을 계승하고, 현재 프로젝트 상태(Level 2 진행 중)를 반영하여 최신화되었습니다. 개발 과정을 **토큰 효율적인 소규모 세션**으로 분산하고, 선택적 집중이 가능하도록 체계화했습니다.

- **Phase 1**: Level 0-1 기반 구축 (완료 ✅)
- **Phase 2**: Level 1 완성 및 Level 2 시작 (완료 ✅)
- **Phase 3**: Level 2 (Cellular) 완성 (진행 중 🟡)
- **Phase 4**: Level 3 (Tissue) 구현 (예정 📅)
- **Phase 5**: 프로젝트 안정화 및 강화 (예정 📅)
- **Phase 6**: 최적화, 배포 및 연구 (예정 📅)

---

## 2. 현재 상태 (2025-11-27 기준)

### 2.1. 전체 진행 상황

| 항목 | 완료 | 전체 | 진행률 |
|---|---|---|---|
| **시드 구현** | 18 | 32 | 56.3% |
| **Phase** | 2.1 | 6 | 35.0% |

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

### 2.3. 주요 문서 및 현황 불일치

- `ROADMAP.md` (v2.1)는 Level 1 완성을 최종 목표로 하고 있어 최신 상태를 반영하지 못합니다.
- `CHANGELOG.md`는 v1.2.0이 최신으로, M07, M08, C01 구현 내역이 누락되었습니다.
- `VERSION` 파일은 `1.1.0`으로 기록되어 있어, 실제 개발 현황과 차이가 있습니다.

**조치**: 본 로드맵(v3.0)을 기준으로 프로젝트 문서를 최신화하고, 버전 관리를 체계화할 필요가 있습니다.

---

## 3. Phase 3: Level 2 (Cellular) 완성 (진행 중 🟡)

**목표**: Level 2 시드 7개 추가 구현 및 Level 2 통합 벤치마크 구축

### 3.1. 개발 전략: 소규모 분산 세션

- 각 시드 구현을 독립적인 **개발 세션**으로 분리하여 토큰 사용량을 최적화하고, 선택적 집중 개발을 지원합니다.
- 각 세션은 **코드 구현, 단위 테스트, 문서 작성**을 포함하는 완결된 단위입니다.

### 3.2. Level 2 개발 세션 계획

Level 2 시드는 Level 0-1 시드에만 의존하므로, 모든 잔여 시드는 **병렬 개발이 가능**합니다. 아래는 권장 개발 순서이며, 필요에 따라 순서를 조정하거나 동시 진행할 수 있습니다.

| 세션 | 대상 시드 | 우선순위 | 예상 기간 | 주요 의존성 |
|---|---|---|---|---|
| **S3.1** | C03 Schema Learner | **P0** | 5-7일 | M01, M05, A05 |
| **S3.2** | C02 Counterfactual Reasoner | P1 | 5-7일 | M02, M08, A08 |
| **S3.3** | C06 Attention Director | P1 | 5-7일 | M06, M01, A05 |
| **S3.4** | C07 Boundary Detector | P2 | 5-7일 | A01, M03, M06 |
| **S3.5** | C08 Novelty Assessor | P2 | 5-7일 | M05, M07, A04 |
| **S3.6** | C04 Perspective Shifter | P3 | 5-7일 | M04, M07, A02 |
| **S3.7** | C05 Narrative Constructor | P3 | 5-7일 | M06, M03, A06 |
| **S3.8** | Level 2 통합 및 벤치마크 | **P0** | 7-10일 | C01-C08 완료 |

### 3.3. 세션별 산출물 가이드

각 개발 세션(S3.1 ~ S3.7)은 다음 산출물을 생성해야 합니다.

1.  **시드 구현 코드**: `seeds/cellular/<seed_name>.py`
2.  **단위 테스트 코드**: `tests/cellular/test_<seed_name>.py` (최소 5개 케이스)
3.  **구현 완료 보고서**: `<SEED_ID>_IMPLEMENTATION_COMPLETE.md`
4.  **초기화 파일 업데이트**: `seeds/cellular/__init__.py`
5.  **Git 커밋**: `feat: Implement <SEED_ID> <Seed_Name>` 형식

---

## 4. Phase 4: Level 3 (Tissue) 구현 (예정 📅)

**목표**: Level 3 시드 8개 전체 구현 및 Level 3 통합 벤치마크 구축

- Phase 3 완료 후, Level 2 시드와의 의존성을 분석하여 세부 개발 세션을 계획합니다.
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

## 5. Phase 5: 프로젝트 안정화 및 강화 (예정 📅)

**목표**: 프로젝트의 장기적인 유지보수성, 보안, 사용성을 강화합니다.

| 세션 | 작업 내용 | 우선순위 | 비고 |
|---|---|---|---|
| **S5.1** | **문서 최신화 및 버전 관리** | **P0** | CHANGELOG, VERSION, README 등 일관성 확보 |
| **S5.2** | **보안 강화** | P1 | SECURITY.md 작성, Dependabot/CodeQL 설정 |
| **S5.3** | **CI/CD 파이프라인 구축** | P1 | GitHub Actions 기반 자동 테스트 및 빌드 |
| **S5.4** | **기여 가이드라인 마련** | P2 | CONTRIBUTING.md, Issue/PR 템플릿 작성 |
| **S5.5** | **코드 리팩토링** | P3 | 기술 부채 해소 및 코드 품질 개선 |

---

## 6. Phase 6: 최적화, 배포 및 연구 (예정 📅)

**목표**: 프레임워크의 성능을 극대화하고, 실제 적용 사례를 제시하며, 연구를 확장합니다.

- **양자화 (Quantization)**: FP8, INT8 등 양자화 기법 적용 및 성능 평가
- **백본 네트워크 통합**: 다양한 백본(e.g., ResNet, ViT)과의 통합 예제 제공
- **배포 자동화**: PyPI 패키지 배포 자동화
- **허깅페이스 허브 연동**: 모델 및 데모를 허깅페이스 허브에 공개
- **신규 시드 연구**: 신경과학 등 최신 연구 기반의 신규 시드 설계

---

## 7. 마일스톤 및 일정

| 마일스톤 | 목표 | 예상 완료일 |
|---|---|---|
| **M1: Level 2 완성** | C01-C08 구현 및 벤치마크 통과 | 2026-01-31 |
| **M2: 프로젝트 안정화** | CI/CD 구축 및 보안 강화 | 2026-02-28 |
| **M3: Level 3 완성** | T01-T08 구현 및 벤치마크 통과 | 2026-06-30 |
| **M4: 프로젝트 최종 완성** | 32개 시드 전체 구현 및 최적화 | 2026-08-31 |

