# CHANGELOG

모든 주요 변경 사항이 이 파일에 기록됩니다.

## [2.1.0] - 2026-01-06

### Added
- **T01: Abductive Reasoner 구현** (Level 3)
  - 최선 설명 추론 (Abductive Reasoning) 기능 구현
  - 인과 추론, 반사실 추론, 충돌 해소 통합
  - 13단계 추론 파이프라인 구현
  - 3가지 평가 메트릭: 설명력, 그럴듯함, 간결성
  - 어텐션 기반 최선 설명 선택
  - 20개 단위 테스트 작성
- **분할 개발 계획 수립** (DEVELOPMENT_PLAN_2026-01-06.md)
  - 토큰 효율을 고려한 체계적인 개발 전략
  - Phase 4 & 5 병행 개발 계획

### Changed
- **VERSION**: 2.0.0 → 2.1.0
- **Level 3 (Tissue)**: 0/8 → 1/8 (12.5%) 🟡

### Progress
- **Level 3 (Tissue)**: 1/8 완료 (12.5%) 🟡
- **전체 진행률**: 25/32 (78.1%)

---

## [2.0.0] - 2025-12-18

### Added
- **Level 2 (Cellular) 100% 완료**
  - **C02: Counterfactual Reasoner 구현**
  - **C04: Perspective Shifter 구현**
  - **C05: Narrative Constructor 구현**
  - **C07: Boundary Detector 구현**
  - **C08: Novelty Assessor 구현**

### Changed
- **로드맵 v4.0 업데이트**: Level 2 완료를 반영하고, Level 3 구현을 위한 상세 개발 세션 계획을 수립했습니다.
- **README.md 업데이트**: 프로젝트 진행률(75%) 및 최근 업데이트 내역을 최신 정보로 수정했습니다.
- **VERSION**: 1.5.0 → 2.0.0

### Progress
- **Level 2 (Cellular)**: 8/8 완료 (100%) ✅
- **전체 진행률**: 24/32 (75.0%)

---

## [1.5.0] - 2025-12-09

### Added
- **C06: Attention Director 구현** (Level 2)

### Progress
- **Level 2 (Cellular)**: 4/8 완료 (50.0%) 🟡

---

## [1.4.0] - 2025-12-01

### Added
- **C03: Schema Learner 구현** (Level 2)

### Progress
- **Level 2 (Cellular)**: 2/8 완료 (25.0%) 🟡

---

## [1.3.0] - 2025-11-27
### Added
- **C01: Metaphor Engine 구현** (Level 2)
- **M08: Conflict Resolver 구현** (Level 1)
- **M07: Analogy Mapper 구현** (Level 1)

### Changed
- **로드맵 v3.0 업데이트**
- **Level 1 벤치마크 구축**

### Progress
- **Level 1 (Molecular)**: 8/8 완료 (100%) ✅
- **Level 2 (Cellular)**: 1/8 완료 (12.5%) 🟡

---

## [1.2.0] - 2025-11-01

### Added
- **M06: Context Integrator 구현**

### Progress
- **Level 1 (Molecular)**: 5/8 완료 (62.5%)

---

## [1.1.0] - 2025-10-20

### Added
- **코어 아키텍처 구현** (Registry, Router, Engine, Cache, Metrics)

---

## [1.0.0] - 2025-10-19

### Added
- 초기 프로젝트 구조 생성 및 32개 시드 카탈로그 정의
