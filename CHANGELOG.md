# Changelog

모든 주요 변경 사항이 이 파일에 기록됩니다.

## [1.1.0] - 2025-10-20

### Added

#### 코어 아키텍처 구현
- **SeedRegistry**: 32개 시드의 등록, 메타데이터 관리, 검색 기능
  - 시드 등록/해제 API
  - 레벨, 태그, 기하학, 비트폭 기반 검색
  - 의존성 조회 (재귀적 지원)
  
- **SeedRouter**: 동적 시드 선택 라우터
  - TaskEncoder: 태스크 설명 인코딩 (LSTM 기반)
  - InputAnalyzer: 입력 데이터 분석
  - GatingNetwork: 시드 활성화 확률 계산
  - Top-k 및 threshold 기반 선택
  - 의존성 자동 포함 기능
  
- **CompositionEngine**: 시드 조합 및 실행 엔진
  - DAG (Directed Acyclic Graph) 생성
  - 위상 정렬 기반 실행 순서 결정
  - 의존성 자동 해결
  - 캐시 통합 지원
  - 그래프 시각화 기능
  
- **CacheManager**: LRU 캐시 기반 결과 캐싱
  - 메모리 크기 및 항목 수 제한
  - LRU 제거 정책
  - 캐시 히트율 통계
  
- **MetricsCollector**: 성능 지표 수집
  - 시드별 실행 시간 추적
  - 캐시 히트율 모니터링
  - 전체 실행 통계
  - Top-N 시드 분석

#### 문서화
- **CORE_ARCHITECTURE.md**: 코어 아키텍처 설계 가이드
  - 5대 컴포넌트 상세 설명
  - API 설계 및 사용법
  - 학습 전략 (3단계 하이브리드)
  - 참조 연구 (Dynamic Neural Networks, Neural Module Networks)
  
- **dynamic_networks_survey.md**: Dynamic Neural Networks 서베이 핵심 내용
  - 동적 라우팅 메커니즘
  - 게이팅 함수 설계
  - 효율성 최적화 기법
  
- **neural_module_networks.md**: Neural Module Networks 핵심 내용
  - 모듈 조합 패턴
  - 공동 학습 방법론
  - 인터페이스 표준화

#### 테스트 및 예제
- **tests/test_core.py**: 코어 컴포넌트 단위 테스트
  - SeedRegistry 테스트
  - CacheManager 테스트
  - CompositionEngine 테스트
  - MetricsCollector 테스트
  
- **examples/basic_usage.py**: 기본 사용법 예제
  - 시드 정의 및 등록
  - 조합 그래프 생성
  - 실행 및 캐싱
  - 통계 수집

### Changed
- README.md 업데이트: 코어 아키텍처 섹션 추가
- requirements.txt 업데이트: PyTorch 의존성 명시

### Technical Details
- **언어**: Python 3.11+
- **프레임워크**: PyTorch 2.0+
- **아키텍처**: 모듈식, 플러그인 기반
- **디자인 패턴**: Registry, Strategy, DAG

---

## [1.0.0] - 2025-10-19

### Added
- 초기 프로젝트 구조 생성
- README.md 작성
- 32개 시드 카탈로그 정의
- 표준 인지 시드 설계 가이드 v1 문서화
- 시드 제작 관련 최신 문헌 조사
  - 쌍곡 신경망 (Hyperbolic Networks)
  - 스케일 등변 신경망 (Scale-Equivariant Networks)
  - FP8 양자화 (FP8 Quantization)
- LICENSE (Apache 2.0)
- .gitignore
- requirements.txt

### Documentation
- RESEARCH_SUMMARY.md: 40편 이상 논문 조사 요약
- hyperbolic_networks_notes.md
- scale_equivariant_notes.md
- fp8_quantization_notes.md

---

## Versioning

이 프로젝트는 [Semantic Versioning](https://semver.org/)을 따릅니다:
- **MAJOR**: 호환되지 않는 API 변경
- **MINOR**: 하위 호환되는 기능 추가
- **PATCH**: 하위 호환되는 버그 수정

