# Level 1 Benchmark Setup Guide

## 개요

이 문서는 Level 1 (Molecular) 시드에 대한 벤치마크를 실행하기 위한 가이드입니다.

## 사전 요구사항

### 1. Python 환경

Python 3.11 이상이 필요합니다.

### 2. 의존성 설치

```bash
# 가상환경 생성 (권장)
python3.11 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 필수 패키지

- torch >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0

## 벤치마크 실행

### 1. 통합 테스트

전체 Level 1 시드 (M01~M08)에 대한 통합 테스트를 실행합니다:

```bash
# pytest 사용
pytest tests/test_level1_integration.py -v -s

# 또는 직접 실행
python tests/test_level1_integration.py
```

**테스트 항목**:
- 전체 시드 로드 검증
- 메타데이터 일관성 검증
- Forward pass 실행
- 파라미터 수 검증
- 성능 프로파일링
- 시드 조합 실행

### 2. 성능 벤치마크

포괄적인 성능 벤치마크를 실행합니다:

```bash
python benchmarks/level1_benchmark.py
```

**벤치마크 항목**:
- 성능 측정 (Latency, Throughput, Memory)
- 정확도 평가 (Consistency Score)
- 스케일링 테스트 (Batch Size, Sequence Length)
- 결과 시각화

**출력 파일**:
- `benchmarks/level1_results.json` - 전체 결과 (JSON)
- `benchmarks/level1_benchmark_report.md` - 요약 보고서
- `benchmarks/level1_performance_comparison.png` - 성능 비교 차트
- `benchmarks/level1_scaling_analysis.png` - 스케일링 분석
- `benchmarks/level1_params_vs_performance.png` - 파라미터 vs 성능

## 수용 기준

Level 1 (Molecular) 시드는 다음 기준을 충족해야 합니다:

| 메트릭 | 기준 | 설명 |
|--------|------|------|
| **Latency** | < 100ms | 배치당 평균 지연 시간 |
| **AMI/ARI** | ≥ 0.85 | 클러스터링 정확도 (해당 시드) |
| **Consistency** | ≥ 0.95 | 출력 일관성 점수 |
| **Memory** | < 500MB | 피크 메모리 사용량 |

## 결과 해석

### 1. 성능 메트릭

- **Latency (ms)**: 낮을수록 좋음
- **Throughput (samples/sec)**: 높을수록 좋음
- **Memory (MB)**: 낮을수록 좋음
- **Parameters (M)**: 모델 크기 (작을수록 효율적)

### 2. 정확도 메트릭

- **Consistency Score**: 동일 입력에 대한 출력 일관성 (1.0에 가까울수록 좋음)

### 3. 스케일링

- **Batch Size Scaling**: 배치 크기에 따른 지연 시간 증가율
- **Sequence Length Scaling**: 시퀀스 길이에 따른 지연 시간 증가율

## 문제 해결

### 1. CUDA 메모리 부족

```bash
# 배치 크기 줄이기
python benchmarks/level1_benchmark.py --batch-size 2
```

### 2. 느린 실행 속도

```bash
# 실행 횟수 줄이기
python benchmarks/level1_benchmark.py --num-runs 10
```

### 3. 의존성 오류

```bash
# 의존성 재설치
pip install -r requirements.txt --force-reinstall
```

## 추가 정보

### 벤치마크 커스터마이징

`benchmarks/level1_benchmark.py` 파일의 `main()` 함수에서 파라미터를 조정할 수 있습니다:

```python
benchmark.benchmark_performance(
    batch_size=4,      # 배치 크기
    seq_len=10,        # 시퀀스 길이
    input_dim=128,     # 입력 차원
    num_runs=50        # 실행 횟수
)
```

### CI/CD 통합

GitHub Actions 워크플로우에서 벤치마크를 자동으로 실행하려면 `.github/workflows/benchmark.yml`을 참조하세요.

## 참고 문서

- [README.md](README.md) - 프로젝트 개요
- [ROADMAP.md](ROADMAP.md) - 개발 로드맵
- [PROJECT_COMPREHENSIVE_ANALYSIS.md](PROJECT_COMPREHENSIVE_ANALYSIS.md) - 종합 분석 보고서

---

**작성일**: 2025-11-24  
**작성자**: Manus AI
