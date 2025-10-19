# FP8 Quantization: The Power of the Exponent - 핵심 내용

**출처**: Andrey Kuzmin, Mart Van Baalen, Yuwei Ren, Markus Nagel, Jorn Peters, Tijmen Blankevoort  
**발행**: arXiv:2208.09225, Aug 2022 (최종 수정: Feb 2024)  
**인용**: 111회  
**분야**: Machine Learning (cs.LG)

## 핵심 개념

### FP8 vs INT8: 부동소수점의 장점

신경망 양자화 시 저비트 정수(low-bit integers)가 효율성을 위한 표준 포맷이었으나, **저비트 부동소수점 숫자는 추가적인 자유도**를 가짐:
- 일부 비트를 지수 스케일(exponential scale)로 작동하도록 할당
- 이는 넓은 동적 범위(dynamic range)를 제공

### FP8 포맷의 선택사항

FP8 포맷 설계 시 중요한 결정:
1. **가수(mantissa) 비트 수**
2. **지수(exponent) 비트 수**
3. 이 선택이 성능에 미치는 영향을 분석적으로 제시

## 주요 연구 결과

### 1. PTQ (Post-Training Quantization)에서 FP8의 우위

**핵심 결론**: 광범위한 네트워크에 대한 사후학습 양자화(PTQ) 수행 시, **FP8 포맷이 INT8보다 정확도 측면에서 우수**

**이유**:
- FP8의 지수 표현이 네트워크의 이상치(outliers)를 더 잘 처리
- 동적 범위가 넓어 극단값 표현에 유리

### 2. 지수 비트 수의 선택 기준

**지수 비트 수 결정 요인**: 네트워크의 이상치 심각도(severity of outliers)
- 이상치가 많은 네트워크: 더 많은 지수 비트 필요
- 이상치가 적은 네트워크: 더 많은 가수 비트로 정밀도 향상

### 3. QAT (Quantization-Aware Training)에서의 차이

**양자화 인식 학습(QAT)** 시:
- FP8과 INT8의 차이가 사라짐
- 네트워크가 이상치 효과를 줄이도록 학습되기 때문
- 하지만 PTQ가 더 실용적인 경우가 많음 (재학습 불필요)

## 기술적 기여

### 1. FP8 포맷 분석

**분석적 접근**:
- 다양한 가수/지수 비트 조합의 성능 분석
- 이론적으로 어떤 설정이 더 나은 성능을 제공하는지 증명

### 2. 효율적 FP8 시뮬레이션 구현

실제 FP8 하드웨어 없이도 FP8 양자화 효과를 시뮬레이션할 수 있는 효율적 구현 제공

### 3. 학습 가능한 양자화 파라미터

**새로운 알고리즘 제안**:
- 스케일 파라미터 학습
- FP8 포맷의 지수 비트 수 학습
- 네트워크별 최적 포맷 자동 발견

## FP8 포맷 상세

### 비트 할당 예시

**FP8 (8비트 총)**:
- 1비트: 부호(sign)
- E비트: 지수(exponent)
- M비트: 가수(mantissa)
- E + M = 7 (부호 제외)

**일반적인 FP8 변형**:
1. **E4M3**: 4비트 지수, 3비트 가수
   - 넓은 동적 범위
   - 이상치가 많은 네트워크에 유리
   
2. **E5M2**: 5비트 지수, 2비트 가수
   - 매우 넓은 동적 범위
   - 극단적 이상치 처리
   
3. **E3M4**: 3비트 지수, 4비트 가수
   - 높은 정밀도
   - 이상치가 적은 네트워크에 유리

### INT8과의 비교

**INT8 (8비트 정수)**:
- 균일한 간격의 256개 값
- 동적 범위 제한적
- 이상치 처리 어려움

**FP8 장점**:
- 지수 표현으로 넓은 동적 범위
- 작은 값과 큰 값 동시 표현 가능
- 이상치에 강건

**INT8 장점**:
- 하드웨어 구현 단순
- 연산 속도 빠름 (현재 하드웨어 기준)

## 인지 시드 프레임워크와의 연관성

### 레벨별 비트폭 전략

가이드의 비트폭 권고가 본 논문 결과와 일치:

**Level 0 (Atomic)**:
- 권고: INT8
- 이유: 단순한 원자 연산, 이상치 적음
- 하드웨어 효율성 우선

**Level 1 (Molecular)**:
- 권고: INT8/FP8
- 이유: 중간 복잡도, 일부 이상치 발생 가능
- FP8 선택적 적용으로 정확도 향상

**Level 2 (Cellular)**:
- 권고: FP16/BF16
- 이유: 복잡한 추상화, 이상치 많음
- 학습 시 높은 정밀도 필요

**Level 3 (Tissue)**:
- 권고: FP16 학습 / INT8 추론 (캐시 포함)
- 이유: 최고 복잡도, QAT로 INT8 추론 가능
- PTQ 시 FP8 고려

### 구현 전략

#### 1. 적응적 포맷 선택

```python
# 의사코드: 시드별 최적 양자화 포맷 선택
def select_quantization_format(seed_level, outlier_severity):
    if seed_level == 0:  # Atomic
        return "INT8"
    elif seed_level == 1:  # Molecular
        if outlier_severity > threshold:
            return "FP8_E4M3"
        else:
            return "INT8"
    elif seed_level == 2:  # Cellular
        return "FP16"  # or BF16
    else:  # Tissue
        if training:
            return "FP16"
        else:  # inference
            if qat_applied:
                return "INT8"
            else:
                return "FP8_E4M3"
```

#### 2. 이상치 분석

```python
# 의사코드: 네트워크 이상치 심각도 분석
def analyze_outlier_severity(weights, activations):
    # 분포 분석
    weight_range = weights.max() - weights.min()
    weight_std = weights.std()
    
    # 이상치 비율 계산
    outlier_threshold = weight_std * 3
    outlier_ratio = (abs(weights) > outlier_threshold).mean()
    
    # 심각도 점수
    severity = outlier_ratio * (weight_range / weight_std)
    
    return severity
```

#### 3. 학습 가능한 양자화

```python
# 의사코드: 스케일과 지수 비트 수 학습
class LearnableQuantization(nn.Module):
    def __init__(self):
        self.scale = nn.Parameter(torch.ones(1))
        self.exponent_bits = nn.Parameter(torch.tensor(4.0))
        
    def forward(self, x):
        # 스케일 적용
        x_scaled = x * self.scale
        
        # 동적 FP8 양자화
        E = int(self.exponent_bits.round().clamp(2, 5))
        M = 7 - E
        
        # FP8 시뮬레이션
        x_quantized = simulate_fp8(x_scaled, E, M)
        
        return x_quantized / self.scale
```

## 실험 결과 요약

### PTQ 성능 비교

다양한 네트워크에서 FP8이 INT8 대비 우수:
- ResNet 계열: FP8 E4M3가 최적
- MobileNet 계열: FP8 E3M4 또는 E4M3
- Transformer: FP8 E5M2 (이상치 많음)

### QAT 성능 비교

QAT 적용 시 INT8도 FP8와 유사한 성능:
- 네트워크가 양자화에 적응
- 이상치 감소 학습
- 하지만 재학습 비용 발생

## 하드웨어 고려사항

### 현재 상황 (2024)

- **NVIDIA H100**: FP8 하드웨어 지원 (E4M3, E5M2)
- **Intel**: FP8 지원 계획 발표
- **AMD**: MI300 시리즈에서 FP8 지원

### 실용적 권장사항

1. **하드웨어 지원 확인**: FP8 가속 가능 여부
2. **PTQ 우선 시도**: 재학습 없이 FP8로 정확도 향상
3. **이상치 분석**: 네트워크별 최적 E/M 비트 결정
4. **QAT는 필요시**: PTQ로 부족할 때만 적용

## 시드별 양자화 전략 매핑

### Atomic Seeds (INT8)
- A01-A08: 단순 연산, INT8 충분
- 하드웨어 최적화 우선

### Molecular Seeds (INT8/FP8)
- M01, M02, M05: 이상치 가능성 → FP8 고려
- M03, M04, M06-M08: INT8 가능

### Cellular Seeds (FP16/BF16)
- C01-C08: 복잡한 추상화 → FP16 학습
- PTQ 시 FP8 추론 고려

### Tissue Seeds (FP16 학습 / INT8 추론)
- T01-T08: QAT 적용으로 INT8 추론
- PTQ만 가능하면 FP8 추론

## 구현 라이브러리

### PyTorch
```python
# FP8 시뮬레이션 (PyTorch 2.0+)
import torch
from torch.ao.quantization import FakeQuantize

# FP8 E4M3 설정
fp8_e4m3_config = {
    'dtype': torch.float8_e4m3fn,
    'quant_min': -448,
    'quant_max': 448
}
```

### TensorFlow
```python
# TensorFlow Lite FP8 양자화
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float8]
```

### ONNX Runtime
- FP8 양자화 지원 (실험적)
- ONNX 모델 변환 시 FP8 지정 가능

## 추가 읽을거리

- IEEE 754 부동소수점 표준
- OCP (Open Compute Project) FP8 사양
- NVIDIA Transformer Engine (FP8 최적화)
- Mixed-precision training 기법

---

**Note**: 이 논문은 인지 시드 프레임워크의 레벨별 비트폭 전략에 강력한 실증적 근거를 제공하며, 특히 PTQ 시나리오에서 FP8의 우위를 명확히 보여줌.

