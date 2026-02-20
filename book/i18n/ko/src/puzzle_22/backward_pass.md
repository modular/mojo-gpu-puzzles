<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# ⛓️ Autograd 통합과 역전파

## 개요

이 퍼즐에서는 Fused LayerNorm + Linear 연산의 역전파 구현을 살펴봅니다. 역전파는 다음에 대한 기울기를 계산합니다:

- 입력 텐서
- LayerNorm scale (\\(\gamma\\))과 shift (\\(\beta\\)) 파라미터
- Linear 레이어의 가중치 행렬과 bias

구현할 수학적 연산은 다음과 같습니다:

1. LayerNorm 역전파 (유도 과정의 상세 내용은 [LayerNorm 역전파의 상세 유도](#layernorm-역전파의-상세-유도) 참조):
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \gamma \odot \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)}) \\]

2. Linear 역전파:
\\[\Large \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}x^T \\]
\\[\Large \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \\]
\\[\Large \frac{\partial L}{\partial x} = W^T\frac{\partial L}{\partial y} \\]

3. Fused 연산의 연쇄 법칙:
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y_{linear}} \frac{\partial y_{linear}}{\partial y_{norm}} \frac{\partial y_{norm}}{\partial x} \\]
여기서:

- \\(y_{norm}\\)은 LayerNorm 출력
- \\(y_{linear}\\)은 Linear 레이어 출력
- 연쇄 법칙이 두 연산을 통한 적절한 기울기 흐름을 보장

## 핵심 개념

- **스레드 구성**:
  - 시퀀스 위치당 하나의 스레드 블록 (그리드: `[batch_size, seq_len]`)
  - 중복을 방지하기 위해 시퀀스 위치당 단일 스레드
  - 각 시퀀스 위치의 모든 기울기를 하나의 스레드에서 계산
  - Atomic 연산을 위한 적절한 스레드 동기화 보장

- **메모리 접근**:
  - 입력 텐서: `[batch_idx, seq_idx, h]`로 접근
  - 출력 텐서: `[batch_idx, seq_idx, out_idx]`로 접근
  - 가중치: 선형 레이어에서 `[out_idx, h]`로 접근
  - Atomic 연산을 위한 메모리 정렬 보장
  - 자주 접근하는 데이터에 공유 메모리 사용

- **연산 흐름**:
  - 순전파와 동일한 순서로 LayerNorm 통계량 계산
  - 모든 출력 차원에 정규화된 값 재사용
  - 정규화와 선형 변환 결합
  - 전체 과정에서 수치 안정성 유지
  - 엣지 케이스를 적절히 처리

- **성능**:
  - 통계량의 중복 계산 방지
  - 연산을 결합하여 메모리 트래픽 최소화
  - `rebind[Scalar[dtype]]`로 적절한 타입 캐스팅 사용
  - 적절한 메모리 정렬 보장
  - Autograd 통합에 최적화

## 구성

- 배치 크기: `BATCH_SIZE = 4`
- 시퀀스 길이: `SEQ_LEN = 4`
- Hidden 차원: `HIDDEN_DIM = 8`
- 출력 차원: `OUTPUT_DIM = 16`
- Epsilon: `EPS = 1e-5`
- 데이터 타입: `DType.float32`

## 구현 (고급)

Fused 역전파 커널은 LayerNorm과 Linear의 역전파 연산을 하나의 GPU 커널로 결합합니다. 이 구현은 다음을 신중하게 다뤄야 하는 도전적인 과제입니다:

- 기울기 누적을 위한 [atomic 연산](https://docs.modular.com/mojo/std/os/atomic/Atomic/)
- 기울기 계산에서의 수치 안정성
- 효율적인 GPU 활용을 위한 메모리 접근 패턴
- 연산 간 적절한 동기화

```mojo
{{#include ../../../../../problems/p22/op/layernorm_linear.mojo:minimal_fused_backward_kernel}}
```

**핵심 최적화:**

- 모든 기울기 계산을 위한 단일 커널 실행
- 안전한 기울기 누적을 위한 atomic 연산
- 병합 메모리 접근 패턴
- 메모리 대역폭 사용량 절감
- 중간 텐서 할당 불필요

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. **스레드 구성**:
   - 시퀀스 위치당 하나의 스레드 블록
   - 시퀀스 위치당 단일 스레드
   - 모든 기울기를 하나의 스레드에서 계산

2. **메모리 접근**:
   - 입력/출력 텐서에 대한 병합 접근
   - 가중치 행렬에 대한 stride 접근
   - Atomic 연산을 위한 적절한 정렬

3. **연산 흐름**:
   - 순전파와 동일한 순서로 통계량 계산
   - 정규화된 값 재사용
   - 수치 안정성 유지

4. **성능**:
   - 메모리 트래픽 최소화
   - 적절한 타입 캐스팅 사용
   - 적절한 정렬 보장

</div>
</details>

### 코드 실행

Fused 역전파 구현을 테스트하려면 다음을 실행하세요:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p22 --backward
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p22 --backward
```

  </div>
  <div class="tab-content">

```bash
uv run poe p22 --backward
```

  </div>
</div>

출력은 다음과 같습니다:

```txt
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]
✅ Loaded Mojo operations library
============================================================
           Comprehensive Backward Pass Test
           Testing Custom LayerNorm + Linear Gradients
============================================================
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]

Testing CPU Backward Pass:

Testing CPU Backward Implementation - Backward Pass
---------------------------------------------------------
   Computing PyTorch autograd reference...
   Computing Mojo backward implementation (CPU)...
✅ CPU Backward Implementation backward completed
   Forward max difference: 1.49e-08
   grad_input: 2.98e-08 ✅
   grad_ln_weight: 5.96e-08 ✅
   grad_ln_bias: 2.38e-07 ✅
   grad_linear_weight: 9.54e-07 ✅
   grad_linear_bias: 0.00e+00 ✅

   Forward pass: ✅ CORRECT
   Gradients:    ✅ CORRECT
   Overall:      ✅ CORRECT

Testing GPU Backward Pass:

Testing GPU Backward Implementation - Backward Pass
---------------------------------------------------------
   Computing PyTorch autograd reference...
   Computing Mojo backward implementation (GPU)...

✅ GPU Backward Implementation backward completed
   Forward max difference: 1.86e-08
   grad_input: 4.47e-08 ✅
   grad_ln_weight: 5.96e-08 ✅
   grad_ln_bias: 3.58e-07 ✅
   grad_linear_weight: 9.54e-07 ✅
   grad_linear_bias: 0.00e+00 ✅

   Forward pass: ✅ CORRECT
   Gradients:    ✅ CORRECT
   Overall:      ✅ CORRECT

Backward Pass Test Summary:
   - CPU Backward:  ✅ CORRECT
   - GPU Backward:  ✅ CORRECT

   Overall Result: ✅ ALL CORRECT

BACKWARD PASS Test Completed!
```

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p22/op/layernorm_linear.mojo:minimal_fused_backward_kernel_solution}}
```

<div class="solution-explanation">

Fused 역전파 구현은 연산들을 효율적으로 결합합니다:

1. **스레드 구성과 메모리 레이아웃**:
   - 그리드 차원: `[batch_size, seq_len]`으로 시퀀스 위치당 하나의 스레드 블록
   - 스레드 인덱스: `batch_idx = block_idx.x`, `seq_idx = block_idx.y`
   - 메모리 레이아웃:
     - 입력 텐서: `[batch_size, seq_len, hidden_dim]`
     - 출력 텐서: `[batch_size, seq_len, output_dim]`
     - 가중치 행렬: `[output_dim, hidden_dim]`
     - 기울기: 입력 기울기용 `[batch_size, seq_len, hidden_dim]`
     - 파라미터 기울기: LayerNorm용 `[hidden_dim]`, Linear용 `[output_dim, hidden_dim]`

2. **LayerNorm 역전파 단계**:
   - 순전파와 동일한 순서로 순전파 통계량을 재계산합니다:
     - 평균: \\[\Large \mu = \frac{1}{H} \sum_{i=1}^{H} x_i \\]
     - 분산: \\[\Large \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\]
     - 역표준편차: \\[\Large \text{inv\_std} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \\]
   - 정규화된 값을 계산합니다: \\[\Large \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \\]
   - 기울기를 계산합니다:
     - 입력 기울기: \\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \gamma \odot \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)}) \\]
     - Scale 기울기: \\[\Large \frac{\partial L}{\partial \gamma} = \sum_{i=1}^{H} \frac{\partial L}{\partial y_i} \odot \hat{x}_i \\]
     - Shift 기울기: \\[\Large \frac{\partial L}{\partial \beta} = \sum_{i=1}^{H} \frac{\partial L}{\partial y_i} \\]

3. **Linear 역전파 단계**:
   - 각 출력 차원에 대해:
     - Bias 기울기: \\[\Large \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \\]
     - 가중치 기울기: \\[\Large \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}x^T \\]
     - 입력 기울기: \\[\Large \frac{\partial L}{\partial x} = W^T\frac{\partial L}{\partial y} \\]
   - 기울기 누적을 위한 atomic 연산 사용:
     - Bias 기울기에 적절한 정렬로 `atomic_add` 사용
     - 가중치 기울기에 적절한 정렬로 `atomic_add` 사용
     - LayerNorm 파라미터 기울기에 적절한 정렬로 `atomic_add` 사용

4. **메모리 접근 패턴**:
   - 입력/출력 텐서에 대한 병합 접근
   - 가중치 행렬에 대한 stride 접근
   - 기울기 누적을 위한 atomic 연산
   - 중간 결과를 위한 공유 메모리
   - 자주 접근하는 값을 위한 레지스터 사용
   - 모든 연산에 대한 적절한 메모리 정렬

5. **수치 안정성**:
   - 분모의 epsilon 처리에 주의
   - 기울기의 적절한 스케일링
   - 안정적인 통계량 계산
   - `rebind[Scalar[dtype]]`로 타입 캐스팅
   - 엣지 케이스의 적절한 처리
   - 순전파와 동일한 연산 순서 유지

6. **성능 최적화**:
   - 모든 연산을 위한 단일 커널 실행
   - 계산된 통계량 재사용
   - 메모리 트래픽 최소화
   - 중간 텐서 할당 불필요
   - 효율적인 스레드 활용
   - 동기화 지점 감소
   - 최적화된 메모리 접근 패턴
   - 적절한 메모리 정렬

7. **구현 세부 사항**:
   - 컴파일 타임 상수를 위한 `@parameter` 사용
   - 텐서 차원의 적절한 처리
   - 효율적인 타입 캐스팅과 변환
   - 공유 메모리의 신중한 관리
   - 연산 간 적절한 동기화
   - 오류 처리와 경계 검사
   - PyTorch autograd 시스템과의 통합

이 구현은 다음을 통해 unfused 버전보다 더 나은 성능을 달성합니다:

- Kernel fusion을 통한 메모리 대역폭 사용량 절감
- 커널 실행 오버헤드 최소화
- 메모리 접근 패턴 최적화
- GPU 리소스의 효율적 활용
- 수치 안정성 유지
- 기울기 누적의 적절한 처리
- 적절한 메모리 정렬 보장
- 효율적인 autograd 통합

Fused 역전파는 LayerNorm + Linear 연산이 자주 함께 사용되는 트랜스포머 아키텍처에서 특히 중요하며, 실제 애플리케이션에서 상당한 성능 이점을 제공합니다.
</div>
</details>

## 성능 고려 사항

역전파 구현은 오버헤드를 최소화하기 위해 최적화된 `torch.compile`을 사용합니다:

```python
# Compilation configuration
torch._dynamo.config.cache_size_limit = 64  # Increase cache
torch._dynamo.config.suppress_errors = True  # Handle errors gracefully
torch._dynamo.config.automatic_dynamic_shapes = True  # Dynamic shapes
```

이러한 최적화가 역전파에서 특히 중요한 이유는 다음과 같습니다:

- 작은 텐서 연산은 컴파일 캐싱의 이점을 받습니다
- 동적 형상은 역전파에서 흔하게 발생합니다
- 기울기 계산에는 강건한 오류 처리가 필요합니다
- 캐시 크기는 반복적인 역전파 연산에 도움이 됩니다
- 적절한 오류 처리는 기울기 계산에 매우 중요합니다
- 컴파일 오버헤드는 학습 시간에 큰 영향을 줄 수 있습니다

역전파는 정확성을 유지하면서 컴파일 오버헤드를 최소화하기 위해 `reduce-overhead` 모드로 컴파일됩니다. 이것이 특히 중요한 이유는:

- 역전파는 학습 중에 빈번하게 호출됩니다
- 기울기 계산은 수치적으로 안정적이어야 합니다
- 메모리 접근 패턴이 최적화되어야 합니다
- Atomic 연산에는 적절한 동기화가 필요합니다
- Autograd 통합이 효율적이어야 합니다

## LayerNorm 역전파의 상세 유도

LayerNorm의 역전파 기울기는 연쇄 법칙을 주의 깊게 적용하여 유도됩니다. 단계별 유도 과정은 다음과 같습니다:

### 순전파 연산

- 평균: \\(\mu = \frac{1}{H} \sum_{i=1}^{H} x_i\\)
- 분산: \\(\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2\\)
- 정규화된 값: \\(\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\\)
- 최종 출력: \\(y = \gamma \odot \hat{x} + \beta\\)

### 연쇄 법칙 적용

\\(\frac{\partial L}{\partial x}\\)를 계산하기 위해 연쇄 법칙을 적용합니다:
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial \hat{x}} \frac{\partial \hat{x}}{\partial x}\\]

### 기울기 구성 요소

#### 출력에서 정규화된 값으로

- \\(\frac{\partial y}{\partial \hat{x}} = \gamma\\) (요소별 곱셈)

#### 정규화된 값에서 입력으로

기울기 \\(\frac{\partial \hat{x}}{\partial x}\\)에는 세 가지 구성 요소가 있습니다:

- 분자를 통한 직접적 효과: \\(\frac{1}{\sqrt{\sigma^2 + \epsilon}}\\)
- 평균을 통한 간접적 효과: \\(-\frac{1}{H} \frac{1}{\sqrt{\sigma^2 + \epsilon}}\\)
- 분산을 통한 간접적 효과: \\(-\frac{(x - \mu)}{H(\sigma^2 + \epsilon)^{3/2}} (x - \mu)\\)

### 항 결합

정규화 항을 통한 기울기는 다음과 같이 정리됩니다:
\\[\Large \frac{\partial \hat{x}}{\partial x} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)})\\]

### 최종 기울기 표현식

모든 항을 결합하면:
\\[\Large \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot \gamma \odot \frac{1}{\sqrt{\sigma^2 + \epsilon}} (1 - \frac{1}{H} - \frac{(x - \mu)^2}{H(\sigma^2 + \epsilon)})\\]

### 핵심 통찰

- 연쇄 법칙은 x가 출력에 영향을 미치는 모든 경로를 고려합니다
- 정규화 항 \\(\sqrt{\sigma^2 + \epsilon}\\)은 분자와 분모 모두에 등장합니다
- 평균과 분산 항은 기울기 흐름의 추가 경로를 생성합니다
- 최종 표현식은 모든 효과를 하나의 효율적인 계산으로 결합합니다

### 구현 시 고려 사항

- 기울기가 \\(\gamma\\)의 스케일링 효과를 적절히 반영합니다
- 평균과 분산의 정규화 효과가 보존됩니다
- 수치 안정성 항 \\(\epsilon\\)이 유지됩니다
- 기울기가 hidden 차원 H 전체에 걸쳐 적절히 스케일링됩니다
- 수치 안정성을 위해 연산 순서가 순전파와 일치합니다

이 유도를 통해 역전파가 순전파와 동일한 수치적 특성을 유지하면서 필요한 모든 기울기를 효율적으로 계산할 수 있습니다.
