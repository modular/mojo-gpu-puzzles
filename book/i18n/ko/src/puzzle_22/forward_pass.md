<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

# ⚛️ Fused vs Unfused Kernels

## 개요

이 퍼즐에서는 [LayerNorm](https://arxiv.org/abs/1607.06450)과 Linear 연산에 대한 두 가지 접근 방식을 구현하고 비교하며, kernel fusion의 성능 이점을 탐구합니다:

1. **Unfused 방식**: LayerNorm과 Linear를 별도의 연산으로 실행
2. **Fused 커널**: LayerNorm과 Linear 연산을 하나의 GPU 커널로 결합

이 비교를 통해 kernel fusion이 다음과 같은 방법으로 성능을 크게 개선할 수 있음을 보여줍니다:

- 메모리 대역폭 사용량 절감
- 커널 실행 오버헤드 최소화
- 캐시 활용도 향상
- 중간 결과 저장을 위한 메모리 할당 제거

## 핵심 개념

이 퍼즐에서 배울 내용:

- 여러 연산을 결합하는 **kernel fusion 기법**
- Fused 연산을 통한 **메모리 대역폭 최적화**
- 서로 다른 커널 구현의 **성능 벤치마킹**
- Fused 연산에서의 **수치 안정성**
- **PyTorch 커스텀 연산 통합**

결합할 수학적 연산은 다음과 같습니다:

1. LayerNorm:
\\[\Large \text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]

2. Linear:
\\[\Large \text{Linear}(x) = Wx + b \\]

Fused 연산으로 결합하면 다음을 계산합니다:
\\[\Large \text{Fused}(x) = W(\gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta) + b \\]

## LayerNorm 이해하기

LayerNorm은 deep neural networks의 학습을 안정화하고 가속하는 정규화 기법입니다. 구성 요소와 파라미터를 하나씩 살펴보겠습니다:

### LayerNorm이 하는 일

1. **정규화**: LayerNorm은 각 샘플의 특성(hidden 차원) 전체에 걸쳐 활성화 값을 독립적으로 정규화합니다. 구체적으로:
   - 각 시퀀스 위치에서 hidden 차원에 대한 통계량을 계산합니다
   - 배치의 각 샘플은 독립적으로 정규화됩니다
   - 배치 차원에 대해 정규화하는 [BatchNorm](https://arxiv.org/abs/1502.03167)과는 다릅니다

2. **파라미터**:
   - \\(\gamma\\) (scale): 네트워크가 각 특성의 최적 스케일을 학습할 수 있게 하는 학습 가능한 파라미터 벡터
   - \\(\beta\\) (shift): 네트워크가 각 특성의 최적 이동량을 학습할 수 있게 하는 학습 가능한 파라미터 벡터
   - \\(\epsilon\\): 0으로 나누는 것을 방지하기 위해 분산에 더하는 작은 상수 (1e-5)

### LayerNorm의 실제 역할

LayerNorm은 deep neural networks에서 여러 중요한 기능을 수행합니다:

1. **특성 표준화**:
   - 각 특성을 평균 0, 분산 1로 변환합니다
   - 네트워크의 학습 과정을 더 안정적으로 만듭니다
   - 학습 중 레이어 입력의 분포가 변하는 "내부 공변량 이동(internal covariate shift)" 문제를 방지합니다

2. **기울기 흐름**:
   - 네트워크를 통한 기울기 흐름을 개선합니다
   - 기울기 소실/폭발 문제를 방지합니다
   - 더 높은 학습률을 사용할 수 있어 학습 효율이 향상됩니다

3. **정규화 효과**:
   - 암묵적인 정규화 역할을 합니다
   - 특성 분포를 정규화하여 과적합을 방지합니다
   - 입력 변동에 대한 네트워크의 강건성을 높입니다

4. **시퀀스 모델링**:
   - 트랜스포머 아키텍처에서 특히 효과적입니다
   - 서로 다른 시퀀스 길이에서도 일관된 신호 크기를 유지합니다
   - 가변 길이 시퀀스를 더 잘 처리할 수 있게 합니다

5. **학습 역학**:
   - 학습 수렴을 가속합니다
   - 세밀한 학습률 조정의 필요성을 줄입니다
   - 가중치 초기화에 대한 네트워크의 민감도를 낮춥니다

### 수학적 구성 요소

1. **평균 계산** (\\(\mu\\)):
   \\[\Large \mu = \frac{1}{H} \sum_{i=1}^{H} x_i \\]
   - Hidden 차원(H)에 걸쳐 평균을 계산합니다
   - 각 시퀀스 위치마다 고유한 평균을 가집니다

2. **분산 계산** (\\(\sigma^2\\)):
   \\[\Large \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\]
   - Hidden 차원에 걸쳐 분산을 계산합니다
   - 정규화된 값의 스케일링에 사용됩니다

3. **정규화와 스케일링**:
   \\[\Large \text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]
   - 먼저 입력을 평균 0, 분산 1로 정규화합니다
   - 그런 다음 학습 가능한 scale (\\(\gamma\\))과 shift (\\(\beta\\)) 파라미터를 적용합니다
   - \\(\odot\\) 기호는 요소별 곱셈(아다마르 곱)을 나타냅니다
   - 예를 들어, \\(\gamma = [1.2, 0.8, 1.5]\\)이고 정규화된 입력이 \\([0.5, -0.3, 0.7]\\)이면, \\(\gamma \odot x = [0.6, -0.24, 1.05]\\)입니다

### LayerNorm이 중요한 이유

1. **학습 안정성**:
   - 활성화 값이 너무 크거나 작아지는 것을 방지합니다
   - 네트워크 전체에 걸쳐 일관된 신호 크기를 유지합니다

2. **특성 학습**:
   - scale (\\(\gamma\\))과 shift (\\(\beta\\)) 파라미터를 통해 어떤 특성이 중요한지 학습할 수 있습니다
   - 특정 특성을 무시하거나 강조하는 것을 효과적으로 학습할 수 있습니다

3. **독립성**:
   - BatchNorm과 달리, LayerNorm의 통계량은 각 샘플에 대해 독립적으로 계산됩니다
   - 가변 길이 시퀀스와 작은 배치 크기에 더 적합합니다

## 구성

- 배치 크기: `BATCH_SIZE = 4`
- 시퀀스 길이: `SEQ_LEN = 4`
- Hidden 차원: `HIDDEN_DIM = 8`
- 출력 차원: `OUTPUT_DIM = 16`
- Epsilon: `EPS = 1e-5`
- 데이터 타입: `DType.float32`

## 구현 방식

### 1. Unfused 구현

Unfused 방식은 여러 커널을 사용하여 연산을 개별적으로 실행합니다. 이전 챕터에서 작성한 커널들을 살펴보겠습니다:

#### 행렬 곱셈 커널

[Puzzle 16: 행렬 곱셈 (MatMul)](../puzzle_16/puzzle_16.md)에서 사용한 tiled 행렬 곱셈 커널을 선형 변환에 재사용합니다. 이 커널은 다양한 행렬 크기를 안전하게 처리하기 위한 경계 검사를 포함합니다:

```mojo
{{#include ../../../../../problems/p22/op/layernorm_linear.mojo:matmul_idiomatic_tiled}}
```

#### 전치 커널

효율적인 메모리 접근 패턴을 위해 공유 메모리 tiling을 사용하는 전치 커널입니다:

```mojo
{{#include ../../../../../problems/p22/op/layernorm_linear.mojo:transpose_kernel}}
```

#### Bias 합산 커널

Bias 항을 더하는 간단한 요소별 합산 커널입니다:

```mojo
{{#include ../../../../../problems/p22/op/layernorm_linear.mojo:add_bias_kernel}}
```

#### LayerNorm 커널

이제 이 커널을 완성하여 LayerNorm 연산을 구현합니다. 다음이 필요합니다:

1. 각 시퀀스 위치에 대한 평균 \\(\mu\\)과 분산 \\(\sigma^2\\) 계산
2. 이 통계량을 사용하여 입력 정규화
3. Scale \\(\gamma\\)과 shift \\(\beta\\) 파라미터 적용

```mojo
{{#include ../../../../../problems/p22/op/layernorm_linear.mojo:layernorm_kernel}}
```

**구현 단계:**

1. 먼저, 병렬 reduction을 사용하여 평균과 분산을 계산합니다
2. 그런 다음, 이 통계량으로 입력을 정규화합니다
3. 마지막으로, scale과 shift 파라미터를 적용합니다

**Unfused 방식의 특성:**

- 여러 번의 커널 실행 (LayerNorm → MatMul → Bias)
- 연산 간 중간 텐서 할당
- 별도의 패스로 인한 메모리 대역폭 사용량 증가
- 관심사 분리가 명확한 간결한 구현
- 각 연산이 격리되어 디버깅이 용이

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. **스레드 구성**:
   - 시퀀스 위치당 하나의 스레드 블록 사용 (그리드: `[batch_size, seq_len]`)
   - 각 스레드가 하나의 hidden 차원 요소를 처리
   - 시퀀스당 통계량을 한 번만 계산하여 중복 연산 방지

2. **메모리 접근**:
   - 입력 텐서: `[batch_idx, seq_idx, hidden_idx]`로 접근
   - 출력 텐서: `[batch_idx, seq_idx, hidden_idx]`로 접근
   - LayerNorm 파라미터: `[hidden_idx]`로 접근

3. **수치 안정성**:
   - 제곱근을 취하기 전에 epsilon(1e-5)을 더합니다
   - 적절한 타입 캐스팅을 위해 `rebind[Scalar[dtype]]` 사용
   - 분산은 (sq_sum / hidden_dim) - (mean * mean)으로 계산

4. **성능**:
   - 한 번의 패스로 평균과 분산을 동시에 계산
   - 계산된 통계량을 시퀀스 내 모든 요소에 재사용
   - 불필요한 메모리 barrier 방지

</div>
</details>

### 코드 실행

Unfused 구현을 테스트하려면 다음을 실행하세요:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p22 --unfused
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p22 --unfused
```

  </div>
  <div class="tab-content">

```bash
uv run poe p22 --unfused
```

  </div>
</div>

출력은 다음과 같습니다:

```txt
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]
✅ Loaded Mojo operations library
============================================================
   Puzzle 22: UNFUSED Algorithm Test & Benchmark
============================================================

🧪 Correctness Testing for UNFUSED Algorithm
====================================================

Testing Reference PyTorch Implementation
-----------------------------------------------
✅ Reference PyTorch
   Max difference: 0.00e+00
   Result: ✅ CORRECT

Testing CPU Implementation
---------------------------------
✅ Using Mojo fused kernel (CPU)
   Max difference: 1.86e-08
   Result: ✅ CORRECT

Testing GPU Unfused Implementation
-----------------------------------------
✅ Using Mojo unfused kernel (GPU)
   Max difference: 1.86e-08
   Result: ✅ CORRECT

Correctness Summary:
   - Reference:   ✅ CORRECT
   - CPU:         ✅ CORRECT
   - GPU unfused: ✅ CORRECT

   Overall Correctness: ✅ ALL CORRECT

Benchmarking CPU vs GPU UNFUSED
------------------------------------------
   Testing CPU performance...
   CPU: 3173.70ms (50 iterations)
   Testing GPU unfused performance...
   GPU unfused: 3183.57ms (50 iterations)

   GPU unfused vs CPU: 1.00x slower
   CPU wins (GPU overhead > computation benefit)

UNFUSED Algorithm Test Completed!
```

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p22/op/layernorm_linear.mojo:layernorm_kernel_solution}}
```

<div class="solution-explanation">

Unfused 구현은 각 스레드가 출력 텐서의 하나의 요소를 처리하는 직관적인 방식을 따릅니다. 핵심 구성 요소를 하나씩 살펴보겠습니다:

1. **스레드와 블록 구성**:

   ```mojo
   batch_idx = block_idx.x
   seq_idx = block_idx.y
   hidden_idx = thread_idx.x
   ```

   - 각 스레드 블록이 배치 내 하나의 시퀀스 위치를 처리합니다
   - 그리드 차원: `[batch_size, seq_len]`
   - 각 스레드가 hidden 차원의 하나의 요소를 처리합니다
   - 인덱스가 범위를 벗어나면 조기 반환합니다:

     ```mojo
     if (batch_idx >= batch_size or seq_idx >= seq_len or hidden_idx >= hidden_dim):
         return
     ```

2. **통계량 계산**:

   ```mojo
   var sum_val: Scalar[dtype] = 0
   var sq_sum: Scalar[dtype] = 0

   @parameter
   for h in range(hidden_dim):
       val = input[batch_idx, seq_idx, h]
       sum_val += rebind[Scalar[dtype]](val)
       sq_sum += rebind[Scalar[dtype]](val * val)
   ```

   - 한 번의 패스로 합계와 제곱합을 동시에 계산합니다
   - 컴파일 타임 루프 전개를 위해 `@parameter`를 사용합니다
   - `rebind[Scalar[dtype]]`로 적절한 타입 캐스팅을 수행합니다
   - 평균과 분산을 계산합니다:

     ```mojo
     mean_val = sum_val / hidden_dim
     var_val = (sq_sum / hidden_dim) - (mean_val * mean_val)
     inv_std = 1.0 / sqrt(var_val + 1e-5)
     ```

3. **정규화와 스케일링**:

   ```mojo
   input_val = input[batch_idx, seq_idx, hidden_idx]
   normalized = (input_val - mean_val) * inv_std * rebind[Scalar[dtype]](
       ln_weight[hidden_idx]
   ) + rebind[Scalar[dtype]](ln_bias[hidden_idx])
   output[batch_idx, seq_idx, hidden_idx] = normalized
   ```

   - 정규화를 적용합니다: \\[\Large \text{normalized} = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]
   - 학습 가능한 파라미터 `γ` (ln_weight)로 스케일링합니다
   - 학습 가능한 bias `β` (ln_bias)를 더합니다
   - 결과를 출력 텐서에 저장합니다

4. **성능 특성**:
   - 각 스레드가 독립적으로 통계량을 계산합니다
   - 공유 메모리 사용 없음 (간단하지만 덜 효율적)
   - 메모리 접근 패턴:
     - 입력: `[batch_idx, seq_idx, h]`
     - 출력: `[batch_idx, seq_idx, hidden_idx]`
     - 파라미터: `[hidden_idx]`
   - 다음을 통해 수치 안정성을 보장합니다:
     - 제곱근 전에 epsilon(1e-5) 추가
     - 적절한 타입 캐스팅 사용
     - 수치적으로 안정적인 방식으로 분산 계산

5. **구현 세부 사항**:
   - **타입 안전성**:
     - 중간 계산에 `Scalar[dtype]` 사용
     - 적절한 타입 캐스팅을 위해 `rebind[Scalar[dtype]]` 사용
     - 일관된 부동소수점 정밀도 보장

   - **메모리 접근**:
     - 입력 텐서에서 병합 읽기
     - 출력 텐서에 병합 쓰기
     - LayerNorm 파라미터에 순차적 접근

   - **연산 흐름**:
     - 통계량 계산: \\[\Large O(H) \text{ operations per thread} \\]
     - 정규화: \\[\Large O(1) \text{ operations per thread} \\]
     - 전체 복잡도: \\[\Large O(H) \text{ per output element} \\]

   - **한계점**:
     - 통계량의 중복 계산
     - 중간 결과를 위한 공유 메모리 없음
     - 높은 메모리 대역폭 사용량
     - 여러 번의 커널 실행 필요

이 구현은 정확하지만 성능 면에서 최적이 아니며, 벤치마크 결과에서 CPU 버전보다 약간 느린 것을 확인할 수 있습니다. Fused 구현에서는 다음을 통해 이러한 성능 한계를 해결합니다:

- 시퀀스당 통계량을 한 번만 계산
- 정규화된 값 재사용
- 메모리 트래픽 감소
- 중간 텐서 할당 제거

</div>
</details>

### 2. Fused 커널 구현

Fused 커널은 LayerNorm과 Linear 연산을 하나의 GPU 커널로 결합합니다:

```mojo
{{#include ../../../../../problems/p22/op/layernorm_linear.mojo:minimal_fused_forward_kernel}}
```

**핵심 최적화:**

- 두 번 대신 한 번의 커널 실행
- 중간 결과를 위한 공유 메모리 활용
- 병합 메모리 접근 패턴
- 메모리 대역폭 사용량 절감
- 중간 텐서 할당 불필요

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. **스레드 구성**:
   - 시퀀스 위치당 하나의 스레드 블록 (그리드: `[batch_size, seq_len]`)
   - 중복을 방지하기 위해 시퀀스 위치당 단일 스레드
   - 각 시퀀스 위치의 모든 출력을 하나의 스레드에서 계산

2. **메모리 접근**:
   - 입력 텐서: `[batch_idx, seq_idx, h]`로 접근
   - 출력 텐서: `[batch_idx, seq_idx, out_idx]`로 접근
   - 가중치: 선형 레이어에서 `[out_idx, h]`로 접근

3. **연산 흐름**:
   - 시퀀스당 LayerNorm 통계량을 한 번만 계산
   - 모든 출력 차원에 정규화된 값을 재사용
   - 정규화와 선형 변환을 결합

4. **성능**:
   - 통계량의 중복 계산 방지
   - 연산을 결합하여 메모리 트래픽 최소화
   - `rebind[Scalar[dtype]]`로 적절한 타입 캐스팅 사용

</div>
</details>

### 코드 실행

Fused 구현을 테스트하려면 다음을 실행하세요:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p22 --fused
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p22 --fused
```

  </div>
  <div class="tab-content">

```bash
uv run poe p22 --fused
```

  </div>
</div>

출력은 다음과 같습니다:

```txt
Testing with dimensions: [4, 4, 8] -> [4, 4, 16]
✅ Loaded Mojo operations library
============================================================
   Puzzle 22: FUSED Algorithm Test & Benchmark
============================================================

🧪 Correctness Testing for FUSED Algorithm
==================================================

Testing Reference PyTorch Implementation
-----------------------------------------------
✅ Reference PyTorch
   Max difference: 0.00e+00
   Result: ✅ CORRECT

Testing CPU Implementation
---------------------------------
✅ Using Mojo fused kernel (CPU)
   Max difference: 1.86e-08
   Result: ✅ CORRECT

Testing GPU Fused Implementation
---------------------------------------
✅ Using Mojo fused kernel (GPU)
   Max difference: 1.86e-08
   Result: ✅ CORRECT

Correctness Summary:
   - Reference:   ✅ CORRECT
   - CPU:         ✅ CORRECT
   - GPU fused: ✅ CORRECT

   Overall Correctness: ✅ ALL CORRECT

⚡ Benchmarking CPU vs GPU FUSED
----------------------------------------
   Testing CPU performance...
   CPU: 3144.75ms (50 iterations)
   Testing GPU fused performance...
   GPU fused: 3116.11ms (50 iterations)

   GPU fused vs CPU: 1.01x faster
   GPU fused wins!

FUSED Algorithm Test Completed!
```

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p22/op/layernorm_linear.mojo:minimal_fused_forward_kernel_solution}}
```

<div class="solution-explanation">

Fused 구현은 연산들을 효율적으로 결합합니다:

1. **스레드 구성**:
   - 시퀀스 위치당 하나의 스레드 블록 (그리드: `[batch_size, seq_len]`)
   - 시퀀스 위치당 단일 스레드
   - 스레드 인덱스: `batch_idx = block_idx.x`, `seq_idx = block_idx.y`

2. **LayerNorm 단계**:
   - 시퀀스 위치에 대한 합계와 제곱합 계산
   - 평균 계산: \\[\Large \mu = \frac{1}{H} \sum_{i=1}^{H} x_i \\]
   - 분산 계산: \\[\Large \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\]
   - 역표준편차 계산: \\[\Large \text{inv\_std} = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \\]

3. **Linear 단계**:
   - 각 출력 차원에 대해:
     - 정규화된 값 계산: \\[\Large \text{normalized} = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \\]
     - 선형 가중치와 곱하고 누적: \\[\Large \text{acc} = \sum_{h=1}^{H} \text{normalized}_h \cdot W_{out,h} \\]
     - 선형 bias 추가: \\[\Large \text{output} = \text{acc} + b_{out} \\]
   - 결과를 `output[batch_idx, seq_idx, out_idx]`에 저장

4. **성능 최적화**:
   - 두 연산을 위한 단일 커널 실행
   - 계산된 통계량 재사용
   - 메모리 트래픽 최소화
   - 중간 텐서 할당 불필요
   - 효율적인 메모리 접근 패턴

이 구현은 메모리 대역폭 사용량과 커널 실행 오버헤드를 줄여 unfused 버전보다 더 나은 성능을 달성합니다.
</div>
</details>

## Kernel fusion의 장점

이 퍼즐에서 LayerNorm + Linear 연산을 구현하는 두 가지 방식을 살펴보았습니다:

1. **Unfused 구현**:
   - LayerNorm과 Linear를 별도의 커널로 실행
   - 구현이 간단하지만 덜 효율적
   - 높은 메모리 대역폭 사용량
   - 여러 번의 커널 실행
   - 벤치마크 결과: 3183.57ms (GPU)

2. **Fused 구현**:
   - 두 연산을 결합한 단일 커널
   - 더 복잡하지만 훨씬 효율적
   - 메모리 대역폭 사용량 절감
   - 단일 커널 실행
   - 벤치마크 결과: 3116.11ms (GPU)

### 메모리 대역폭 최적화

1. **메모리 트래픽 제거**:
   - 연산 간 중간 텐서 할당 불필요
   - 글로벌 메모리 읽기/쓰기 감소
   - 선형 변환을 위한 정규화된 값 재사용
   - 메모리 대역폭 절감률: \\[\Large \text{reduction} = \frac{\text{unfused\_bandwidth} - \text{fused\_bandwidth}}{\text{unfused\_bandwidth}}\\]

2. **캐시 효율**:
   - L1/L2 캐시 활용도 향상
   - 캐시 미스 감소
   - 개선된 메모리 접근 패턴
   - 더 높은 산술 강도

### 오버헤드 감소

1. **커널 실행 최적화**:
   - 여러 번 대신 단일 커널 실행
   - 드라이버 오버헤드 감소
   - 동기화 지점 감소
   - 메모리 할당 횟수 감소

2. **리소스 관리**:
   - 연산 간 공유 메모리 재사용
   - 레지스터 활용도 향상
   - 스레드 점유율 개선
   - GPU 활용률 향상

### 성능 특성

1. **확장성**:
   - 입력 크기에 따른 성능 확장성 향상
   - 메모리 대역폭 병목 감소
   - GPU 리소스의 더 효율적인 활용
   - 대규모 모델에서 처리량 향상

2. **수치적 효율**:
   - 수치 안정성 유지
   - 반올림 오차 감소
   - 중간 결과의 정밀도 향상
   - 최적화된 연산 순서

💡 **핵심 통찰**: Kernel fusion은 트랜스포머 아키텍처의 LayerNorm + Linear처럼 neural networks에서 자주 함께 사용되는 연산에 특히 유리합니다. 입력 크기가 크고 모델이 복잡할수록 성능 이점은 더욱 커집니다.
