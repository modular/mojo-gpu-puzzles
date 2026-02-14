<!-- i18n-source-commit: 2bb7071f5afbc671e252b0dc6902b520793c11d5 -->

# Puzzle 19: Attention Op

## 개요

이 퍼즐에서는 어텐션 메커니즘을 커스텀 MAX Graph 연산으로 구현합니다. 어텐션은 [트랜스포머](https://arxiv.org/abs/1706.03762)와 함께 널리 알려진 현대 신경망의 핵심 요소로, 모델이 예측할 때 입력에서 관련된 부분에 집중할 수 있게 해줍니다.

수학적으로 어텐션 함수는 다음과 같이 정의됩니다:

$$\\Large \\text{Attention}(Q, K, V) = \\text{softmax}(Q \\cdot K^T) \\cdot V$$

여기서:

- \\(Q\\)는 shape \\((d,)~\\)의 **쿼리 벡터** - 찾으려는 대상을 나타냅니다
- \\(K\\)는 shape \\((\text{seq\_len}, d)~\\)의 **키 행렬** - 매칭할 수 있는 대상을 나타냅니다
- \\(V\\)는 shape \\((\text{seq\_len}, d)~\\)의 **값 행렬** - 검색할 정보를 나타냅니다
- 출력은 shape \\((d,)\\)의 **가중합** 벡터입니다

연산은 세 가지 주요 단계로 이루어집니다:

1. **어텐션 점수**: \\(Q \cdot K^T\\)를 계산하여 쿼리가 각 키 벡터와 얼마나 잘 매칭되는지 측정합니다
2. **어텐션 가중치**: softmax를 적용하여 점수를 확률 분포로 변환합니다 (가중치의 합 = 1)
3. **가중 합**: 어텐션 가중치를 사용하여 값 벡터들을 결합해 최종 출력을 생성합니다

## 어텐션 이해하기: 단계별 분석

어텐션을 **스마트 검색 메커니즘**으로 생각해 보세요. 쿼리(찾고자 하는 것)가 주어지면, 어텐션은 키-값 쌍의 모음에서 가장 관련성 높은 정보를 찾아냅니다:

1. **1단계 - 유사도 매칭**: 쿼리 \\(Q\\)를 모든 키 \\(K\\)와 비교하여 유사도 점수를 구합니다
   - \\(Q \cdot K^T\\)를 계산하여 \\(Q\\)가 각 키 벡터와 얼마나 잘 매칭되는지 측정합니다
   - 높은 점수 = 더 좋은 매칭

2. **2단계 - 확률 분포**: 원시 점수를 정규화된 가중치로 변환합니다
   - softmax를 적용하여 모든 가중치의 합이 1.0이 되도록 합니다
   - 어떤 값에 집중할지에 대한 확률 분포를 만듭니다

3. **3단계 - 가중 검색**: 어텐션 가중치를 사용하여 값들을 결합합니다
   - 각 값 벡터에 해당하는 가중치를 곱합니다
   - 모든 것을 더해 최종 출력을 구합니다

**실생활 비유**: 도서관에서 검색하는 것을 상상해 보세요. 쿼리는 찾고 싶은 것이고, 책 제목은 키이며, 책 내용은 값입니다. 어텐션은 각 책이 쿼리와 얼마나 관련 있는지 계산한 다음, 관련도에 따라 가중 요약을 제공합니다.

### 연산 흐름 시각화

```
Input:  Q(16,)    K(16,16)    V(16,16)
         ↓           ↓           ↓
Step 1: Q(1,16) @ K^T(16,16) → Scores(1,16)
         ↓
Step 2: softmax(Scores) → Weights(1,16)  [sum = 1.0]
         ↓
Step 3: Weights(1,16) @ V(16,16) → Output(1,16) → reshape → Output(16,)
```

**핵심 아이디어**: 쿼리 벡터 \\(Q\\)의 shape \\((16,)\\)에서 \\((1,16)\\)으로 reshape하여 내적 대신 행렬 곱셈을 사용할 수 있게 합니다. 이를 통해 Puzzle 18의 고도로 최적화된 tiled matmul 커널을 활용할 수 있습니다!

GPU 구현은 **이전 퍼즐에서 최적화된 커널들을 재사용하고 결합합니다**:

- **[Puzzle 16의 tiled 행렬 곱셈](../puzzle_16/puzzle_16.md)** — 효율적인 \\(Q \cdot K^T\\) 및 \\(\text{weights} \cdot V\\) 연산에 사용
- **공유 메모리 전치** — \\(K^T\\)를 효율적으로 계산
- **[Puzzle 18의 병렬 softmax](../puzzle_18/puzzle_18.md)** — 수치적으로 안정적인 어텐션 가중치 계산에 사용

> **🔄 커널 재사용 전략**: 이 퍼즐은 이전 퍼즐에서 검증된 최적화 커널들을 결합하여 복잡한 연산을 구축하는 방법을 보여줍니다. 모든 것을 처음부터 작성하는 대신, Puzzle 16의 `matmul_idiomatic_tiled`과 Puzzle 18의 `softmax_kernel`을 활용하여 모듈형 GPU 커널 설계의 강력함을 보여줍니다.

## 핵심 개념

- 시퀀스 처리를 위한 벡터 어텐션 메커니즘
- **커널 재사용**: [Puzzle 16](../puzzle_16/puzzle_16.md)과 [Puzzle 18](../puzzle_18/puzzle_18.md)의 검증된 구현 활용
- 공유 메모리 tiling을 활용한 효율적인 행렬 곱셈
- 버퍼 할당을 최소화하는 메모리 최적화 텐서 reshape
- 여러 최적화 커널을 단일 연산으로 통합
- 다중 입력을 지원하는 커스텀 MAX Graph 연산
- 호환성을 위한 CPU 폴백 구현

## 설정

- **시퀀스 길이**: \\(\text{SEQ\_LEN} = 16~\\) - 시퀀스 내 키/값 벡터의 수
- **모델 차원**: \\(\text{D} = 16~\\) - 각 벡터(쿼리, 키, 값)의 차원
- **블록당 스레드 수**: 각 커널에 맞게 개별 최적화
- **그리드 차원**: 다양한 행렬 크기를 효율적으로 처리하도록 동적으로 계산
- **공유 메모리**: 전치, matmul, softmax kernel에서 성능을 위해 활용

레이아웃 설정:

- 쿼리 텐서: `Layout.row_major(d)`
- 키 텐서: `Layout.row_major(seq_len, d)`
- 값 텐서: `Layout.row_major(seq_len, d)`
- 출력 텐서: `Layout.row_major(d)`
- 커스텀 op 파라미터: `{"seq_len": seq_len, "d": d, "dtype": dtype}`

이 퍼즐의 핵심 요소는 다음과 같습니다:

1. **다중 커널 오케스트레이션**: 전치, matmul, softmax 연산의 결합
2. **메모리 최적화**: reshape 연산과 버퍼 재사용으로 메모리 할당 최소화
3. **수치 안정성**: [Puzzle 18](../puzzle_18/puzzle_18.md)의 검증된 softmax 구현 활용
4. **성능 최적화**: 모든 행렬 연산에 [Puzzle 16](../puzzle_16/puzzle_16.md)의 tiled 알고리즘 사용
5. **다중 입력 연산**: 단일 커스텀 op에서 세 개의 입력 텐서(Q, K, V) 처리

어텐션 커스텀 연산은 다음과 같은 일을 수행합니다:

- Python에서 쿼리, 키, 값 텐서를 입력으로 받기
- 최적화된 커널을 사용하여 GPU에서 효율적으로 처리
- 어텐션 가중 출력 벡터 반환
- NumPy 참조 구현 결과와 일치

## 완성할 코드

이 퍼즐을 완성하려면 [Puzzle 16](../puzzle_16/puzzle_16.md)의 tiled matmul kernel과 [Puzzle 18](../puzzle_18/puzzle_18.md)의 softmax kernel을 활용합니다. 공유 메모리를 사용하여 Mojo 파일에서 전치 커널만 구현하면 됩니다.

### 1. 전치 커널 구현하기

```mojo
{{#include ../../../../../problems/p19/op/attention.mojo:transpose_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p19/op/attention.mojo" class="filename">전체 파일 보기: problems/p19/op/attention.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

**전치 커널 구현 가이드:**

1. **공유 메모리 설정**: `LayoutTensor[dtype, Layout.row_major(TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`을 사용하여 `TRANSPOSE_BLOCK_DIM_XY` × `TRANSPOSE_BLOCK_DIM_XY` 크기의 정사각형 공유 메모리 타일을 생성합니다. 이를 통해 스레드 간 효율적인 데이터 교환이 가능합니다.

2. **스레드 인덱싱**: 스레드를 행렬 요소에 매핑합니다:
   - `local_row = thread_idx.y`, `local_col = thread_idx.x` (블록 내 위치)
   - `global_row = block_idx.y * TRANSPOSE_BLOCK_DIM_XY + local_row` (전체 행렬에서의 위치)

3. **2단계 연산**:
   - **1단계**: 글로벌 메모리에서 공유 메모리로 일반 인덱싱으로 데이터를 로드합니다
   - **2단계**: 공유 메모리에서 글로벌 메모리로 뒤바꾼 인덱싱으로 데이터를 저장합니다

4. **필수 동기화**: 로드와 저장 사이에 `barrier()`를 호출하여 모든 스레드가 로드를 완료한 후에야 저장을 시작하도록 보장합니다

5. **전치의 핵심**: 전치는 뒤바꾼 인덱싱을 통해 이루어집니다: `shared_tile[local_row, local_col]` 대신 `shared_tile[local_col, local_row]`를 사용합니다

6. **경계 처리**: 글로벌 메모리 접근 시 경계 검사를 수행하여 `TRANSPOSE_BLOCK_DIM_XY` x `TRANSPOSE_BLOCK_DIM_XY`로 정확히 나누어지지 않는 행렬에서 범위를 벗어난 읽기/쓰기를 방지합니다

7. **메모리 병합**: 이 패턴은 읽기와 쓰기 모두 병합되도록 보장하여 최적의 메모리 대역폭을 활용합니다

</div>
</details>

### 2. 어텐션 오케스트레이션

```mojo
{{#include ../../../../../problems/p19/op/attention.mojo:attention_orchestration}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p19/op/attention.mojo" class="filename">전체 파일 보기: problems/p19/op/attention.mojo</a>

### 커널 테스트

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p19
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p19
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p19
```

  </div>
  <div class="tab-content">

```bash
uv run poe p19
```

  </div>
</div>

성공하면 CPU와 GPU에서 다음과 비슷한 출력을 볼 수 있습니다:

```
Input shapes: Q=(16,), K=(16, 16), V=(16, 16)
Sample Q values: [ 0.04967142 -0.01382643  0.06476886  0.15230298 -0.02341534]
Sample K[0] values: [-0.10128311  0.03142473 -0.09080241 -0.14123037  0.14656489]
Sample V[0] values: [ 0.11631638  0.00102331 -0.09815087  0.04621035  0.01990597]

================================================================================
STEP-BY-STEP VECTOR ATTENTION COMPUTATION DEBUG
================================================================================

1. INPUT SHAPES:
   Q shape: (16,) (query vector)
   K shape: (16, 16) (key matrix)
   V shape: (16, 16) (value matrix)
   Q[:5]: [ 0.04967142 -0.01382643  0.06476886  0.15230298 -0.02341534]

2. ATTENTION SCORES (K[i] · Q):
   Scores shape: (16,)
   Scores[:5]: [-0.03479404 -0.01563787  0.04834607  0.06764711  0.04001468]
   Min: -0.061636, Max: 0.067647
   Manual verification:
     Q · K[0] = K[0] · Q = -0.034794 (computed: -0.034794)
     Q · K[1] = K[1] · Q = -0.015638 (computed: -0.015638)
     Q · K[2] = K[2] · Q = 0.048346 (computed: 0.048346)

3. SOFTMAX:
   Max score: 0.067647
   Attention weights shape: (16,)
   Attention weights[:5]: [0.05981331 0.06097015 0.06499878 0.0662655  0.06445949]
   Sum: 1.000000 (should be 1.0)

4. WEIGHTED SUM OF VALUES:
   Output shape: (16,)
   Output[:5]: [-0.00935538 -0.0243433   0.00306551  0.02346884  0.019306  ]
   Output norm: 0.092764
   Manual output[:5]: [-0.00935538 -0.0243433   0.00306551  0.02346884  0.019306  ]
   Match: True

================================================================================
TESTING INDIVIDUAL OPERATIONS
================================================================================

Test 1: Vector Dot Product
a · b = 3.000000

Test 2: Matrix-Vector Multiplication
M @ v = [ 3.  7. 11.]

Test 3: Softmax
Input: [1. 2. 3. 4.]
Softmax: [0.0320586  0.08714432 0.2368828  0.6439143 ]
Sum: 1.000000

================================================================================
TESTING FULL ATTENTION
================================================================================
Compiling attention graph on Device(type=cpu,id=0)
Executing attention on Device(type=cpu,id=0)
====================================================================================================

CPU attention output[:5]: [-0.00935538 -0.02434331  0.00306551  0.02346884  0.019306  ]
CPU matches NumPy: True
Compiling attention graph on Device(type=gpu,id=0)
Executing attention on Device(type=gpu,id=0)
====================================================================================================

GPU attention output[:5]: [-0.00935538 -0.0243433   0.00306551  0.02346884  0.019306  ]
Expected output[:5]: [-0.00935538 -0.0243433   0.00306551  0.02346884  0.019306  ]
GPU matches NumPy: True

================================================================================
FINAL VERIFICATION
================================================================================
✓ CPU implementation PASSED
✓ GPU implementation PASSED

Output vector norms:
  CPU: 0.092764
  GPU: 0.092764
  Expected: 0.092764
```

이 출력은 커스텀 MAX Graph 연산이 어텐션 알고리즘을 올바르게 구현하여 NumPy 참조 구현과 일치하는 결과를 생성했음을 보여줍니다.

## 풀이

<details class="solution-details">
<summary></summary>

이 퍼즐을 풀려면 Mojo에서 전치 커널을 구현하고 어텐션 커스텀 연산을 위한 Python 그래프 정의를 완성해야 합니다. 이 퍼즐은 이전 퍼즐의 개념들을 기반으로, **[Puzzle 16](../puzzle_16/puzzle_16.md)의 tiled 행렬 곱셈**과 **[Puzzle 18](../puzzle_18/puzzle_18.md)의 softmax**를 결합하여 완전한 어텐션 메커니즘을 구성합니다.

### 재사용 커널

구현에서 다음의 검증된 커널들을 직접 활용합니다:

1. **`matmul_idiomatic_tiled`** ([Puzzle 16](../puzzle_16/puzzle_16.md)) - \\(Q \\times K^T\\)와 \\(\\text{weights} \\times V\\) 연산 모두를 수행
2. **`softmax_kernel`** ([Puzzle 18](../puzzle_18/puzzle_18.md)) - 수치적으로 안정적인 어텐션 가중치 계산 제공

이는 **모듈형 GPU 아키텍처**의 좋은 예시입니다: 단일 구현체가 아닌, 검증된 최적화 컴포넌트를 오케스트레이션하여 복잡한 신경망 연산을 구축합니다.

어텐션 연산은 표준적인 수학적 정의를 따릅니다:

$$\\Large \\text{Attention}(Q, K, V) = \\text{softmax}(Q \\cdot K^T) \\cdot V$$

**수식 분석**:

- \\(Q \cdot K^T~\\): 쿼리-키 유사도 점수, shape: \\((1, \text{seq\_len})\\)
- \\(\text{softmax}(\cdot)~\\): 점수를 확률로 정규화, shape: \\((1, \text{seq\_len})\\)
- \\(\text{weights} \cdot V~\\): 값의 가중 결합, shape: \\((1, d)\\)

이 과정에는 이전 퍼즐의 GPU 커널을 활용하여 최적화하는 여러 연산 단계가 포함됩니다.

### 1. 전치 커널 구현

```mojo
{{#include ../../../../../solutions/p19/op/attention.mojo:transpose_kernel_solution}}
```

<div class="solution-explanation">

전치 커널은 **공유 메모리 tiling**을 사용하여 병합 메모리 접근 패턴을 달성합니다. 핵심 구현 내용은 다음과 같습니다:

#### 핵심 전치 패턴

```mojo
# 일반 인덱싱으로 로드
shared_tile[local_row, local_col] = inp[global_row, global_col]
barrier()
# 뒤바꾼 인덱싱으로 저장하여 전치
output[out_row, out_col] = shared_tile[local_col, local_row]
```

전치는 공유 메모리 접근에서 **뒤바꾼 인덱싱**(`[local_row, local_col]` 대신 `[local_col, local_row]`)과 출력 위치 지정을 위한 **뒤바꾼 블록 좌표**를 통해 이루어집니다. 이를 통해 읽기와 쓰기 모두 병합을 유지하면서 전치 연산을 수행합니다.
</div>

### 2. GPU 커널 오케스트레이션

```mojo
{{#include ../../../../../solutions/p19/op/attention.mojo:attention_orchestration_solution}}
```

<div class="solution-explanation">

GPU 오케스트레이션은 **정교한 커널 체이닝**과 **제로 카피 메모리 최적화**를 보여줍니다:

#### 고급 메모리 최적화 전략

```mojo
# 제로 카피 reshape - 데이터 이동 없이 텐서 shape만 재해석
q_2d = q_tensor.reshape[layout_q_2d]()
# 적극적인 버퍼 재사용 - 같은 메모리, 다른 해석
weights = scores_2d.reshape[layout_scores]()
```

구현은 다음을 통해 **최대 메모리 효율**을 달성합니다:

- **제로 카피 reshape**: 메모리에서 데이터를 이동하지 않고 텐서 shape을 재해석
- **지능적 버퍼 재사용**: 동일한 `scores_weights_buf`가 점수 \\((1,\\text{seq_len})\\)와 가중치 \\((\\text{seq_len},)\\) 이중 용도로 활용
- **최소 할당**: 단 2개의 임시 버퍼로 전체 어텐션 연산 수행
- **메모리 병합**: 모든 연산에서 최적의 메모리 접근 패턴 유지

#### 전략적 커널 재사용 패턴

- **3단계 & 7단계**: 둘 다 [Puzzle 16](../puzzle_16/puzzle_16.md)의 `matmul_idiomatic_tiled` 활용
  - 3단계: \\(Q \\times K^T\\) → 어텐션 점수 계산 \\((1,d) \\times (d,\\text{seq_len}) \\rightarrow (1,\\text{seq_len})\\)
  - 7단계: \\(\\text{weights} \\times V\\) → 최종 가중 출력 \\((1,\\text{seq_len}) \\times (\\text{seq_len},d) \\rightarrow (1,d)\\)
  - 두 연산 모두 다양한 행렬 크기를 안전하게 처리하기 위해 경계 검사 포함
- **5단계**: [Puzzle 18](../puzzle_18/puzzle_18.md)의 `softmax_kernel` 사용
  - 원시 점수를 정규화된 확률 분포로 변환
  - 최댓값 차감과 병렬 reduction을 통한 수치 안정성 보장
  - \\(\\sum_{i} \\text{weights}[i] = 1.0\\) 보장

이는 **모듈형 GPU 아키텍처**의 좋은 예시입니다: 단일 구현체가 아닌, 검증된 최적화 커널들을 오케스트레이션하여 복잡한 신경망 연산을 구축합니다!
</div>

### 핵심 구현 인사이트

<div class="solution-explanation">

#### 메모리 최적화 전략

적극적인 버퍼 재사용으로 **메모리 할당을 최소화**합니다:

```mojo
# 전체 연산에 필요한 임시 버퍼는 단 2개
k_t_buf = gpu_ctx.enqueue_create_buffer[dtype](seq_len * d)
scores_weights_buf = gpu_ctx.enqueue_create_buffer[dtype](seq_len)
```

**핵심 최적화 포인트**:

- 동일한 `scores_weights_buf`가 reshape 연산을 통해 어텐션 점수와 가중치 모두에 재사용됩니다
- 제로 카피 텐서 reshape으로 불필요한 데이터 이동을 제거합니다

#### 커널 재사용 아키텍처

이 퍼즐은 세 가지 특화된 커널을 결합하여 **모듈형 커널 설계**를 보여줍니다:

- **`matmul_idiomatic_tiled`** (2회 사용) - \\(Q \\times K^T\\)와 \\(\\text{weights} \\times V\\) 연산 모두를 수행
- **`softmax_kernel`** - 병렬 reduction을 활용하여 수치적으로 안정적인 어텐션 가중치 계산
- **`transpose_kernel`** - 병합 메모리 접근으로 효율적인 \\(K^T\\) 계산

**아키텍처의 장점**:

- **조합 가능성**: 검증된 컴포넌트로 복잡한 연산 구축
- **유지보수성**: 각 커널이 명확하게 정의된 단일 역할 수행
- **성능**: 이전 퍼즐의 고도로 최적화된 구현 활용
- **확장성**: 모듈형 설계로 더 큰 어텐션 메커니즘으로 확장 용이

이 구현은 **정교한 신경망 연산**이 단일 구현체가 아닌, 더 단순하고 잘 검증된 GPU 커널들을 오케스트레이션하여 구축할 수 있음을 보여줍니다.
</div>

</details>
