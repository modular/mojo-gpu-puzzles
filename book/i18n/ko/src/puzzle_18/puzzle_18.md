<!-- i18n-source-commit: 51143596e241ae5954474ecb3133b1d7b147f6fc -->

# Puzzle 18: Softmax Op

## 개요

이 퍼즐에서는 softmax 함수를 커스텀 MAX Graph 연산으로 구현합니다. Softmax는 실수 벡터를 받아 확률 분포로 정규화하는 함수입니다.

Softmax 함수는 두 가지 주요 단계로 동작합니다:

1. 지수 함수 적용: 입력 벡터의 각 요소에 지수 함수를 적용합니다. 이를 통해 모든 값이 양수가 되고 값 사이의 차이가 증폭됩니다. 큰 입력값은 훨씬 큰 지수 출력을 만들고, 작거나 음수인 값은 0에 가까운 출력을 만들어냅니다.

2. 정규화: 각 지수 값을 모든 지수 값의 합으로 나눕니다. 이 정규화 단계를 통해 결과값이 유효한 확률 분포가 됩니다. 즉, 모든 값이 0과 1 사이이고 합이 정확히 1이 됩니다.

수학적으로 softmax 함수는 다음과 같이 정의됩니다:

$$\Large \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

여기서:

- \\(x_i\\)는 입력 벡터의 \\(i\\)번째 요소
- \\(n\\)은 입력 벡터의 길이

그러나 이 직접적인 구현은 값이 클 때 수치 오버플로우 문제를 일으킬 수 있습니다. 이를 해결하기 위해 수치적으로 더 안정적인 버전을 사용합니다:

$$\Large \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}$$

GPU 구현에서는 최댓값 찾기와 지수 합 계산 모두에 병렬 reduction을 사용하여 큰 벡터에서도 높은 효율을 달성합니다.

## 핵심 개념

- 효율적인 최댓값 및 합계 계산을 위한 병렬 reduction
- 최댓값 차감 기법을 통한 수치 안정성
- 스레드 간 통신을 위한 공유 메모리 활용
- 커스텀 MAX Graph 연산의 Python 통합
- Barrier를 통한 스레드 동기화

## 설정

- 벡터 크기: `SIZE = 128`
- 블록당 스레드 수: `BLOCK_DIM_X = 1 << log2_ceil(SIZE)`. 트리 기반 reduction이 올바르게 동작하려면 `BLOCK_DIM_X`가 `SIZE` 이상인 가장 작은 2의 거듭제곱이어야 합니다.
- 그리드 차원: \\(1 \times 1\\) 블록
- 공유 메모리: 최댓값과 합계를 위한 두 개의 공유 변수

레이아웃 설정:

- 입력 텐서: `Layout.row_major(SIZE)`
- 출력 텐서: `Layout.row_major(SIZE)`
- 커스텀 op 파라미터: `{"input_size": input_tensor.shape[0]}`

이 퍼즐의 핵심 요소는 다음과 같습니다:

1. **수치 안정성**: 잠재적인 수치 문제를 처리하는 방법 이해하기
2. **병렬 reduction**: 공유 메모리를 사용한 효율적인 최댓값 및 합계 계산
3. **커스텀 op 통합**: Mojo GPU kernel을 위한 Python 인터페이스 완성하기
4. **테스트와 검증**: 구현이 기대 결과와 일치하는지 확인하기

Softmax 커스텀 연산은 다음과 같은 일을 수행합니다:

- Python에서 NumPy 배열을 입력으로 받기
- GPU에서 효율적으로 처리하기
- 정규화된 확률 분포를 반환하기
- SciPy의 softmax 구현 결과와 일치시키기

## 완성할 코드

이 퍼즐을 완성하려면 Mojo 파일에서 GPU와 CPU kernel을 모두 구현하고, Python 코드에서 그래프 정의를 완성해야 합니다.

### 1. `softmax.mojo`에서 GPU kernel 구현하기

```mojo
{{#include ../../../../../problems/p18/op/softmax.mojo:softmax_gpu_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p18/op/softmax.mojo" class="filename">전체 파일 보기: problems/p18/op/softmax.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 모든 스레드가 접근할 수 있도록 최댓값과 합계 모두에 공유 메모리를 사용하세요
2. 스레드를 동기화하기 위해 적절한 지점에서 `barrier()`를 호출하는 것을 잊지 마세요
3. 각 스레드가 입력 배열의 일부를 처리하도록 병렬 reduction을 구현하세요
4. 스레드 분기를 최소화하기 위해 트리 기반 reduction 패턴을 사용하세요
5. 특히 큰 입력에서 범위를 벗어난 접근을 주의 깊게 처리하세요
6. 수치 안정성을 위해 \\(e^{x_i}\\) 대신 \\(e^{x_i - max}\\)를 계산하세요

</div>
</details>

### 2. `softmax.mojo`에서 CPU kernel 구현하기

```mojo
{{#include ../../../../../problems/p18/op/softmax.mojo:softmax_cpu_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p18/op/softmax.mojo" class="filename">전체 파일 보기: problems/p18/op/softmax.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. GPU 버전과 동일한 수학적 단계를 따르는 순차적 구현을 작성하세요
2. 먼저 모든 입력에서 최댓값을 찾으세요
3. 그다음 각 요소에 대해 \\(e^{x_i - max}\\)를 계산하고 합계를 누적하세요
4. 마지막으로 각 요소를 합계로 나눠 정규화하세요
5. CPU 구현에는 병렬 스레드가 없으므로 스칼라 연산을 사용하세요

</div>
</details>

### CPU와 GPU kernel 테스트

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p18-test-kernels
```

  </div>
  <div class="tab-content">

```bash
pixi run p18-test-kernels
```

  </div>
</div>

올바르게 구현하면 다음과 같이 출력됩니다:

```txt
Total Discovered Tests: 1

Passed : 1 (100.00%)
Failed : 0 (0.00%)
Skipped: 0 (0.00%)
```

### 3. `p18.py`에서 그래프 정의 완성하기

```python
{{#include ../../../../../problems/p18/p18.py:softmax_custom_op_graph}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p18/p18.py" class="filename">전체 파일 보기: problems/p18/p18.py</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. `graph.inputs[0]`으로 그래프에 전달된 입력 텐서에 접근하세요
2. 등록한 커스텀 op 이름("softmax")으로 `ops.custom()`을 호출하세요
3. 입력 텐서를 커스텀 연산의 값으로 전달하세요
4. 입력 shape과 일치하는 출력 타입을 지정하세요
5. kernel에 필요한 "input_size" 파라미터를 포함하세요
6. `graph.outputs`를 연산의 출력 텐서가 담긴 리스트로 설정하세요

</div>
</details>

다음 명령으로 퍼즐을 실행할 수 있습니다:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p18
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p18
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p18
```

  </div>
  <div class="tab-content">

```bash
uv run poe p18
```

  </div>
</div>

성공하면 CPU와 GPU에서 다음과 비슷한 출력을 볼 수 있습니다:

```
Input shape: (128,)
First few random input values: [ 1.1810775   0.60472375  0.5718309   0.6644599  -0.08899796]
Compiling softmax graph on Device(type=cpu,id=0)
Executing softmax on Device(type=cpu,id=0)
====================================================================================================
Compiling softmax graph on Device(type=gpu,id=0)
Executing softmax on Device(type=gpu,id=0)
====================================================================================================
First few softmax results on CPU (custom Mojo kernel): [0.01718348 0.00965615 0.0093437  0.01025055 0.0048253 ]
First few softmax results on GPU (custom Mojo kernel): [0.01718348 0.00965615 0.0093437  0.01025055 0.0048253 ]
First few expected results (SciPy calculation): [0.01718348 0.00965615 0.0093437  0.01025055 0.0048253 ]
Verification passed: Custom kernel results match SciPy calculation
Sum of all probabilities on CPU: 1.0
Sum of all probabilities on GPU: 1.0
```

이 출력은 커스텀 MAX Graph 연산이 softmax 알고리즘을 올바르게 구현하여 유효한 확률 분포를 생성했음을 보여줍니다.

## 풀이

<details class="solution-details">
<summary></summary>

이 퍼즐을 풀려면 Mojo kernel(GPU와 CPU)과 Python 그래프 정의를 모두 구현해야 합니다. [Puzzle 17: 1D Convolution Op](../puzzle_17/puzzle_17.md)에서 했던 것처럼, Python의 생태계와 Mojo의 GPU 가속 컴퓨팅 역량을 잇는 다리를 만듭니다.

구현할 softmax 연산은 수학적으로 다음과 같이 정의됩니다:

$$\Large \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

하지만 수치 오버플로우를 방지하기 위해 더 안정적인 형태를 사용합니다:

$$\Large \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}$$

### GPU kernel 구현

```mojo
{{#include ../../../../../solutions/p18/op/softmax.mojo:softmax_gpu_kernel_solution}}
```

<div class="solution-explanation">
GPU kernel은 고도로 최적화된 병렬 reduction 기법을 사용하여 수치적으로 안정적인 softmax 알고리즘을 구현합니다. Kernel을 상세히 분석해 보겠습니다:

#### Kernel 시그니처와 메모리 관리

```mojo
fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
)
```

Kernel의 파라미터 구성:

- 입출력 텐서에 공통으로 사용되는 레이아웃 파라미터
- 정수 파라미터로 지정되는 벡터 크기
- 기본값이 float32인 설정 가능한 데이터 타입
- 연산 결과를 직접 저장하는 변경 가능한(mutable) 출력 텐서
- 변경 불가능한(mut=False) 입력 텐서

#### 공유 메모리 할당

```mojo
shared_max = LayoutTensor[dtype, Layout.row_major(BLOCK_DIM_X), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
shared_sum = LayoutTensor[dtype, Layout.row_major(BLOCK_DIM_X), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
```

Kernel은 두 개의 공유 메모리 버퍼를 할당합니다:

- `shared_max`: 병렬 최댓값 탐색 reduction용
- `shared_sum`: 병렬 합계 연산용
- 둘 다 `BLOCK_DIM_X = 128` 크기를 사용
- 공유 메모리는 블록 내 모든 스레드에 빠른 접근을 제공

#### 스레드 인덱싱

```mojo
global_i = thread_idx.x
```

이 softmax 구현은 단일 1D 스레드 블록에서 동작합니다. 즉, 전역 인덱스와 로컬 인덱스가 동일합니다.

#### 최댓값 탐색 단계

```mojo
var val: Scalar[dtype] = min_finite[dtype]()
if global_i < input_size:
    val = rebind[Scalar[dtype]](input[global_i])

shared_max[local_i] = val
barrier()
```

각 스레드를 다음과 같이 초기화합니다:

- 유효 범위를 벗어난 요소에는 최소 유한(finite) 값 할당
- 유효한 요소에 매핑되는 스레드에는 실제 입력값 할당
- Reduction 과정을 위해 공유 메모리에 저장
- 모든 스레드의 메모리 쓰기가 완료되도록 barrier 동기화

#### 병렬 max reduction

```mojo
stride = BLOCK_DIM_X // 2
while stride > 0:
    if local_i < stride:
        shared_max[local_i] = max(shared_max[local_i], shared_max[local_i + stride])
    barrier()
    stride = stride // 2
```

병렬 트리 reduction 패턴을 구현합니다:

1. `stride = 64`(`BLOCK_DIM_X`의 절반)로 시작
2. 각 활성 스레드가 stride만큼 떨어진 두 값 비교
3. 더 작은 인덱스에 최댓값 저장
4. Barrier로 모든 스레드 동기화
5. Stride를 절반으로 줄이고 반복
6. \\(\log_2(BLOCK\_DIM\_X)~\\) 단계 후 `shared_max[0]`에 전체 최댓값이 담김

이 로그 reduction은 대규모 입력에서 선형 스캔보다 훨씬 빠릅니다.

#### 수치적으로 안정적인 지수 함수 적용

```mojo
block_max = shared_max[0]

var exp_val: Scalar[dtype] = 0.0
if global_i < input_size:
    exp_val = rebind[Scalar[dtype]](exp(val - block_max))
```

각 스레드가 수행하는 작업:

1. 공유 메모리에서 전체 최댓값 읽음
2. 지수 함수를 적용하기 전에 입력값에서 최댓값 차감
3. 이 차감이 수치 안정성의 핵심 — 오버플로우 방지
4. 가장 큰 지수가 \\(e^0 = 1\\)이 되고, 나머지는 모두 \\(e^{음수} < 1\\)

#### 병렬 sum reduction

```mojo
shared_sum[local_i] = exp_val
barrier()

stride = BLOCK_DIM_X // 2
while stride > 0:
    if local_i < stride:
        shared_sum[local_i] += shared_sum[local_i + stride]
    barrier()
    stride = stride // 2
```

두 번째 reduction 단계:

1. 모든 지수 값을 공유 메모리에 저장
2. max와 동일한 트리 기반 reduction 패턴 사용
3. 단, 최댓값 비교 대신 덧셈 수행
4. \\(\log_2(BLOCK\_DIM\_X)~\\) 단계 후 `shared_sum[0]`에 모든 지수 값의 총합이 담김

#### 최종 정규화

```mojo
block_sum = shared_sum[0]

if global_i < input_size:
    output[global_i] = exp_val / block_sum
```

각 스레드가 수행하는 작업:

1. 공유 메모리에서 총합을 읽음
2. 자신의 지수 값을 이 총합으로 나눔
3. 정규화된 확률을 출력 버퍼에 기록
4. 합이 1인 유효한 확률 분포 생성

#### 성능 특성

이 구현은 뛰어난 성능 특성을 갖습니다:

- **복잡도**: 순차적 접근의 \\(O(n)\\)에 비해 max와 sum 계산 모두 \\(O(\log n)\\)
- **메모리 효율**: 공유 메모리를 \\(2 \times BLOCK\_DIM\_X~\\) 요소만 사용
- **작업 효율**: 각 스레드가 약 \\(2 \times \log_2(BLOCK\_DIM\_X)~\\)회 연산 수행
- **부하 분산**: 각 스레드가 동일한 양의 작업 처리
- **동기화**: 필요한 곳에서만 최소한의 barrier 사용
- **메모리 접근**: 최적 대역폭을 위한 coalesced 글로벌 메모리 접근 패턴

이 알고리즘은 수치적으로도 견고합니다. 최댓값 차감 기법을 적용하여 신경망 활성화에서 흔한 넓은 범위의 값에서도 정밀도를 유지하며, 오버플로우/언더플로우 가능성을 처리합니다.
</div>

### CPU 폴백 구현

```mojo
{{#include ../../../../../solutions/p18/op/softmax.mojo:softmax_cpu_kernel_solution}}
```

<div class="solution-explanation">
CPU 구현은 같은 수학적 접근 방식을 따르되 단일 스레드 실행에 최적화된 순차적 폴백을 제공합니다. 각 단계를 분석해 보겠습니다:

1. **최댓값 탐색**:

   ```mojo
   var max_val: Scalar[dtype] = min_finite[dtype]()
   for i in range(input_size):
       max_val = max(max_val, rebind[Scalar[dtype]](input[i]))
   ```

   최소 유한값으로 초기화하고 배열을 선형 스캔하며 만난 최댓값을 추적합니다. \\(O(n)\\) 복잡도이지만, 병렬화할 코어가 많지 않은 CPU에서는 효율적으로 동작합니다.

2. **지수 함수 적용과 합산**:

   ```mojo
   var sum_exp: Scalar[dtype] = 0.0
   for i in range(input_size):
       var exp_val = rebind[Scalar[dtype]](exp(input[i] - max_val))
       output[i] = exp_val
       sum_exp += exp_val
   ```

   각 요소에 대해 \\(e^{x_i - max}\\)를 계산하고 결과를 출력 버퍼에 저장하면서 합계 \\(\sum_{j=1}^{n} e^{x_j - max}\\)를 한 번의 순회로 누적합니다. 별도의 반복문을 사용하는 것에 비해 메모리 연산을 최소화합니다.

3. **정규화**:

   ```mojo
   for i in range(input_size):
       output[i] = output[i] / sum_exp
   ```

   마지막으로 각 요소를 합계로 나눠 softmax 공식에 따른 올바른 확률 분포를 생성합니다:

   $$\Large \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{n} e^{x_j - \max(x)}}$$

CPU 구현은 동일한 수치 안정성 기법(최댓값 차감)을 사용하되, 병렬이 아닌 순차적 연산으로 처리합니다. 공유 메모리나 스레드 동기화를 다룰 필요가 없어 GPU 버전보다 단순하지만, 대규모 입력에서는 효율이 떨어집니다.

두 구현 모두 `@compiler.register("softmax")` 데코레이터를 통해 MAX Graph의 커스텀 연산 시스템에 등록되므로, 가용 여부에 따라 어느 디바이스에서든 매끄럽게 실행됩니다.
</div>

### Python 통합

```python
{{#include ../../../../../solutions/p18/p18.py:softmax_custom_op_graph_solution}}
```

<div class="solution-explanation">
Python 통합은 NumPy 배열과 최적화된 Mojo GPU kernel 사이에 매끄러운 다리를 만듭니다. 구현은 여러 핵심 구성 요소로 이뤄져 있습니다:

1. **그래프 설정과 구성**:

   ```python
   with Graph(
       "softmax_graph",
       input_types=[
           TensorType(
               dtype,
               shape=input_tensor.shape,
               device=DeviceRef.from_device(device),
           ),
       ],
       custom_extensions=[mojo_kernels],
   ) as graph:
   ```

   "softmax_graph"라는 이름의 연산 그래프를 생성합니다:
   - 적절한 dtype과 shape으로 입력 텐서 타입 정의
   - 텐서를 대상 디바이스(CPU 또는 GPU)에 매핑
   - 지정된 디렉토리에서 커스텀 Mojo 연산 로드
   - `custom_extensions` 파라미터가 Mojo 구현과의 연결 핵심

2. **커스텀 연산 구성**:

   ```python
   output = ops.custom(
       name="softmax",
       values=[input_value],
       out_types=[
           TensorType(
               dtype=input_value.tensor.dtype,
               shape=input_value.tensor.shape,
               device=DeviceRef.from_device(device),
           )
       ],
       parameters={
           "target": "gpu" if device == Accelerator() else "cpu",
           "input_size": input_tensor.shape[0],
           "dtype": dtype,
       },
   )[0].tensor
   ```

   커스텀 연산을 다음과 같이 설정합니다:
   - Mojo 코드의 `@compiler.register("softmax")`와 일치하는 이름
   - 리스트로 전달되는 입력 값
   - 입력 shape과 타입에 맞는 출력 타입 정의
   - 대상 디바이스, 벡터 크기, 데이터 타입을 포함한 kernel 필수 파라미터
   - `[0].tensor`로 첫 번째 반환 요소에서 텐서 추출

3. **그래프 출력 정의**:

   ```python
   graph.output(output)
   ```

   연산의 결과를 그래프의 출력으로 등록합니다.

메인 스크립트는 다음과 같은 꼼꼼한 검증을 포함합니다:

- 랜덤 입력 데이터 생성: `np.random.randn(INPUT_SIZE).astype(np.float32)`
- SciPy로 기대 결과 계산: `scipy_softmax(input_array)`
- 수치 정확도 검증: `np.testing.assert_allclose(..., rtol=1e-5)`
- 출력이 유효한 확률 분포인지 확인: `np.sum(result.to_numpy())`

이 구현은 고성능 Mojo kernel과 Python의 과학 컴퓨팅 생태계를 통합하는 MAX Graph의 강력한 역량을 보여주며, 효율성과 사용 편의성을 동시에 제공합니다.
</div>

</details>
