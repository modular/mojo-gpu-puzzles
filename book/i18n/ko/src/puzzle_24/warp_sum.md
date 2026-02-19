<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# warp.sum()의 핵심 - Warp 레벨 내적

[Puzzle 12](../puzzle_12/puzzle_12.md)에서 살펴본 내적을 Mojo의 Warp 연산으로 구현합니다. 복잡한 공유 메모리 패턴을 간단한 함수 호출로 대체합니다. 각 Warp Lane이 하나의 요소를 처리하고 `warp.sum()`으로 결과를 자동으로 합산하여, Warp 프로그래밍이 GPU 동기화를 어떻게 변환하는지 보여줍니다.

**핵심 통찰:** _[warp.sum()](https://docs.modular.com/mojo/std/gpu/primitives/warp/sum) 연산은 SIMT 실행을 활용하여 공유 메모리 + barrier + 트리 reduction을 단일 하드웨어 가속 명령으로 대체합니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- `warp.sum()`을 활용한 **Warp 레벨 reduction**
- **SIMT 실행 모델**과 Lane 동기화
- `WARP_SIZE`를 활용한 **크로스 아키텍처 호환성**
- 복잡한 패턴에서 간단한 패턴으로의 **성능 변환**
- **Lane ID 관리**와 조건부 쓰기

수학적 연산은 내적(dot product)입니다:
\\[\Large \text{output}[0] = \sum_{i=0}^{N-1} a[i] \times b[i]\\]

하지만 구현 과정에서 Mojo의 모든 Warp 레벨 GPU 프로그래밍에 적용되는 기본 패턴을 배웁니다.

## 구성

- 벡터 크기: `SIZE = WARP_SIZE` (GPU 아키텍처에 따라 32 또는 64)
- 데이터 타입: `DType.float32`
- 블록 구성: `(WARP_SIZE, 1)` 블록당 스레드 수
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)

## 기존 방식의 복잡성 (Puzzle 12에서)

[solutions/p12/p12.mojo](../../../../../solutions/p12/p12.mojo)의 복잡한 방식을 떠올려 봅시다. 공유 메모리, barrier, 트리 reduction이 필요했습니다:

```mojo
{{#include ../../../../../problems/p24/p24.mojo:traditional_approach_from_p12}}
```

**이 방식이 복잡한 이유:**

- **공유 메모리 할당**: 블록 내에서 수동으로 메모리를 관리
- **명시적 barrier**: 스레드 동기화를 위한 `barrier()` 호출
- **트리 reduction**: stride 기반 인덱싱을 사용하는 복잡한 루프
- **조건부 쓰기**: 스레드 0만 최종 결과를 기록

동작은 하지만, 코드가 장황하고 오류가 발생하기 쉬우며 GPU 동기화에 대한 깊은 이해가 필요합니다.

**기존 방식 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p24 --traditional
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p24 --traditional
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p24 --traditional
```

  </div>
  <div class="tab-content">

```bash
uv run poe p24 --traditional
```

  </div>
</div>

## 완성할 코드

### 1. 간단한 Warp 커널 방식

복잡한 기존 방식을 `warp_sum()`을 사용하는 간단한 Warp 커널로 변환합니다:

```mojo
{{#include ../../../../../problems/p24/p24.mojo:simple_warp_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p24/p24.mojo" class="filename">전체 파일 보기: problems/p24/p24.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **간단한 Warp 커널 구조 이해하기**

`simple_warp_dot_product` 함수를 **6줄 이내**로 완성해야 합니다:

```mojo
fn simple_warp_dot_product[...](output, a, b):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # 여기를 채우세요 (최대 6줄)
```

**따라야 할 패턴:**

1. 이 스레드의 요소에 대한 부분곱 계산
2. `warp_sum()`으로 모든 Warp Lane의 값을 합산
3. Lane 0이 최종 결과를 기록

### 2. **부분곱 계산하기**

```mojo
var partial_product: Scalar[dtype] = 0
if global_i < size:
    partial_product = (a[global_i] * b[global_i]).reduce_add()
```

**`.reduce_add()`가 필요한 이유:** Mojo의 값은 SIMD 기반이므로 `a[global_i] * b[global_i]`는 SIMD 벡터를 반환합니다. `.reduce_add()`로 벡터를 스칼라 값으로 합산합니다.

**경계 검사:** 모든 스레드가 유효한 데이터를 가지고 있지 않을 수 있으므로 필수적입니다.

### 3. **Warp reduction의 마법**

```mojo
total = warp_sum(partial_product)
```

**`warp_sum()`이 하는 일:**

- 각 Lane의 `partial_product` 값을 가져옴
- Warp 내 모든 Lane의 값을 합산 (하드웨어 가속)
- **모든 Lane**에 같은 합계를 반환 (Lane 0만이 아님)
- **명시적 동기화가 전혀 필요 없음** (SIMT가 처리)

### 4. **결과 기록하기**

```mojo
if lane_id() == 0:
    output[global_i // WARP_SIZE] = total
```

**왜 Lane 0만?** `warp_sum()` 이후 모든 Lane이 같은 `total` 값을 갖지만, 경쟁 상태를 피하기 위해 한 번만 기록합니다.

**왜 `output[0]`에 직접 쓰지 않을까?** 유연성을 위해서입니다. 이 함수는 Warp가 여러 개인 경우에도 사용할 수 있으며, 각 Warp의 결과가 `global_i // WARP_SIZE` 위치에 기록됩니다.

**`lane_id()`:** 0-31 (NVIDIA) 또는 0-63 (AMD)을 반환 - Warp 내에서 어느 Lane인지 식별합니다.

</div>
</details>

**간단한 Warp 커널 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p24 --kernel
```

  </div>
  <div class="tab-content">

```bash
pixi run p24 --kernel
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
SIZE: 32
WARP_SIZE: 32
SIMD_WIDTH: 8
=== RESULT ===
out: 10416.0
expected: 10416.0
🚀 Notice how simple the warp version is compared to p12.mojo!
   Same kernel structure, but warp_sum() replaces all the complexity!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p24/p24.mojo:simple_warp_kernel_solution}}
```

<div class="solution-explanation">

간단한 Warp 커널은 복잡한 동기화에서 하드웨어 가속 기본 요소로의 근본적인 변환을 보여줍니다:

**기존 방식에서 사라진 것들:**

- **15줄 이상 → 6줄**: 획기적인 코드 축소
- **공유 메모리 할당**: 메모리 관리 불필요
- **3회 이상의 barrier() 호출**: 명시적 동기화 제로
- **복잡한 트리 reduction**: 단일 함수 호출로 대체
- **Stride 기반 인덱싱**: 완전히 제거

**SIMT 실행 모델:**

```
Warp Lane (SIMT 실행):
Lane 0: partial_product = a[0] * b[0]    = 0.0
Lane 1: partial_product = a[1] * b[1]    = 4.0
Lane 2: partial_product = a[2] * b[2]    = 16.0
...
Lane 31: partial_product = a[31] * b[31] = 3844.0

warp_sum() 하드웨어 연산:
모든 Lane → 0.0 + 4.0 + 16.0 + ... + 3844.0 = 10416.0
모든 Lane이 수신 → total = 10416.0 (broadcast 결과)
```

**Barrier 없이 동작하는 이유:**

1. **SIMT 실행**: 모든 Lane이 각 명령 동시 실행
2. **하드웨어 동기화**: `warp_sum()`이 시작될 때 모든 Lane이 이미 `partial_product` 계산 완료
3. **내장 통신**: GPU 하드웨어가 reduction 연산 처리
4. **Broadcast 결과**: 모든 Lane이 같은 `total` 값 수신

</div>
</details>

### 2. 함수형 방식

이번에는 Mojo의 함수형 프로그래밍 패턴을 사용하여 같은 Warp 내적을 구현합니다:

```mojo
{{#include ../../../../../problems/p24/p24.mojo:functional_warp_approach}}
```

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **함수형 방식의 구조 이해하기**

`compute_dot_product` 함수를 **10줄 이내**로 완성해야 합니다:

```mojo
@parameter
@always_inline
fn compute_dot_product[simd_width: Int, rank: Int](indices: IndexList[rank]) capturing -> None:
    idx = indices[0]
    # 여기를 채우세요 (최대 10줄)
```

**함수형 패턴의 차이점:**

- `elementwise`를 사용하여 정확히 `WARP_SIZE`개의 스레드 실행
- 각 스레드가 `idx`를 기반으로 하나의 요소 처리
- 같은 Warp 연산, 다른 실행 메커니즘

### 2. **부분곱 계산하기**

```mojo
var partial_product: Scalar[dtype] = 0.0
if idx < size:
    a_val = a.load[1](idx, 0)
    b_val = b.load[1](idx, 0)
    partial_product = (a_val * b_val).reduce_add()
else:
    partial_product = 0.0
```

**로딩 패턴:** `a.load[1](idx, 0)`은 위치 `idx`에서 정확히 1개 요소를 로드합니다 (SIMD 벡터화 없음).

**경계 처리:** 범위를 벗어난 스레드의 `partial_product`를 `0.0`으로 설정하여 합산에 기여하지 않도록 합니다.

### 3. **Warp 연산과 저장**

```mojo
total = warp_sum(partial_product)

if lane_id() == 0:
    output.store[1](Index(idx // WARP_SIZE), total)
```

**저장 패턴:** `output.store[1](Index(idx // WARP_SIZE), 0, total)`은 출력 텐서의 위치 `(idx // WARP_SIZE, 0)`에 1개 요소를 저장합니다.

**동일한 Warp 로직:** `warp_sum()`과 Lane 0의 기록 로직은 함수형 방식에서도 동일하게 동작합니다.

### 4. **import에서 사용 가능한 함수들**

```mojo
from gpu import lane_id
from gpu.primitives.warp import sum as warp_sum, WARP_SIZE

# 함수 내에서:
my_lane = lane_id()           # 0 ~ WARP_SIZE-1
total = warp_sum(my_value)    # 하드웨어 가속 reduction
warp_size = WARP_SIZE         # 32 (NVIDIA) 또는 64 (AMD)
```

</div>
</details>

**함수형 방식 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p24 --functional
```

  </div>
  <div class="tab-content">

```bash
pixi run p24 --functional
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
SIZE: 32
WARP_SIZE: 32
SIMD_WIDTH: 8
=== RESULT ===
out: 10416.0
expected: 10416.0
🔧 Functional approach shows modern Mojo style with warp operations!
   Clean, composable, and still leverages warp hardware primitives!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p24/p24.mojo:functional_warp_approach_solution}}
```

<div class="solution-explanation">

함수형 Warp 방식은 Warp 연산을 활용한 현대적인 Mojo 프로그래밍 패턴을 보여줍니다:

**함수형 방식의 특징:**

```mojo
elementwise[compute_dot_product, 1, target="gpu"](size, ctx)
```

**장점:**

- **타입 안전성**: 컴파일 타임 텐서 레이아웃 검사
- **조합 가능성**: 다른 함수형 연산과 쉽게 통합
- **현대적 패턴**: Mojo의 함수형 프로그래밍 기능 활용
- **자동 최적화**: 컴파일러가 고수준 최적화를 적용 가능

**커널 방식과의 주요 차이:**

- **실행 메커니즘**: `enqueue_function` 대신 `elementwise` 사용
- **메모리 접근**: `.load[1]()`과 `.store[1]()` 패턴 사용
- **통합성**: 다른 함수형 연산과 자연스럽게 결합

**동일한 Warp의 이점:**

- **동기화 제로**: `warp_sum()`이 동일하게 동작
- **하드웨어 가속**: 커널 방식과 같은 성능
- **크로스 아키텍처**: `WARP_SIZE`가 자동으로 적응

</div>
</details>

## 벤치마크를 통한 성능 비교

종합 벤치마크를 실행하여 Warp 연산의 확장성을 확인합니다:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p24 --benchmark
```

  </div>
  <div class="tab-content">

```bash
pixi run p24 --benchmark
```

  </div>
</div>

전체 벤치마크 실행 결과의 예시입니다:

```
SIZE: 32
WARP_SIZE: 32
SIMD_WIDTH: 8
--------------------------------------------------------------------------------
Testing SIZE=1 x WARP_SIZE, BLOCKS=1
Running traditional_1x
Running simple_warp_1x
Running functional_warp_1x
--------------------------------------------------------------------------------
Testing SIZE=4 x WARP_SIZE, BLOCKS=4
Running traditional_4x
Running simple_warp_4x
Running functional_warp_4x
--------------------------------------------------------------------------------
Testing SIZE=32 x WARP_SIZE, BLOCKS=32
Running traditional_32x
Running simple_warp_32x
Running functional_warp_32x
--------------------------------------------------------------------------------
Testing SIZE=256 x WARP_SIZE, BLOCKS=256
Running traditional_256x
Running simple_warp_256x
Running functional_warp_256x
--------------------------------------------------------------------------------
Testing SIZE=2048 x WARP_SIZE, BLOCKS=2048
Running traditional_2048x
Running simple_warp_2048x
Running functional_warp_2048x
--------------------------------------------------------------------------------
Testing SIZE=16384 x WARP_SIZE, BLOCKS=16384 (Large Scale)
Running traditional_16384x
Running simple_warp_16384x
Running functional_warp_16384x
--------------------------------------------------------------------------------
Testing SIZE=65536 x WARP_SIZE, BLOCKS=65536 (Massive Scale)
Running traditional_65536x
Running simple_warp_65536x
Running functional_warp_65536x
| name                   | met (ms)              | iters |
| ---------------------- | --------------------- | ----- |
| traditional_1x         | 0.00460128            | 100   |
| simple_warp_1x         | 0.00574047            | 100   |
| functional_warp_1x     | 0.00484192            | 100   |
| traditional_4x         | 0.00492671            | 100   |
| simple_warp_4x         | 0.00485247            | 100   |
| functional_warp_4x     | 0.00587679            | 100   |
| traditional_32x        | 0.0062406399999999996 | 100   |
| simple_warp_32x        | 0.0054918400000000004 | 100   |
| functional_warp_32x    | 0.00552447            | 100   |
| traditional_256x       | 0.0050614300000000004 | 100   |
| simple_warp_256x       | 0.00488768            | 100   |
| functional_warp_256x   | 0.00461472            | 100   |
| traditional_2048x      | 0.01120031            | 100   |
| simple_warp_2048x      | 0.00884383            | 100   |
| functional_warp_2048x  | 0.007038720000000001  | 100   |
| traditional_16384x     | 0.038533750000000005  | 100   |
| simple_warp_16384x     | 0.0323264             | 100   |
| functional_warp_16384x | 0.01674271            | 100   |
| traditional_65536x     | 0.19784991999999998   | 100   |
| simple_warp_65536x     | 0.12870176            | 100   |
| functional_warp_65536x | 0.048680310000000004  | 100   |

Benchmarks completed!

WARP OPERATIONS PERFORMANCE ANALYSIS:
   GPU Architecture: NVIDIA (WARP_SIZE=32) vs AMD (WARP_SIZE=64)
   - 1,...,256 x WARP_SIZE: Grid size too small to benchmark
   - 2048 x WARP_SIZE: Warp primative benefits emerge
   - 16384 x WARP_SIZE: Large scale (512K-1M elements)
   - 65536 x WARP_SIZE: Massive scale (2M-4M elements)

   Expected Results at Large Scales:
   • Traditional: Slower due to more barrier overhead
   • Warp operations: Faster, scale better with problem size
   • Memory bandwidth becomes the limiting factor
```

**이 예시에서 얻을 수 있는 성능 인사이트:**

- **소규모 (1x-4x)**: Warp 연산이 소폭의 개선을 보임 (~10-15% 빠름)
- **중규모 (32x-256x)**: 함수형 방식이 가장 좋은 성능을 보이는 경우가 많음
- **대규모 (16K-65K)**: 메모리 대역폭이 지배적이 되면서 모든 방식의 성능이 수렴
- **변동성**: 성능은 특정 GPU 아키텍처와 메모리 서브시스템에 크게 의존

**참고:** 하드웨어(GPU 모델, 메모리 대역폭, `WARP_SIZE`)에 따라 결과가 크게 달라집니다. 핵심은 절대적인 수치보다 상대적인 성능 추세를 관찰하는 것입니다.

## 다음 단계

Warp sum 연산을 배웠으니, 다음으로 진행할 수 있습니다:

- **[언제 Warp 프로그래밍을 사용할까](./warp_extra.md)**: Warp vs 기존 방식에 대한 전략적 의사결정 프레임워크
- **고급 Warp 연산**: 복잡한 통신 패턴을 위한 `shuffle_idx()`, `shuffle_down()`, `prefix_sum()`
- **멀티 Warp 알고리즘**: Warp 연산과 블록 레벨 동기화의 결합
- **메모리 Coalescing 최적화**: 최대 대역폭을 위한 메모리 접근 패턴 최적화

💡 **핵심 요점**: Warp 연산은 복잡한 동기화 패턴을 하드웨어 가속 기본 요소로 대체하여 GPU 프로그래밍을 변환합니다. 실행 모델을 이해하면 성능을 희생하지 않고도 획기적인 단순화가 가능합니다.
