<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# `warp.shuffle_xor()` Butterfly 통신

Warp 레벨 butterfly 통신에서는 `shuffle_xor()`을 사용하여 Warp 내에 정교한 트리 기반 통신 패턴을 구성할 수 있습니다. 이 강력한 기본 요소를 통해 공유 메모리나 명시적 동기화 없이 효율적인 병렬 reduction, 정렬 네트워크, 고급 조정 알고리즘을 구현할 수 있습니다.

**핵심 통찰:** _[shuffle_xor()](https://docs.modular.com/mojo/std/gpu/primitives/warp/shuffle_xor) 연산은 SIMT 실행을 활용하여 XOR 기반 통신 트리를 생성하며, Warp 크기에 대해 \\(O(\\log n)\\) 복잡도로 확장되는 효율적인 butterfly 네트워크와 병렬 알고리즘을 가능하게 합니다._

> **Butterfly 네트워크란?** [Butterfly 네트워크](https://en.wikipedia.org/wiki/Butterfly_network)는 스레드들이 인덱스의 XOR 패턴에 따라 데이터를 교환하는 통신 토폴로지입니다. 이름은 시각적으로 그렸을 때 나비 날개처럼 보이는 연결 패턴에서 유래했습니다. 이 네트워크는 \\(O(\\log n)\\) 통신 복잡도를 가능하게 하기 때문에 FFT, bitonic 정렬, 병렬 reduction 같은 병렬 알고리즘의 기반이 됩니다.

## 핵심 개념

이 퍼즐에서 배울 내용:

- `shuffle_xor()`을 활용한 **XOR 기반 통신 패턴**
- 병렬 알고리즘을 위한 **butterfly 네트워크 토폴로지**
- \\(O(\\log n)\\) 복잡도의 **트리 기반 병렬 reduction**
- 고급 조정을 위한 **조건부 butterfly 연산**
- 복잡한 공유 메모리를 대체하는 **하드웨어 최적화 병렬 기본 요소**

`shuffle_xor` 연산은 각 Lane이 [XOR](https://en.wikipedia.org/wiki/Exclusive_or) 패턴에 따라 다른 Lane과 데이터를 교환할 수 있게 합니다:
\\[\Large \text{shuffle\_xor}(\text{value}, \text{mask}) = \text{value_from_lane}(\text{lane\_id} \oplus \text{mask})\\]

이를 통해 복잡한 병렬 알고리즘이 우아한 butterfly 통신 패턴으로 변환되어, 명시적 조정 없이 효율적인 트리 reduction과 정렬 네트워크가 가능합니다.

## 1. 기본 butterfly 페어 교환

### 구성

- 벡터 크기: `SIZE = WARP_SIZE` (GPU에 따라 32 또는 64)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 블록 구성: `(WARP_SIZE, 1)` 블록당 스레드 수
- 데이터 타입: `DType.float32`
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)

### shuffle_xor 개념

기존 페어 교환 방식은 복잡한 인덱싱과 조정이 필요합니다:

```mojo
# 기존 방식 - 복잡하고 동기화가 필요
shared_memory[lane] = input[global_i]
barrier()
if lane % 2 == 0:
    partner = lane + 1
else:
    partner = lane - 1
if partner < WARP_SIZE:
    swapped_val = shared_memory[partner]
```

**기존 방식의 문제점:**

- **메모리 오버헤드**: 공유 메모리 할당이 필요
- **동기화**: 명시적 barrier가 필요
- **복잡한 로직**: 수동 파트너 계산과 경계 검사
- **낮은 확장성**: 하드웨어 통신을 활용하지 못함

`shuffle_xor()`을 사용하면 페어 교환이 우아해집니다:

```mojo
# Butterfly XOR 방식 - 간단하고 하드웨어 최적화
current_val = input[global_i]
swapped_val = shuffle_xor(current_val, 1)  # 1과 XOR하면 페어가 생성됨
output[global_i] = swapped_val
```

**shuffle_xor의 장점:**

- **메모리 오버헤드 제로**: 레지스터 간 직접 통신
- **동기화 불필요**: SIMT 실행이 정확성을 보장
- **하드웨어 최적화**: 모든 Lane에 대해 단일 명령으로 처리
- **Butterfly 기반**: 복잡한 병렬 알고리즘의 빌딩 블록

### 완성할 코드

`shuffle_xor()`을 사용하여 인접 페어 간 값을 교환하는 페어 교환을 구현합니다.

**수학적 연산:** XOR 패턴으로 인접 페어를 만들어 값을 교환합니다:
\\[\Large \\text{output}[i] = \\text{input}[i \oplus 1]\\]

입력 데이터 `[0, 1, 2, 3, 4, 5, 6, 7, ...]`을 페어 `[1, 0, 3, 2, 5, 4, 7, 6, ...]`으로 변환하며, 각 페어 `(i, i+1)`이 XOR 통신으로 값을 교환합니다.

```mojo
{{#include ../../../../../problems/p26/p26.mojo:butterfly_pair_swap_solution}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p26/p26.mojo" class="filename">전체 파일 보기: problems/p26/p26.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **shuffle_xor 이해하기**

`shuffle_xor(value, mask)` 연산은 각 Lane이 XOR 마스크만큼 차이나는 Lane과 데이터를 교환할 수 있게 합니다. 서로 다른 마스크 값으로 Lane ID를 XOR했을 때 어떤 일이 일어나는지 생각해 보세요.

**탐구할 핵심 질문:**

- Lane 0이 마스크 1로 XOR하면 어떤 파트너를 얻나요?
- Lane 1이 마스크 1로 XOR하면 어떤 파트너를 얻나요?
- 패턴이 보이나요?

**힌트**: 처음 몇 개의 Lane ID에 대해 XOR 연산을 직접 해보면 페어링 패턴을 이해할 수 있습니다.

### 2. **XOR 페어 패턴**

Lane ID의 이진 표현과 최하위 비트를 뒤집으면 어떻게 되는지 생각해 보세요.

**고려할 질문:**

- 짝수 Lane을 1과 XOR하면 어떻게 되나요?
- 홀수 Lane을 1과 XOR하면 어떻게 되나요?
- 왜 이것이 완벽한 페어를 만드나요?

### 3. **경계 검사 불필요**

`shuffle_down()`과 달리 `shuffle_xor()` 연산은 Warp 경계 내에서 유지됩니다. 작은 마스크로의 XOR이 절대로 범위 밖의 Lane ID를 만들지 않는 이유를 생각해 보세요.

**생각해 보세요**: 유효한 Lane ID를 1과 XOR했을 때 나올 수 있는 최대 Lane ID는 얼마인가요?

</div>
</details>

**Butterfly 페어 교환 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p26 --pair-swap
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p26 --pair-swap
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p26 --pair-swap
```

  </div>
  <div class="tab-content">

```bash
uv run poe p26 --pair-swap
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0, 13.0, 12.0, 15.0, 14.0, 17.0, 16.0, 19.0, 18.0, 21.0, 20.0, 23.0, 22.0, 25.0, 24.0, 27.0, 26.0, 29.0, 28.0, 31.0, 30.0]
expected: [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0, 13.0, 12.0, 15.0, 14.0, 17.0, 16.0, 19.0, 18.0, 21.0, 20.0, 23.0, 22.0, 25.0, 24.0, 27.0, 26.0, 29.0, 28.0, 31.0, 30.0]
✅ Butterfly pair swap test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p26/p26.mojo:butterfly_pair_swap_solution}}
```

<div class="solution-explanation">

이 풀이는 `shuffle_xor()`이 XOR 통신 패턴을 통해 완벽한 페어 교환을 어떻게 만드는지 보여줍니다.

**알고리즘 분석:**

```mojo
if global_i < size:
    current_val = input[global_i]              # 각 Lane이 자신의 요소를 읽음
    swapped_val = shuffle_xor(current_val, 1)  # XOR로 페어 교환 생성

    # 교환된 값을 저장
    output[global_i] = swapped_val
```

**SIMT 실행 상세 분석:**

```
사이클 1: 모든 Lane이 동시에 값을 로드
  Lane 0: current_val = input[0] = 0
  Lane 1: current_val = input[1] = 1
  Lane 2: current_val = input[2] = 2
  Lane 3: current_val = input[3] = 3
  ...
  Lane 31: current_val = input[31] = 31

사이클 2: shuffle_xor(current_val, 1)이 모든 Lane에서 실행
  Lane 0: Lane 1에서 수신 (0⊕1=1) → swapped_val = 1
  Lane 1: Lane 0에서 수신 (1⊕1=0) → swapped_val = 0
  Lane 2: Lane 3에서 수신 (2⊕1=3) → swapped_val = 3
  Lane 3: Lane 2에서 수신 (3⊕1=2) → swapped_val = 2
  ...
  Lane 30: Lane 31에서 수신 (30⊕1=31) → swapped_val = 31
  Lane 31: Lane 30에서 수신 (31⊕1=30) → swapped_val = 30

사이클 3: 결과 저장
  Lane 0: output[0] = 1
  Lane 1: output[1] = 0
  Lane 2: output[2] = 3
  Lane 3: output[3] = 2
  ...
```

**수학적 통찰:** XOR 속성을 활용한 완벽한 페어 교환을 구현합니다:
\\[\Large \\text{XOR}(i, 1) = \\begin{cases}
i + 1 & \\text{if } i \\bmod 2 = 0 \\\\
i - 1 & \\text{if } i \\bmod 2 = 1
\\end{cases}\\]

**shuffle_xor이 우월한 이유:**

1. **완벽한 대칭**: 모든 Lane이 정확히 하나의 페어에 참여
2. **조정 불필요**: 모든 페어가 동시에 교환
3. **하드웨어 최적화**: Warp 전체에 대해 단일 명령으로 처리
4. **Butterfly 기반**: 복잡한 병렬 알고리즘의 빌딩 블록

**성능 특성:**

- **Latency**: 1 사이클 (하드웨어 레지스터 교환)
- **대역폭**: 0 바이트 (메모리 트래픽 없음)
- **병렬성**: WARP_SIZE개 Lane 모두 동시에 교환
- **확장성**: 데이터 크기에 관계없이 \\(O(1)\\) 복잡도

</div>
</details>

## 2. Butterfly 병렬 최댓값

### 구성

- 벡터 크기: `SIZE = WARP_SIZE` (GPU에 따라 32 또는 64)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 블록 구성: `(WARP_SIZE, 1)` 블록당 스레드 수

### 완성할 코드

감소하는 offset으로 butterfly `shuffle_xor`을 사용하여 병렬 최댓값 reduction을 구현합니다.

**수학적 연산:** 트리 reduction을 통해 모든 Warp Lane에서 최댓값을 계산합니다:
\\[\Large \\text{max\_result} = \\max_{i=0}^{\\small\\text{WARP\_SIZE}-1} \\text{input}[i]\\]

**Butterfly reduction 패턴:** XOR offset을 `WARP_SIZE/2`에서 `1`까지 절반씩 줄여가며, 통신 범위가 단계마다 반으로 좁아지는 이진 트리를 구성합니다:

- **1단계**: `WARP_SIZE/2` 거리의 Lane과 비교 (Warp 전체를 포괄)
- **2단계**: `WARP_SIZE/4` 거리의 Lane과 비교 (범위를 절반으로 좁힘)
- **3단계**: `WARP_SIZE/8` 거리의 Lane과 비교
- **4단계**: `offset = 1`이 될 때까지 계속 절반으로 줄임

\\(\\log_2(\\text{WARP\_SIZE})\\) 단계를 거치면 모든 Lane이 전역 최댓값을 갖게 됩니다. 이 방식은 모든 `WARP_SIZE` (32, 64 등)에서 동작합니다.

```mojo
{{#include ../../../../../problems/p26/p26.mojo:butterfly_parallel_max}}
```

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **Butterfly reduction 이해하기**

Butterfly reduction은 이진 트리 통신 패턴을 생성합니다. 각 단계에서 문제 크기를 체계적으로 줄이는 방법을 생각해 보세요.

**핵심 질문:**

- 최대 범위를 커버하려면 시작 offset이 얼마여야 하나요?
- 단계 사이에 offset을 어떻게 변경해야 하나요?
- 언제 reduction을 멈춰야 하나요?

**힌트**: "butterfly"라는 이름은 통신 패턴에서 유래합니다 - 작은 예제에 대해 직접 그려보세요.

### 2. **XOR reduction 특성**

XOR은 각 단계에서 겹치지 않는 통신 페어를 생성합니다. 이것이 병렬 reduction에서 왜 중요한지 생각해 보세요.

**생각해 보세요:**

- 서로 다른 offset으로의 XOR이 어떻게 다른 통신 패턴을 만드나요?
- 같은 단계에서 Lane들이 왜 서로 간섭하지 않나요?
- XOR이 트리 reduction에 특히 적합한 이유는 무엇인가요?

### 3. **최댓값 누적**

각 Lane은 자신의 "영역"에서 최댓값의 지식을 점진적으로 쌓아가야 합니다.

**알고리즘 구조:**

- 자신의 값으로 시작
- 각 단계에서 이웃의 값과 비교
- 최댓값을 유지하고 계속 진행

**핵심 통찰**: 각 단계 후, "지식의 영역"이 두 배로 확장됩니다.

- 마지막 단계 후: 각 Lane이 전역 최댓값을 알게 됩니다

### 4. **이 패턴이 동작하는 이유**

Butterfly reduction은 \\(\\log_2(\\text{WARP\\_SIZE})\\) 단계 후에 다음을 보장합니다:

- **모든 Lane**이 **다른 모든 Lane의** 값을 간접적으로 확인
- **중복 통신 없음**: 각 페어가 단계당 정확히 한 번 교환
- **최적 복잡도**: \\(O(n)\\) 순차 비교 대신 \\(O(\\log n)\\) 단계

**추적 예제** (4개 Lane, 값 [3, 1, 7, 2]):

```
초기 상태: Lane 0=3, Lane 1=1, Lane 2=7, Lane 3=2

1단계 (offset=2): 0 ↔ 2, 1 ↔ 3
  Lane 0: max(3, 7) = 7
  Lane 1: max(1, 2) = 2
  Lane 2: max(7, 3) = 7
  Lane 3: max(2, 1) = 2

2단계 (offset=1): 0 ↔ 1, 2 ↔ 3
  Lane 0: max(7, 2) = 7
  Lane 1: max(2, 7) = 7
  Lane 2: max(7, 2) = 7
  Lane 3: max(2, 7) = 7

결과: 모든 Lane이 전역 최댓값 = 7을 가짐
```

</div>
</details>

**Butterfly 병렬 최댓값 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p26 --parallel-max
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p26 --parallel-max
```

  </div>
  <div class="tab-content">

```bash
uv run poe p26 --parallel-max
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
expected: [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
✅ Butterfly parallel max test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p26/p26.mojo:butterfly_parallel_max_solution}}
```

<div class="solution-explanation">

이 풀이는 `shuffle_xor()`이 \\(O(\\log n)\\) 복잡도의 효율적인 병렬 reduction 트리를 어떻게 생성하는지 보여줍니다.

**전체 알고리즘 분석:**

```mojo
if global_i < size:
    max_val = input[global_i]  # 로컬 값으로 시작

    # Butterfly reduction 트리: 모든 WARP_SIZE에 동적으로 대응
    offset = WARP_SIZE // 2
    while offset > 0:
        max_val = max(max_val, shuffle_xor(max_val, offset))
        offset //= 2

    output[global_i] = max_val  # 모든 Lane이 전역 최댓값을 가짐
```

**Butterfly 실행 추적 (8-Lane 예제, 값 [0,2,4,6,8,10,12,1000]):**

```
초기 상태:
  Lane 0: max_val = 0,    Lane 1: max_val = 2
  Lane 2: max_val = 4,    Lane 3: max_val = 6
  Lane 4: max_val = 8,    Lane 5: max_val = 10
  Lane 6: max_val = 12,   Lane 7: max_val = 1000

1단계: shuffle_xor(max_val, 4) - 절반 교환
  Lane 0↔4: max(0,8)=8,     Lane 1↔5: max(2,10)=10
  Lane 2↔6: max(4,12)=12,   Lane 3↔7: max(6,1000)=1000
  Lane 4↔0: max(8,0)=8,     Lane 5↔1: max(10,2)=10
  Lane 6↔2: max(12,4)=12,   Lane 7↔3: max(1000,6)=1000

2단계: shuffle_xor(max_val, 2) - 1/4 교환
  Lane 0↔2: max(8,12)=12,   Lane 1↔3: max(10,1000)=1000
  Lane 2↔0: max(12,8)=12,   Lane 3↔1: max(1000,10)=1000
  Lane 4↔6: max(8,12)=12,   Lane 5↔7: max(10,1000)=1000
  Lane 6↔4: max(12,8)=12,   Lane 7↔5: max(1000,10)=1000

3단계: shuffle_xor(max_val, 1) - 페어 교환
  Lane 0↔1: max(12,1000)=1000,  Lane 1↔0: max(1000,12)=1000
  Lane 2↔3: max(12,1000)=1000,  Lane 3↔2: max(1000,12)=1000
  Lane 4↔5: max(12,1000)=1000,  Lane 5↔4: max(1000,12)=1000
  Lane 6↔7: max(12,1000)=1000,  Lane 7↔6: max(1000,12)=1000

최종 결과: 모든 Lane의 max_val = 1000
```

**수학적 통찰:** Butterfly 통신으로 병렬 reduction 연산자를 구현합니다:
\\[\Large \\text{Reduce}(\\oplus, [a_0, a_1, \\ldots, a_{n-1}]) = a_0 \\oplus a_1 \\oplus \\cdots \\oplus a_{n-1}\\]

여기서 \\(\\oplus\\)는 `max` 연산이며, butterfly 패턴이 최적 \\(O(\\log n)\\) 복잡도를 보장합니다.

**Butterfly reduction이 우월한 이유:**

1. **로그 복잡도**: 순차 reduction의 \\(O(n)\\)에 비해 \\(O(\\log n)\\)
2. **완벽한 부하 분산**: 모든 Lane이 각 단계에서 동등하게 참여
3. **메모리 병목 없음**: 순수 레지스터 간 통신
4. **하드웨어 최적화**: GPU butterfly 네트워크에 직접 매핑

**성능 특성:**

- **단계 수**: \\(\\log_2(\\text{WARP\_SIZE})\\) (예: 32-스레드 Warp는 5단계, 64-스레드 Warp는 6단계)
- **단계당 latency**: 1 사이클 (레지스터 교환 + 비교)
- **총 latency**: 순차 방식의 \\((\\text{WARP\_SIZE}-1)\\) 사이클 대비 \\(\\log_2(\\text{WARP\_SIZE})\\) 사이클
- **병렬성**: 알고리즘 전체에서 모든 Lane이 활성 상태

</div>
</details>

## 3. Butterfly 조건부 최댓값

### 구성

- 벡터 크기: `SIZE_2 = 64` (멀티 블록 시나리오)
- 그리드 구성: `BLOCKS_PER_GRID_2 = (2, 1)` 그리드당 블록 수
- 블록 구성: `THREADS_PER_BLOCK_2 = (WARP_SIZE, 1)` 블록당 스레드 수

### 완성할 코드

짝수 Lane은 최댓값을, 홀수 Lane은 최솟값을 저장하는 조건부 butterfly reduction을 구현합니다.

**수학적 연산:** 최댓값과 최솟값 모두에 대해 butterfly reduction을 수행한 후, Lane 홀짝에 따라 조건부로 출력합니다:
\\[\Large \\text{output}[i] = \\begin{cases}
\\max_{j=0}^{\\text{WARP\_SIZE}-1} \\text{input}[j] & \\text{if } i \\bmod 2 = 0 \\\\
\\min_{j=0}^{\\text{WARP\_SIZE}-1} \\text{input}[j] & \\text{if } i \\bmod 2 = 1
\\end{cases}\\]

**이중 reduction 패턴:** Butterfly 트리를 통해 최댓값과 최솟값을 동시에 추적한 후, Lane ID 홀짝에 따라 조건부로 출력합니다. 이는 butterfly 패턴이 복잡한 다중 값 reduction으로 어떻게 확장되는지를 보여줍니다.

```mojo
{{#include ../../../../../problems/p26/p26.mojo:butterfly_conditional_max}}
```

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **이중 추적 butterfly reduction**

이 퍼즐은 butterfly 트리를 통해 두 가지 다른 값을 동시에 추적해야 합니다. 여러 reduction을 병렬로 실행하는 방법을 생각해 보세요.

**핵심 질문:**

- Reduction 과정에서 최댓값과 최솟값을 어떻게 동시에 유지할 수 있나요?
- 두 연산에 같은 butterfly 패턴을 사용할 수 있나요?
- 어떤 변수를 추적해야 하나요?

### 2. **조건부 출력 로직**

Butterfly reduction을 완료한 후, Lane 홀짝에 따라 다른 값을 출력해야 합니다.

**고려할 점:**

- Lane이 짝수인지 홀수인지 어떻게 판별하나요?
- 어떤 Lane이 최댓값을, 어떤 Lane이 최솟값을 출력해야 하나요?
- Lane ID에 어떻게 접근하나요?

### 3. **min과 max 동시 butterfly reduction**

이 과제의 핵심은 같은 butterfly 통신 패턴으로 min과 max를 효율적으로 병렬 계산하는 것입니다.

**생각해 보세요:**

- min과 max에 별도의 shuffle 연산이 필요한가요?
- 두 연산에 같은 이웃 값을 재사용할 수 있나요?
- 두 reduction 모두 올바르게 완료되려면 어떻게 해야 하나요?

### 4. **멀티 블록 경계 고려사항**

이 퍼즐은 여러 블록을 사용합니다. 이것이 reduction 범위에 어떤 영향을 미치는지 생각해 보세요.

**중요한 고려사항:**

- 각 butterfly reduction의 범위는 어디까지인가요?
- 블록 구조가 Lane 번호 매기기에 어떤 영향을 미치나요?
- 전역 min/max를 계산하나요, 블록별 min/max를 계산하나요?

</div>
</details>

**Butterfly 조건부 최댓값 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p26 --conditional-max
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p26 --conditional-max
```

  </div>
  <div class="tab-content">

```bash
uv run poe p26 --conditional-max
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE_2:  64
output: [9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0]
expected: [9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0, 63.0, 32.0]
✅ Butterfly conditional max test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p26/p26.mojo:butterfly_conditional_max_solution}}
```

<div class="solution-explanation">

이 풀이는 이중 추적과 조건부 출력을 사용하는 고급 butterfly reduction을 보여줍니다.

**전체 알고리즘 분석:**

```mojo
if global_i < size:
    current_val = input[global_i]
    min_val = current_val  # 최솟값을 별도로 추적

    # max와 min 동시 butterfly reduction (log_2(WARP_SIZE) 단계)
    offset = WARP_SIZE // 2
    while offset > 0:
        neighbor_val = shuffle_xor(current_val, offset)
        current_val = max(current_val, neighbor_val)    # Max reduction

        min_neighbor_val = shuffle_xor(min_val, offset)
        min_val = min(min_val, min_neighbor_val)        # Min reduction

        offset //= 2

    # Lane 홀짝에 따른 조건부 출력
    if lane % 2 == 0:
        output[global_i] = current_val  # 짝수 Lane: 최댓값
    else:
        output[global_i] = min_val      # 홀수 Lane: 최솟값
```

**이중 reduction 실행 추적 (4-Lane 예제, 값 [3, 1, 7, 2]):**

```
초기 상태:
  Lane 0: current_val=3, min_val=3
  Lane 1: current_val=1, min_val=1
  Lane 2: current_val=7, min_val=7
  Lane 3: current_val=2, min_val=2

1단계: shuffle_xor(current_val, 2)와 shuffle_xor(min_val, 2) - 절반 교환
  Lane 0↔2: max_neighbor=7, min_neighbor=7 → current_val=max(3,7)=7, min_val=min(3,7)=3
  Lane 1↔3: max_neighbor=2, min_neighbor=2 → current_val=max(1,2)=2, min_val=min(1,2)=1
  Lane 2↔0: max_neighbor=3, min_neighbor=3 → current_val=max(7,3)=7, min_val=min(7,3)=3
  Lane 3↔1: max_neighbor=1, min_neighbor=1 → current_val=max(2,1)=2, min_val=min(2,1)=1

2단계: shuffle_xor(current_val, 1)와 shuffle_xor(min_val, 1) - 페어 교환
  Lane 0↔1: max_neighbor=2, min_neighbor=1 → current_val=max(7,2)=7, min_val=min(3,1)=1
  Lane 1↔0: max_neighbor=7, min_neighbor=3 → current_val=max(2,7)=7, min_val=min(1,3)=1
  Lane 2↔3: max_neighbor=2, min_neighbor=1 → current_val=max(7,2)=7, min_val=min(3,1)=1
  Lane 3↔2: max_neighbor=7, min_neighbor=3 → current_val=max(2,7)=7, min_val=min(1,3)=1

최종 결과: 모든 Lane이 current_val=7 (전역 max)과 min_val=1 (전역 min)을 가짐
```

**동적 알고리즘** (모든 WARP_SIZE에서 동작):

```mojo
offset = WARP_SIZE // 2
while offset > 0:
    neighbor_val = shuffle_xor(current_val, offset)
    current_val = max(current_val, neighbor_val)

    min_neighbor_val = shuffle_xor(min_val, offset)
    min_val = min(min_val, min_neighbor_val)

    offset //= 2
```

**수학적 통찰:** 조건부 디멀티플렉싱을 사용하는 이중 병렬 reduction을 구현합니다:
\\[\Large \\begin{align}
\\text{max\_result} &= \\max_{i=0}^{n-1} \\text{input}[i] \\\\
\\text{min\_result} &= \\min_{i=0}^{n-1} \\text{input}[i] \\\\
\\text{output}[i] &= \\text{lane\_parity}(i) \\; \text{?} \\; \\text{min\_result} : \\text{max\_result}
\\end{align}\\]

**이중 butterfly reduction이 동작하는 이유:**

1. **독립적 reduction**: Max와 min reduction은 수학적으로 독립
2. **병렬 실행**: 둘 다 같은 butterfly 통신 패턴을 사용 가능
3. **통신 공유**: 같은 shuffle 연산이 두 reduction 모두에 활용
4. **조건부 출력**: Lane 홀짝이 어떤 결과를 출력할지 결정

**성능 특성:**

- **통신 단계**: \\(\\log_2(\\text{WARP\_SIZE})\\) (단일 reduction과 동일)
- **단계당 연산**: 단일 reduction의 1개 대비 2개 연산 (max + min)
- **메모리 효율성**: 복잡한 공유 메모리 방식 대비 스레드당 레지스터 2개
- **출력 유연성**: 서로 다른 Lane이 다른 reduction 결과를 출력 가능

</div>
</details>

## 요약

`shuffle_xor()` 기본 요소는 효율적인 병렬 알고리즘의 기반이 되는 강력한 butterfly 통신 패턴을 가능하게 합니다. 세 가지 문제를 통해 다음을 배웠습니다:

### **핵심 Butterfly 패턴**

1. **페어 교환** (`shuffle_xor(value, 1)`):
   - 완벽한 인접 페어 생성: (0,1), (2,3), (4,5), ...
   - 메모리 오버헤드 제로의 \\(O(1)\\) 복잡도
   - 정렬 네트워크와 데이터 재배치의 기반

2. **트리 reduction** (동적 offset: `WARP_SIZE/2` → `1`):
   - 로그 병렬 reduction: 순차의 \\(O(n)\\) 대비 \\(O(\\log n)\\)
   - 모든 결합 연산에 적용 가능 (max, min, sum 등)
   - 모든 Warp Lane에 걸쳐 최적의 부하 분산

3. **조건부 다중 reduction** (이중 추적 + Lane 홀짝):
   - 여러 reduction을 동시에 병렬 수행
   - 스레드 특성에 따른 조건부 출력
   - 명시적 동기화 없는 고급 조정

### **핵심 알고리즘 통찰**

**XOR 통신 특성:**

- `shuffle_xor(value, mask)`가 대칭적이고 겹치지 않는 페어를 생성
- 각 마스크가 고유한 통신 토폴로지를 생성
- 이진 XOR 패턴에서 butterfly 네트워크가 자연스럽게 도출

**동적 알고리즘 설계:**

```mojo
offset = WARP_SIZE // 2
while offset > 0:
    neighbor_val = shuffle_xor(current_val, offset)
    current_val = operation(current_val, neighbor_val)
    offset //= 2
```

**성능 이점:**

- **하드웨어 최적화**: 레지스터 간 직접 통신
- **동기화 불필요**: SIMT 실행이 정확성을 보장
- **확장 가능한 복잡도**: 모든 WARP_SIZE (32, 64 등)에서 \\(O(\\log n)\\)
- **메모리 효율성**: 공유 메모리 불필요

### **실용적 활용**

이 butterfly 패턴들의 기반이 되는 분야:

- **병렬 reduction**: 합계, max, min, 논리 연산
- **Prefix/scan 연산**: 누적 합, 병렬 정렬
- **FFT 알고리즘**: 신호 처리와 convolution
- **Bitonic 정렬**: 병렬 정렬 네트워크
- **그래프 알고리즘**: 트리 순회와 연결성

`shuffle_xor()` 기본 요소는 복잡한 병렬 조정을 우아하고 하드웨어 최적화된 통신 패턴으로 변환하며, 다양한 GPU 아키텍처에서 효율적으로 확장됩니다.
