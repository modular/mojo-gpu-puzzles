<!-- i18n-source-commit: 1bae134f191aad7be60cc8612dcb64a50ef5ab2e -->

# `warp.shuffle_down()` 일대일 통신

Warp 레벨 이웃 통신에서는 `shuffle_down()`을 사용하여 Warp 내 인접 Lane의 데이터에 접근할 수 있습니다. 이 강력한 기본 요소를 통해 공유 메모리나 명시적 동기화 없이 유한 차분, 이동 평균, 이웃 기반 계산을 효율적으로 수행할 수 있습니다.

**핵심 통찰:** _[shuffle_down()](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_down) 연산은 SIMT 실행을 활용하여 각 Lane이 같은 Warp 내 이웃의 데이터에 접근할 수 있게 하며, 효율적인 stencil 패턴과 슬라이딩 윈도우 연산을 가능하게 합니다._

> **Stencil 연산이란?** [Stencil](https://en.wikipedia.org/wiki/Iterative_Stencil_Loops) 연산은 각 출력 요소가 이웃 입력 요소의 고정된 패턴에 의존하는 계산입니다. 대표적인 예로 유한 차분(도함수), convolution, 이동 평균이 있습니다. "Stencil"은 이웃 접근 패턴을 가리킵니다 - 예를 들어 `[i-1, i, i+1]`을 읽는 3점 stencil이나 `[i-2, i-1, i, i+1, i+2]`를 읽는 5점 stencil이 있습니다.

## 핵심 개념

이 퍼즐에서 배울 내용:

- `shuffle_down()`을 활용한 **Warp 레벨 데이터 shuffle**
- Stencil 계산을 위한 **이웃 접근 패턴**
- Warp 가장자리에서의 **경계 처리**
- 확장된 이웃 접근을 위한 **다중 offset shuffle**
- 멀티 블록 시나리오에서의 **Warp 간 조정**

`shuffle_down` 연산은 각 Lane이 더 높은 인덱스의 Lane에서 데이터를 가져올 수 있게 합니다:
\\[\Large \text{shuffle\_down}(\text{value}, \text{offset}) = \text{value_from_lane}(\text{lane\_id} + \text{offset})\\]

이를 통해 복잡한 이웃 접근 패턴이 간단한 Warp 레벨 연산으로 변환되어, 명시적 메모리 인덱싱 없이 효율적인 stencil 계산이 가능합니다.

## 1. 기본 이웃 차분

### 구성

- 벡터 크기: `SIZE = WARP_SIZE` (GPU에 따라 32 또는 64)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 블록 구성: `(WARP_SIZE, 1)` 블록당 스레드 수
- 데이터 타입: `DType.float32`
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)

### shuffle_down 개념

기존 이웃 접근 방식은 복잡한 인덱싱과 경계 검사가 필요합니다:

```mojo
# 기존 방식 - 복잡하고 오류가 발생하기 쉬움
if global_i < size - 1:
    next_value = input[global_i + 1]  # 범위 초과 가능성
    result = next_value - current_value
```

**기존 방식의 문제점:**

- **경계 검사**: 배열 경계를 수동으로 확인해야 함
- **메모리 접근**: 별도의 메모리 로드가 필요
- **동기화**: 공유 메모리 패턴에서 barrier가 필요할 수 있음
- **복잡한 로직**: 경계의 예외 상황 처리가 장황해짐

`shuffle_down()`을 사용하면 이웃 접근이 간결해집니다:

```mojo
# Warp shuffle 방식 - 간단하고 안전
current_val = input[global_i]
next_val = shuffle_down(current_val, 1)  # lane+1에서 값 가져오기
if lane < WARP_SIZE - 1:
    result = next_val - current_val
```

**shuffle_down의 장점:**

- **메모리 오버헤드 제로**: 추가 메모리 접근 불필요
- **자동 경계 처리**: 하드웨어가 Warp 경계를 관리
- **동기화 불필요**: SIMT 실행이 정확성을 보장
- **조합 가능**: 다른 Warp 연산과 쉽게 결합

### 작성할 코드

`shuffle_down()`으로 다음 요소에 접근하여 유한 차분을 구현합니다.

**수학적 연산:** 각 요소의 이산 도함수(유한 차분)를 계산합니다:
\\[\Large \\text{output}[i] = \\text{input}[i+1] - \\text{input}[i]\\]

입력 데이터 `[0, 1, 4, 9, 16, 25, ...]` (제곱수: `i * i`)를 차분값 `[1, 3, 5, 7, 9, ...]` (홀수)로 변환하여, 이차 함수의 이산 도함수를 효과적으로 계산합니다.

```mojo
{{#include ../../../../../problems/p25/p25.mojo:neighbor_difference}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p25/p25.mojo" class="filename">전체 파일 보기: problems/p25/p25.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **shuffle_down 이해하기**

`shuffle_down(value, offset)` 연산은 각 Lane이 더 높은 인덱스의 Lane에서 데이터를 받을 수 있게 합니다. 명시적 메모리 로드 없이 이웃 요소에 접근하는 방법을 살펴보세요.

**`shuffle_down(val, 1)`이 하는 일:**

- Lane 0이 Lane 1의 값을 받음
- Lane 1이 Lane 2의 값을 받음
- ...
- Lane 30이 Lane 31의 값을 받음
- Lane 31은 미정의 값을 받음 (경계 검사로 처리)

### 2. **Warp 경계 고려사항**

Warp의 가장자리에서 어떤 일이 일어나는지 생각해 보세요. 일부 Lane은 shuffle 연산으로 접근할 유효한 이웃이 없을 수 있습니다.

**과제:** Warp 경계에서 shuffle 연산이 미정의 데이터를 반환할 수 있는 경우를 처리하도록 알고리즘을 설계하세요.

`WARP_SIZE = 32`에서의 이웃 차분:

- **유효한 차분** (`lane < WARP_SIZE - 1`): **Lane 0-30** (31개 Lane)
  - **조건**: \\(\text{lane\_id}() \in \{0, 1, \cdots, 30\}\\)
  - **이유**: `shuffle_down(current_val, 1)`이 다음 이웃의 값을 성공적으로 가져옴
  - **결과**: `output[i] = input[i+1] - input[i]` (유한 차분)

- **경계 케이스** (else): **Lane 31** (1개 Lane)
  - **조건**: \\(\text{lane\_id}() = 31\\)
  - **이유**: `shuffle_down(current_val, 1)`이 미정의 데이터를 반환 (Lane 32가 없음)
  - **결과**: `output[i] = 0` (차분 계산 불가)

### 3. **Lane 식별**

```mojo
lane = lane_id()  # 0부터 WARP_SIZE-1까지 반환
```

**Lane 번호 매기기:** 각 Warp 내에서 Lane은 0, 1, 2, ..., `WARP_SIZE-1`로 번호가 매겨집니다

</div>
</details>

**이웃 차분 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p25 --neighbor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p25 --neighbor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p25 --neighbor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p25 --neighbor
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 0.0]
expected: [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 0.0]
✅ Basic neighbor difference test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p25/p25.mojo:neighbor_difference_solution}}
```

<div class="solution-explanation">

이 풀이는 `shuffle_down()`이 기존 배열 인덱싱을 효율적인 Warp 레벨 통신으로 어떻게 변환하는지 보여줍니다.

**알고리즘 분석:**

```mojo
if global_i < size:
    current_val = input[global_i]           # 각 Lane이 자신의 요소를 읽음
    next_val = shuffle_down(current_val, 1) # 하드웨어가 데이터를 오른쪽으로 이동

    if lane < WARP_SIZE - 1:
        output[global_i] = next_val - current_val  # 차분 계산
    else:
        output[global_i] = 0                       # 경계 처리
```

**SIMT 실행 상세 분석:**

```
사이클 1: 모든 Lane이 동시에 값을 로드
  Lane 0: current_val = input[0] = 0
  Lane 1: current_val = input[1] = 1
  Lane 2: current_val = input[2] = 4
  ...
  Lane 31: current_val = input[31] = 961

사이클 2: shuffle_down(current_val, 1)이 모든 Lane에서 실행
  Lane 0: Lane 1에서 current_val 수신 → next_val = 1
  Lane 1: Lane 2에서 current_val 수신 → next_val = 4
  Lane 2: Lane 3에서 current_val 수신 → next_val = 9
  ...
  Lane 30: Lane 31에서 current_val 수신 → next_val = 961
  Lane 31: 미정의 수신 (Lane 32 없음) → next_val = ?

사이클 3: 차분 계산 (Lane 0-30만 해당)
  Lane 0: output[0] = 1 - 0 = 1
  Lane 1: output[1] = 4 - 1 = 3
  Lane 2: output[2] = 9 - 4 = 5
  ...
  Lane 31: output[31] = 0 (경계 조건)
```

**수학적 통찰:** 이산 도함수 연산자 \\(D\\)를 구현합니다:
\\[\Large D[f](i) = f(i+1) - f(i)\\]

이차 입력 \\(f(i) = i^2\\)에 대해:
\\[\Large D[i^2] = (i+1)^2 - i^2 = i^2 + 2i + 1 - i^2 = 2i + 1\\]

**shuffle_down이 우월한 이유:**

1. **메모리 효율성**: 기존 방식은 `input[global_i + 1]` 로드가 필요하여 캐시 미스를 유발할 수 있음
2. **경계 안전성**: 범위 초과 접근 위험이 없음 - 하드웨어가 Warp 경계를 처리
3. **SIMT 최적화**: 단일 명령이 모든 Lane을 동시에 처리
4. **레지스터 통신**: 데이터가 메모리 계층 구조가 아닌 레지스터 사이를 이동

**성능 특성:**

- **Latency**: 1 사이클 (메모리 접근의 100+ 사이클 대비)
- **대역폭**: 0 바이트 (기존 방식의 스레드당 4바이트 대비)
- **병렬성**: 32개 Lane 모두 동시에 처리

</div>
</details>

## 2. 다중 offset 이동 평균

### 구성

- 벡터 크기: `SIZE_2 = 64` (멀티 블록 시나리오)
- 그리드 구성: `BLOCKS_PER_GRID = (2, 1)` 그리드당 블록 수
- 블록 구성: `THREADS_PER_BLOCK = (WARP_SIZE, 1)` 블록당 스레드 수

### 작성할 코드

여러 `shuffle_down` 연산을 사용하여 3점 이동 평균을 구현합니다.

**수학적 연산:** 세 개의 연속 요소를 사용하여 슬라이딩 윈도우 평균을 계산합니다:
\\[\Large \\text{output}[i] = \\frac{1}{3}\\left(\\text{input}[i] + \\text{input}[i+1] + \\text{input}[i+2]\\right)\\]

**경계 처리:** Warp 경계에서 알고리즘이 우아하게 적응합니다:

- **3점 전체 윈도우**: \\(\\text{output}[i] = \\frac{1}{3}\\sum_{k=0}^{2} \\text{input}[i+k]\\) - 모든 이웃이 사용 가능할 때
- **2점 윈도우**: \\(\\text{output}[i] = \\frac{1}{2}\\sum_{k=0}^{1} \\text{input}[i+k]\\) - 다음 이웃만 사용 가능할 때
- **1점 윈도우**: \\(\\text{output}[i] = \\text{input}[i]\\) - 이웃이 사용 불가할 때

이는 `shuffle_down()`이 Warp 범위 내에서 자동 경계 처리와 함께 효율적인 stencil 연산을 가능하게 하는 방법을 보여줍니다.

```mojo
{{#include ../../../../../problems/p25/p25.mojo:moving_average_3}}
```

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **다중 offset shuffle 패턴**

이 퍼즐은 여러 이웃에 동시에 접근해야 합니다. 서로 다른 offset으로 shuffle 연산을 사용해야 합니다.

**핵심 질문:**

- `input[i+1]`과 `input[i+2]`를 shuffle 연산으로 어떻게 가져올 수 있을까요?
- Shuffle offset과 이웃 거리의 관계는 무엇일까요?
- 같은 소스 값에 대해 여러 번 shuffle을 수행할 수 있을까요?

**시각화 개념:**

```
현재 Lane이 필요한 값: current_val, next_val, next_next_val
Shuffle offset:        0 (직접),    1,        2
```

**생각해 보세요:** 몇 번의 shuffle 연산이 필요하고, 어떤 offset을 사용해야 할까요?

### 2. **단계적 경계 처리**

단순한 이웃 차분과 달리, 이 퍼즐은 2개의 이웃에 접근해야 하므로 여러 경계 시나리오가 있습니다.

**고려할 경계 시나리오:**

- **전체 윈도우:** Lane이 두 이웃 모두 접근 가능 → 3개 값 모두 사용
- **부분 윈도우:** Lane이 1개 이웃만 접근 가능 → 2개 값 사용
- **윈도우 없음:** Lane이 이웃에 접근 불가 → 1개 값 사용

**비판적 사고:**

- 어떤 Lane이 각 카테고리에 해당할까요?
- 값이 적을 때 평균의 가중치를 어떻게 조정해야 할까요?
- 어떤 경계 조건을 검사해야 할까요?

**고려할 패턴:**

```
if (두 이웃 모두 접근 가능):
    # 3점 평균
elif (한 이웃만 접근 가능):
    # 2점 평균
else:
    # 1점 (평균 없음)
```

### 3. **멀티 블록 조정**

이 퍼즐은 여러 블록을 사용하며, 각 블록이 데이터의 다른 영역을 처리합니다.

**중요한 고려사항:**

- 각 블록은 Lane 0부터 WARP_SIZE-1까지의 자체 Warp를 가짐
- 경계 조건은 각 Warp 내에서 독립적으로 적용
- 블록마다 Lane 번호가 초기화됨

**생각해 볼 질문:**

- 경계 로직이 Block 0과 Block 1 모두에서 올바르게 동작하나요?
- Lane 경계와 전역 배열 경계를 모두 검사하고 있나요?
- 서로 다른 블록에서 `global_i`와 `lane_id()`의 관계는 어떻게 될까요?

**디버깅 팁:** 각 블록의 경계 Lane에서 어떤 일이 일어나는지 추적하여 로직을 테스트하세요.

</div>
</details>

**이동 평균 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p25 --average
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p25 --average
```

  </div>
  <div class="tab-content">

```bash
uv run poe p25 --average
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE_2:  64
output: HostBuffer([3.3333333, 6.3333335, 10.333333, 15.333333, 21.333334, 28.333334, 36.333332, 45.333332, 55.333332, 66.333336, 78.333336, 91.333336, 105.333336, 120.333336, 136.33333, 153.33333, 171.33333, 190.33333, 210.33333, 231.33333, 253.33333, 276.33334, 300.33334, 325.33334, 351.33334, 378.33334, 406.33334, 435.33334, 465.33334, 496.33334, 512.0, 528.0, 595.3333, 630.3333, 666.3333, 703.3333, 741.3333, 780.3333, 820.3333, 861.3333, 903.3333, 946.3333, 990.3333, 1035.3334, 1081.3334, 1128.3334, 1176.3334, 1225.3334, 1275.3334, 1326.3334, 1378.3334, 1431.3334, 1485.3334, 1540.3334, 1596.3334, 1653.3334, 1711.3334, 1770.3334, 1830.3334, 1891.3334, 1953.3334, 2016.3334, 2048.0, 2080.0])
expected: HostBuffer([3.3333333, 6.3333335, 10.333333, 15.333333, 21.333334, 28.333334, 36.333332, 45.333332, 55.333332, 66.333336, 78.333336, 91.333336, 105.333336, 120.333336, 136.33333, 153.33333, 171.33333, 190.33333, 210.33333, 231.33333, 253.33333, 276.33334, 300.33334, 325.33334, 351.33334, 378.33334, 406.33334, 435.33334, 465.33334, 496.33334, 512.0, 528.0, 595.3333, 630.3333, 666.3333, 703.3333, 741.3333, 780.3333, 820.3333, 861.3333, 903.3333, 946.3333, 990.3333, 1035.3334, 1081.3334, 1128.3334, 1176.3334, 1225.3334, 1275.3334, 1326.3334, 1378.3334, 1431.3334, 1485.3334, 1540.3334, 1596.3334, 1653.3334, 1711.3334, 1770.3334, 1830.3334, 1891.3334, 1953.3334, 2016.3334, 2048.0, 2080.0])
✅ Moving average test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p25/p25.mojo:moving_average_3_solution}}
```

<div class="solution-explanation">

이 풀이는 복잡한 stencil 연산을 위한 고급 다중 offset shuffle을 보여줍니다.

**전체 알고리즘 분석:**

```mojo
if global_i < size:
    # 단계 1: 여러 shuffle로 필요한 데이터 모두 확보
    current_val = input[global_i]                   # 직접 접근
    next_val = shuffle_down(current_val, 1)         # 오른쪽 이웃
    next_next_val = shuffle_down(current_val, 2)    # 오른쪽+1 이웃

    # 단계 2: 사용 가능한 데이터에 따른 적응형 계산
    if lane < WARP_SIZE - 2 and global_i < size - 2:
        # 3점 stencil 전체 사용 가능
        output[global_i] = (current_val + next_val + next_next_val) / 3.0
    elif lane < WARP_SIZE - 1 and global_i < size - 1:
        # 2점 stencil만 사용 가능 (Warp 경계 근처)
        output[global_i] = (current_val + next_val) / 2.0
    else:
        # Stencil 사용 불가 (Warp 경계)
        output[global_i] = current_val
```

**다중 offset 실행 추적 (`WARP_SIZE = 32`):**

```
초기 상태 (Block 0, 요소 0-31):
  Lane 0: current_val = input[0] = 1
  Lane 1: current_val = input[1] = 2
  Lane 2: current_val = input[2] = 4
  ...
  Lane 31: current_val = input[31] = X

첫 번째 shuffle: shuffle_down(current_val, 1)
  Lane 0: next_val = input[1] = 2
  Lane 1: next_val = input[2] = 4
  Lane 2: next_val = input[3] = 7
  ...
  Lane 30: next_val = input[31] = X
  Lane 31: next_val = 미정의

두 번째 shuffle: shuffle_down(current_val, 2)
  Lane 0: next_next_val = input[2] = 4
  Lane 1: next_next_val = input[3] = 7
  Lane 2: next_next_val = input[4] = 11
  ...
  Lane 29: next_next_val = input[31] = X
  Lane 30: next_next_val = 미정의
  Lane 31: next_next_val = 미정의

계산 단계:
  Lane 0-29: 3점 전체 평균 → (current + next + next_next) / 3
  Lane 30:   2점 평균 → (current + next) / 2
  Lane 31:   1점 평균 → current (그대로 전달)
```

**수학적 기반:** 가변 폭 이산 convolution을 구현합니다:
\\[\Large h[i] = \\sum_{k=0}^{K(i)-1} w_k^{(i)} \\cdot f[i+k]\\]

위치에 따라 커널이 적응합니다:

- **내부 점**: \\(K(i) = 3\\), \\(\\mathbf{w}^{(i)} = [\\frac{1}{3}, \\frac{1}{3}, \\frac{1}{3}]\\)
- **경계 근처**: \\(K(i) = 2\\), \\(\\mathbf{w}^{(i)} = [\\frac{1}{2}, \\frac{1}{2}]\\)
- **경계**: \\(K(i) = 1\\), \\(\\mathbf{w}^{(i)} = [1]\\)

**멀티 블록 조정:** `SIZE_2 = 64`와 2개 블록:

```
Block 0 (전역 인덱스 0-31):
  전역 인덱스 29, 30, 31에 Lane 경계 적용

Block 1 (전역 인덱스 32-63):
  전역 인덱스 61, 62, 63에 Lane 경계 적용
  Lane 번호 초기화: global_i=32 → lane=0, global_i=63 → lane=31
```

**성능 최적화:**

1. **병렬 데이터 확보**: 두 shuffle 연산이 동시에 실행
2. **조건부 분기**: GPU가 predication을 통해 분기 Lane을 효율적으로 처리
3. **메모리 coalescing**: 순차적 글로벌 메모리 접근 패턴이 GPU에 최적
4. **레지스터 재사용**: 모든 중간 값이 레지스터에 유지

**신호 처리 관점:** 이것은 임펄스 응답 \\(h[n] = \\frac{1}{3}[\\delta[n] + \\delta[n-1] + \\delta[n-2]]\\)를 가진 인과 FIR 필터로, 차단 주파수 \\(f_c \\approx 0.25f_s\\)에서 스무딩을 제공합니다.

</div>
</details>

## 요약

이 섹션의 핵심 패턴은 다음과 같습니다

```mojo
current_val = input[global_i]
neighbor_val = shuffle_down(current_val, offset)
if lane < WARP_SIZE - offset:
    result = compute(current_val, neighbor_val)
```

**핵심 장점:**

- **하드웨어 효율성**: 레지스터 간 직접 통신
- **경계 안전성**: 자동 Warp 범위 처리
- **SIMT 최적화**: 단일 명령, 모든 Lane 병렬 처리

**활용 분야**: 유한 차분, stencil 연산, 이동 평균, convolution.
