<!-- i18n-source-commit: 1bae134f191aad7be60cc8612dcb64a50ef5ab2e -->

# `warp.broadcast()` 일대다 통신

Warp 레벨 조정에서는 `broadcast()`를 사용하여 하나의 Lane에서 Warp 내 다른 모든 Lane으로 데이터를 공유할 수 있습니다. 이 강력한 기본 요소를 통해 공유 메모리나 명시적 동기화 없이 블록 레벨 계산, 조건부 로직 조정, 일대다 통신 패턴을 효율적으로 수행할 수 있습니다.

**핵심 통찰:** _[broadcast()](https://docs.modular.com/mojo/stdlib/gpu/warp/broadcast) 연산은 SIMT 실행을 활용하여 하나의 Lane(보통 Lane 0)이 계산한 값을 같은 Warp의 모든 Lane에 전달하며, 효율적인 조정 패턴과 집합적 의사결정을 가능하게 합니다._

> **Broadcast 연산이란?** Broadcast 연산은 하나의 스레드가 값을 계산하고 그룹 내 다른 모든 스레드와 공유하는 통신 패턴입니다. 블록 레벨 통계 계산, 집합적 의사결정, Warp 내 모든 스레드에 설정 파라미터 전달 등의 조정 작업에 필수적입니다.

## 핵심 개념

이 퍼즐에서 배울 내용:

- `broadcast()`를 활용한 **Warp 레벨 broadcast**
- **일대다 통신** 패턴
- **집합 계산** 전략
- Lane 간 **조건부 조정**
- **Broadcast-shuffle 결합** 연산

`broadcast` 연산은 하나의 Lane(기본적으로 Lane 0)이 자신의 값을 다른 모든 Lane과 공유할 수 있게 합니다:
\\[\Large \text{broadcast}(\text{value}) = \text{value_from_lane_0_to_all_lanes}\\]

이를 통해 복잡한 조정 패턴이 간단한 Warp 레벨 연산으로 변환되어, 명시적 동기화 없이 효율적인 집합 계산이 가능합니다.

## Broadcast 개념

기존 조정 방식은 복잡한 공유 메모리 패턴이 필요합니다:

```mojo
# 기존 방식 - 복잡하고 오류가 발생하기 쉬움
shared_memory[lane] = local_computation()
sync_threads()  # 비용이 큰 동기화
if lane == 0:
    result = compute_from_shared_memory()
sync_threads()  # 또 다른 비용이 큰 동기화
final_result = shared_memory[0]  # 모든 스레드가 읽음
```

**기존 방식의 문제점:**

- **메모리 오버헤드**: 공유 메모리 할당이 필요
- **동기화**: 비용이 큰 barrier 연산이 여러 번 필요
- **복잡한 로직**: 공유 메모리 인덱스와 접근 패턴 관리
- **오류 발생 가능성**: 경쟁 상태가 쉽게 발생

`broadcast()`를 사용하면 조정이 간결해집니다:

```mojo
# Warp broadcast 방식 - 간단하고 안전
collective_value = 0
if lane == 0:
    collective_value = compute_block_statistic()
collective_value = broadcast(collective_value)  # 모든 Lane과 공유
result = use_collective_value(collective_value)
```

**Broadcast의 장점:**

- **메모리 오버헤드 제로**: 공유 메모리 불필요
- **자동 동기화**: SIMT 실행이 정확성을 보장
- **간단한 패턴**: 하나의 Lane이 계산하고 모든 Lane이 수신
- **조합 가능**: 다른 Warp 연산과 쉽게 결합

## 1. 기본 broadcast

Lane 0이 블록 레벨 통계를 계산하고 모든 Lane과 공유하는 기본 broadcast 패턴을 구현합니다.

**요구사항:**

- Lane 0이 현재 블록의 처음 4개 요소의 합을 계산해야 합니다
- 이 계산된 값을 `broadcast()`를 사용하여 Warp의 다른 모든 Lane과 공유해야 합니다
- 각 Lane은 이 공유된 값을 자신의 입력 요소에 더해야 합니다

**테스트 데이터:** 입력 `[1, 2, 3, 4, 5, 6, 7, 8, ...]`은 출력 `[11, 12, 13, 14, 15, 16, 17, 18, ...]`을 생성해야 합니다

**과제:** 하나의 Lane만 블록 레벨 계산을 수행하되, 모든 Lane이 그 결과를 자신의 개별 연산에 사용하려면 어떻게 조정해야 할까요?

### 구성

- 벡터 크기: `SIZE = WARP_SIZE` (GPU에 따라 32 또는 64)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 블록 구성: `(WARP_SIZE, 1)` 블록당 스레드 수
- 데이터 타입: `DType.float32`
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)

### 완성할 코드

```mojo
{{#include ../../../../../problems/p25/p25.mojo:basic_broadcast}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p25/p25.mojo" class="filename">전체 파일 보기: problems/p25/p25.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **broadcast 동작 방식 이해하기**

`broadcast(value)` 연산은 Lane 0의 값을 가져와 Warp의 모든 Lane에 전달합니다.

**핵심 통찰:** Broadcast에서는 Lane 0의 값만 의미가 있습니다. 다른 Lane의 값은 무시되지만, 모든 Lane이 Lane 0의 값을 수신합니다.

**시각화:**

```
Broadcast 전: Lane 0은 val₀, Lane 1은 val₁, Lane 2는 val₂, ...
Broadcast 후: Lane 0은 val₀, Lane 1은 val₀, Lane 2는 val₀, ...
```

**생각해 보세요:** Lane 0만 broadcast하려는 값을 계산하도록 하려면 어떻게 해야 할까요?

### 2. **Lane별 계산**

Lane 0이 특별한 계산을 수행하고 다른 Lane은 대기하도록 알고리즘을 설계합니다.

**고려할 패턴:**

```
var shared_value = 초기값
if lane == 0:
    # Lane 0만 계산
    shared_value = 특별한_계산()
# 모든 Lane이 broadcast에 참여
shared_value = broadcast(shared_value)
```

**핵심 질문:**

- Broadcast 전에 다른 Lane의 값은 어떤 상태여야 할까요?
- Lane 0이 broadcast할 올바른 값을 갖도록 하려면 어떻게 해야 할까요?

### 3. **집합적 활용**

Broadcast 후 모든 Lane이 같은 값을 갖게 되며, 이를 각자의 개별 계산에 활용할 수 있습니다.

**생각해 보세요:** 각 Lane이 broadcast 값과 자신의 로컬 데이터를 어떻게 결합할까요?

</div>
</details>

**기본 broadcast 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p25 --broadcast-basic
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p25 --broadcast-basic
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p25 --broadcast-basic
```

  </div>
  <div class="tab-content">

```bash
uv run poe p25 --broadcast-basic
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE:  32
output: HostBuffer([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0])
expected: HostBuffer([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0])
✅ Basic broadcast test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p25/p25.mojo:basic_broadcast_solution}}
```

<div class="solution-explanation">

이 풀이는 Warp 레벨 조정을 위한 기본 broadcast 패턴을 보여줍니다.

**알고리즘 분석:**

```mojo
if global_i < size:
    # 단계 1: Lane 0이 특별한 값을 계산
    var broadcast_value: output.element_type = 0.0
    if lane == 0:
        # Lane 0만 이 계산을 수행
        block_start = block_idx.x * block_dim.x
        var sum: output.element_type = 0.0
        for i in range(4):
            if block_start + i < size:
                sum += input[block_start + i]
        broadcast_value = sum

    # 단계 2: Lane 0의 값을 모든 Lane과 공유
    broadcast_value = broadcast(broadcast_value)

    # 단계 3: 모든 Lane이 broadcast 값을 활용
    output[global_i] = broadcast_value + input[global_i]
```

**SIMT 실행 추적:**

```
사이클 1: Lane별 계산
  Lane 0: input[0] + input[1] + input[2] + input[3] = 1+2+3+4 = 10을 계산
  Lane 1: broadcast_value는 0.0 유지 (Lane 0이 아님)
  Lane 2: broadcast_value는 0.0 유지 (Lane 0이 아님)
  ...
  Lane 31: broadcast_value는 0.0 유지 (Lane 0이 아님)

사이클 2: broadcast(broadcast_value) 실행
  Lane 0: 자신의 값 유지 → broadcast_value = 10.0
  Lane 1: Lane 0의 값 수신 → broadcast_value = 10.0
  Lane 2: Lane 0의 값 수신 → broadcast_value = 10.0
  ...
  Lane 31: Lane 0의 값 수신 → broadcast_value = 10.0

사이클 3: broadcast 값을 활용한 개별 계산
  Lane 0: output[0] = 10.0 + input[0] = 10.0 + 1.0 = 11.0
  Lane 1: output[1] = 10.0 + input[1] = 10.0 + 2.0 = 12.0
  Lane 2: output[2] = 10.0 + input[2] = 10.0 + 3.0 = 13.0
  ...
  Lane 31: output[31] = 10.0 + input[31] = 10.0 + 32.0 = 42.0
```

**Broadcast가 우월한 이유:**

1. **조정 효율성**: 단일 연산으로 모든 Lane을 조정
2. **메모리 효율성**: 공유 메모리 할당 불필요
3. **동기화 불필요**: SIMT 실행이 자동으로 조정을 처리
4. **확장 가능한 패턴**: Warp 크기와 무관하게 동일하게 동작

**성능 특성:**

- **Latency**: Broadcast 연산 1 사이클
- **대역폭**: 0 바이트 (레지스터 간 직접 통신)
- **조정**: 32개 Lane 모두 자동 동기화

</div>
</details>

## 2. 조건부 broadcast

Lane 0이 블록 데이터를 분석하고 모든 Lane에 영향을 미치는 결정을 내리는 조건부 조정을 구현합니다.

**요구사항:**

- Lane 0이 현재 블록의 처음 8개 요소를 분석하고 최댓값을 찾아야 합니다
- 이 최댓값을 `broadcast()`를 사용하여 다른 모든 Lane에 전달해야 합니다
- 각 Lane은 조건부 로직을 적용합니다: 자신의 요소가 최댓값의 절반보다 크면 2배로, 그렇지 않으면 절반으로 만듭니다

**테스트 데이터:** 입력 `[3, 1, 7, 2, 9, 4, 6, 8, ...]` (반복 패턴)은 출력 `[1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, ...]`을 생성해야 합니다

**과제:** 블록 레벨 분석과 요소별 조건부 변환을 모든 Lane에 걸쳐 어떻게 조정할까요?

### 구성

- 벡터 크기: `SIZE = WARP_SIZE` (GPU에 따라 32 또는 64)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 블록 구성: `(WARP_SIZE, 1)` 블록당 스레드 수

### 완성할 코드

```mojo
{{#include ../../../../../problems/p25/p25.mojo:conditional_broadcast}}
```

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **분석과 의사결정**

Lane 0이 여러 데이터 포인트를 분석하고 다른 모든 Lane의 동작을 안내할 결정을 내려야 합니다.

**핵심 질문:**

- Lane 0이 여러 요소를 효율적으로 분석하려면 어떻게 해야 할까요?
- Lane의 동작을 조정하기 위해 어떤 종류의 결정을 broadcast해야 할까요?
- 데이터를 분석할 때 경계 조건은 어떻게 처리할까요?

**고려할 패턴:**

```
var decision = 기본값
if lane == 0:
    # 블록 로컬 데이터 분석
    decision = 분석_후_결정()
decision = broadcast(decision)
```

### 2. **조건부 실행 조정**

Broadcast된 결정을 수신한 후, 모든 Lane이 그 결정에 기반하여 서로 다른 로직을 적용해야 합니다.

**생각해 보세요:**

- Lane이 broadcast 값을 사용하여 로컬 결정을 내리는 방법은?
- 각 조건부 분기에서 어떤 연산을 적용해야 할까요?
- 모든 Lane에서 일관된 동작을 보장하려면 어떻게 해야 할까요?

**조건부 패턴:**

```
if (로컬_데이터가 broadcast_기준을 충족):
    # 하나의 변환 적용
else:
    # 다른 변환 적용
```

### 3. **데이터 분석 전략**

Lane 0이 여러 데이터 포인트를 효율적으로 분석하는 방법을 고려하세요.

**고려할 접근법:**

- 최댓값/최솟값 찾기
- 평균이나 합계 계산
- 패턴이나 임계값 감지
- 데이터 특성에 기반한 이진 결정

</div>
</details>

**조건부 broadcast 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p25 --broadcast-conditional
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p25 --broadcast-conditional
```

  </div>
  <div class="tab-content">

```bash
uv run poe p25 --broadcast-conditional
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE:  32
output: HostBuffer([1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0])
expected: HostBuffer([1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0, 1.5, 0.5, 14.0, 1.0, 18.0, 2.0, 12.0, 16.0])
✅ Conditional broadcast test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p25/p25.mojo:conditional_broadcast_solution}}
```

<div class="solution-explanation">

이 풀이는 Lane 간 조건부 조정을 위한 고급 broadcast 패턴을 보여줍니다.

**전체 알고리즘 분석:**

```mojo
if global_i < size:
    # 단계 1: Lane 0이 블록 데이터를 분석하고 결정을 내림
    var decision_value: output.element_type = 0.0
    if lane == 0:
        # 블록의 처음 8개 요소 중 최댓값 찾기
        block_start = block_idx.x * block_dim.x
        decision_value = input[block_start] if block_start < size else 0.0
        for i in range(1, min(8, min(WARP_SIZE, size - block_start))):
            if block_start + i < size:
                current_val = input[block_start + i]
                if current_val > decision_value:
                    decision_value = current_val

    # 단계 2: 결정을 broadcast하여 모든 Lane을 조정
    decision_value = broadcast(decision_value)

    # 단계 3: 모든 Lane이 broadcast에 기반한 조건부 로직을 적용
    current_input = input[global_i]
    threshold = decision_value / 2.0
    if current_input >= threshold:
        output[global_i] = current_input * 2.0  # 임계값 이상이면 2배
    else:
        output[global_i] = current_input / 2.0  # 임계값 미만이면 절반
```

**의사결정 실행 추적:**

```
입력 데이터: [3.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 8.0, ...]

단계 1: Lane 0이 처음 8개 요소의 최댓값을 찾음
  Lane 0 분석:
    input[0] = 3.0으로 시작
    input[1] = 1.0과 비교 → 3.0 유지
    input[2] = 7.0과 비교 → 7.0으로 갱신
    input[3] = 2.0과 비교 → 7.0 유지
    input[4] = 9.0과 비교 → 9.0으로 갱신
    input[5] = 4.0과 비교 → 9.0 유지
    input[6] = 6.0과 비교 → 9.0 유지
    input[7] = 8.0과 비교 → 9.0 유지
    최종 decision_value = 9.0

단계 2: decision_value = 9.0을 모든 Lane에 broadcast
  모든 Lane: decision_value = 9.0, threshold = 4.5

단계 3: Lane별 조건부 실행
  Lane 0: input[0] = 3.0 < 4.5 → output[0] = 3.0 / 2.0 = 1.5
  Lane 1: input[1] = 1.0 < 4.5 → output[1] = 1.0 / 2.0 = 0.5
  Lane 2: input[2] = 7.0 ≥ 4.5 → output[2] = 7.0 * 2.0 = 14.0
  Lane 3: input[3] = 2.0 < 4.5 → output[3] = 2.0 / 2.0 = 1.0
  Lane 4: input[4] = 9.0 ≥ 4.5 → output[4] = 9.0 * 2.0 = 18.0
  Lane 5: input[5] = 4.0 < 4.5 → output[5] = 4.0 / 2.0 = 2.0
  Lane 6: input[6] = 6.0 ≥ 4.5 → output[6] = 6.0 * 2.0 = 12.0
  Lane 7: input[7] = 8.0 ≥ 4.5 → output[7] = 8.0 * 2.0 = 16.0
  ...나머지 Lane에 패턴 반복
```

**수학적 기반:** 임계값 기반 변환을 구현합니다:
\\[\Large f(x) = \\begin{cases}
2x & \\text{if } x \\geq \\tau \\\\
\\frac{x}{2} & \\text{if } x < \\tau
\\end{cases}\\]

여기서 \\(\\tau = \\frac{\\max(\\text{block\_data})}{2}\\)는 broadcast된 임계값입니다.

**조정 패턴의 장점:**

1. **중앙화된 분석**: 하나의 Lane이 분석하고 모든 Lane이 혜택을 받음
2. **일관된 결정**: 모든 Lane이 같은 임계값을 사용
3. **적응형 동작**: 임계값이 블록 로컬 데이터 특성에 따라 적응
4. **효율적 조정**: 단일 broadcast로 복잡한 조건부 로직을 조정

**활용 분야:**

- **적응형 알고리즘**: 로컬 데이터 특성에 따라 파라미터 조정
- **품질 관리**: 데이터 품질 지표에 따라 다른 처리 적용
- **부하 분산**: 블록 로컬 복잡도 분석에 기반한 작업 분배

</div>
</details>

## 3. Broadcast-shuffle 조정

`broadcast()`와 `shuffle_down()`을 모두 결합한 고급 조정을 구현합니다.

**요구사항:**

- Lane 0이 블록의 처음 4개 요소의 평균을 계산하고 이 스케일링 팩터를 모든 Lane에 broadcast해야 합니다
- 각 Lane은 `shuffle_down(offset=1)`을 사용하여 다음 이웃의 값을 가져와야 합니다
- 대부분의 Lane: 스케일링 팩터에 `(현재_값 + 다음_이웃_값)`을 곱합니다
- Warp의 마지막 Lane: 스케일링 팩터에 `현재_값`만 곱합니다 (유효한 이웃 없음)

**테스트 데이터:** 입력은 `[2, 4, 6, 8, 1, 3, 5, 7, ...]` 패턴을 따릅니다 (처음 4개 요소: 2,4,6,8 이후 1,3,5,7 반복)

- Lane 0이 스케일링 팩터를 계산: `(2+4+6+8)/4 = 5.0`
- 예상 출력: `[30.0, 50.0, 70.0, 45.0, 20.0, 40.0, 60.0, 40.0, ...]`

**과제:** 하나의 Lane의 계산이 모든 Lane에 영향을 미치면서, 각 Lane이 자신의 이웃 데이터에도 접근해야 하는 상황에서 여러 Warp 기본 요소를 어떻게 조정할까요?

### 구성

- 벡터 크기: `SIZE = WARP_SIZE` (GPU에 따라 32 또는 64)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 블록 구성: `(WARP_SIZE, 1)` 블록당 스레드 수

### 완성할 코드

```mojo
{{#include ../../../../../problems/p25/p25.mojo:broadcast_shuffle_coordination}}
```

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **다중 기본 요소 조정**

이 퍼즐은 broadcast와 shuffle 연산을 순서대로 조율해야 합니다.

**흐름을 생각해 보세요:**

1. 하나의 Lane이 전체 Warp를 위한 값을 계산
2. 이 값이 모든 Lane에 broadcast됨
3. 각 Lane이 shuffle로 이웃 데이터에 접근
4. Broadcast 값이 이웃 데이터의 처리 방식에 영향

**조정 패턴:**

```
# 단계 1: Broadcast 조정
var shared_param = lane_0이면_계산()
shared_param = broadcast(shared_param)

# 단계 2: Shuffle 이웃 접근
current_val = input[global_i]
neighbor_val = shuffle_down(current_val, offset)

# 단계 3: 결합 계산
result = 결합(current_val, neighbor_val, shared_param)
```

### 2. **파라미터 계산 전략**

이웃 연산을 스케일링하는 데 유용한 블록 레벨 파라미터가 무엇일지 고려하세요.

**탐구할 질문:**

- Lane 0이 블록 데이터에서 어떤 통계를 계산해야 할까요?
- 이 파라미터가 이웃 기반 계산에 어떤 영향을 미쳐야 할까요?
- Shuffle 연산이 포함될 때 Warp 경계에서 무슨 일이 일어날까요?

### 3. **결합 연산 설계**

Broadcast 파라미터와 shuffle 기반 이웃 접근을 의미 있게 결합하는 방법을 생각하세요.

**패턴 고려사항:**

- Broadcast 파라미터가 입력, 출력, 또는 계산을 스케일링해야 할까요?
- Shuffle이 미정의 데이터를 반환하는 경계 케이스를 어떻게 처리할까요?
- 가장 효율적인 연산 순서는 무엇일까요?

</div>
</details>

**Broadcast-shuffle 조정 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p25 --broadcast-shuffle-coordination
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p25 --broadcast-shuffle-coordination
```

  </div>
  <div class="tab-content">

```bash
uv run poe p25 --broadcast-shuffle-coordination
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE:  32
output: HostBuffer([30.0, 50.0, 70.0, 45.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 35.0])
expected: HostBuffer([30.0, 50.0, 70.0, 45.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 40.0, 20.0, 40.0, 60.0, 35.0])
✅ Broadcast + Shuffle coordination test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p25/p25.mojo:broadcast_shuffle_coordination_solution}}
```

<div class="solution-explanation">

이 풀이는 broadcast와 shuffle 기본 요소를 결합한 가장 고급 Warp 조정 패턴을 보여줍니다.

**전체 알고리즘 분석:**

```mojo
if global_i < size:
    # 단계 1: Lane 0이 블록 로컬 스케일링 팩터를 계산
    var scale_factor: output.element_type = 0.0
    if lane == 0:
        block_start = block_idx.x * block_dim.x
        var sum: output.element_type = 0.0
        for i in range(4):
            if block_start + i < size:
                sum += input[block_start + i]
        scale_factor = sum / 4.0

    # 단계 2: 스케일링 팩터를 모든 Lane에 broadcast
    scale_factor = broadcast(scale_factor)

    # 단계 3: 각 Lane이 shuffle을 통해 현재 값과 다음 값을 가져옴
    current_val = input[global_i]
    next_val = shuffle_down(current_val, 1)

    # 단계 4: Broadcast 팩터를 이웃 조정과 결합하여 적용
    if lane < WARP_SIZE - 1 and global_i < size - 1:
        output[global_i] = (current_val + next_val) * scale_factor
    else:
        output[global_i] = current_val * scale_factor
```

**다중 기본 요소 실행 추적:**

```
입력 데이터: [2, 4, 6, 8, 1, 3, 5, 7, ...]

단계 1: Lane 0이 스케일링 팩터를 계산
  Lane 0 계산: (input[0] + input[1] + input[2] + input[3]) / 4
              = (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5.0
  다른 Lane: scale_factor는 0.0 유지

단계 2: scale_factor = 5.0을 모든 Lane에 broadcast
  모든 Lane: scale_factor = 5.0

단계 3: 이웃 접근을 위한 shuffle 연산
  Lane 0: current_val = input[0] = 2, next_val = shuffle_down(2, 1) = input[1] = 4
  Lane 1: current_val = input[1] = 4, next_val = shuffle_down(4, 1) = input[2] = 6
  Lane 2: current_val = input[2] = 6, next_val = shuffle_down(6, 1) = input[3] = 8
  Lane 3: current_val = input[3] = 8, next_val = shuffle_down(8, 1) = input[4] = 1
  ...
  Lane 31: current_val = input[31], next_val = 미정의

단계 4: Broadcast 스케일링과 결합한 계산
  Lane 0: output[0] = (2 + 4) * 5.0 = 6 * 5.0 = 30.0
  Lane 1: output[1] = (4 + 6) * 5.0 = 10 * 5.0 = 50.0
  Lane 2: output[2] = (6 + 8) * 5.0 = 14 * 5.0 = 70.0
  Lane 3: output[3] = (8 + 1) * 5.0 = 9 * 5.0 = 45.0
  ...
  Lane 31: output[31] = 7 * 5.0 = 35.0 (경계 - 이웃 없음)
```

**통신 패턴 분석:**
이 알고리즘은 **계층적 조정 패턴**을 구현합니다:

1. **수직 조정** (broadcast): Lane 0 → 모든 Lane
2. **수평 조정** (shuffle): Lane i → Lane i+1
3. **결합 계산**: Broadcast 데이터와 shuffle 데이터를 모두 활용

**수학적 기반:**
\\[\Large \\text{output}[i] = \\begin{cases}
(\\text{input}[i] + \\text{input}[i+1]) \\cdot \\beta & \\text{if lane } i < \\text{WARP\_SIZE} - 1 \\\\
\\text{input}[i] \\cdot \\beta & \\text{if lane } i = \\text{WARP\_SIZE} - 1
\\end{cases}\\]

여기서 \\(\\beta = \\frac{1}{4}\\sum_{k=0}^{3} \\text{input}[\\text{block\_start} + k]\\)는 broadcast된 스케일링 팩터입니다.

**고급 조정의 장점:**

1. **다단계 통신**: 전역(broadcast)과 지역(shuffle) 조정의 결합
2. **적응형 스케일링**: 블록 레벨 파라미터가 이웃 연산에 영향
3. **효율적 구성**: 두 기본 요소가 매끄럽게 협력
4. **복잡한 알고리즘 구현**: 정교한 병렬 알고리즘을 가능하게 함

**실제 활용 사례:**

- **적응형 필터링**: 블록 레벨 노이즈 추정과 이웃 기반 필터링
- **동적 부하 분산**: 전역 작업 분배와 로컬 조정
- **다중 스케일 처리**: 전역 파라미터가 로컬 stencil 연산을 제어

</div>
</details>

## 요약

이 섹션의 핵심 패턴은 다음과 같습니다

```mojo
var shared_value = initial_value
if lane == 0:
    shared_value = compute_block_statistic()
shared_value = broadcast(shared_value)
result = use_shared_value(shared_value, local_data)
```

**핵심 장점:**

- **일대다 조정**: 하나의 Lane이 계산하고 모든 Lane이 혜택을 받음
- **동기화 오버헤드 제로**: SIMT 실행이 조정을 처리
- **조합 가능한 패턴**: Shuffle과 다른 Warp 연산과 쉽게 결합

**활용 분야**: 블록 통계, 집합적 의사결정, 파라미터 공유, 적응형 알고리즘.
