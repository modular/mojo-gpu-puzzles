<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# block.prefix_sum()과 병렬 히스토그램 구간 분류

이 퍼즐은 블록 레벨 [block.prefix_sum](https://docs.modular.com/mojo/std/gpu/primitives/block/prefix_sum) 연산을 사용하여 고급 병렬 필터링과 추출을 위한 병렬 히스토그램 구간 분류를 구현합니다. 각 스레드가 자신의 요소가 속할 대상 구간을 결정한 다음, `block.prefix_sum()`을 적용하여 특정 구간의 요소를 추출하기 위한 쓰기 위치를 계산합니다. prefix sum이 단순한 reduction을 넘어 고급 병렬 파티셔닝을 가능하게 하는 방법을 보여줍니다.

**핵심 통찰:** _[block.prefix_sum()](https://docs.modular.com/mojo/std/gpu/primitives/block/prefix_sum) 연산은 블록 내 모든 스레드에 걸쳐 일치하는 요소의 누적 쓰기 위치를 계산하여 병렬 필터링과 추출을 제공합니다._

## 핵심 개념

이 퍼즐에서 다루는 내용:

- `block.prefix_sum()`을 활용한 **블록 레벨 prefix sum**
- 누적 연산을 사용한 **병렬 필터링과 추출**
- **고급 병렬 파티셔닝** 알고리즘
- 블록 전체 조율을 통한 **히스토그램 구간 분류**
- **Exclusive vs inclusive** prefix sum 패턴

이 알고리즘은 특정 값 범위(구간)에 속하는 요소를 추출하여 히스토그램을 구성합니다:
\\[\Large \text{Bin}_k = \\{x_i : k/N \leq x_i < (k+1)/N\\}\\]

각 스레드가 자신의 요소가 속하는 구간을 결정하고, `block.prefix_sum()`이 병렬 추출을 조율합니다.

## 구성

- 벡터 크기: `SIZE = 128` 요소
- 데이터 타입: `DType.float32`
- 블록 구성: `(128, 1)` 블록당 스레드 수 (`TPB = 128`)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 구간 수: `NUM_BINS = 8` (범위 [0.0, 0.125), [0.125, 0.25) 등)
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)
- 블록당 Warp 수: `128 / WARP_SIZE` (GPU에 따라 2개 또는 4개)

## 도전 과제: 병렬 구간 추출

기존의 순차적 히스토그램 구성은 요소를 하나씩 처리합니다:

```python
# 순차적 방식 - 병렬화가 어려움
histogram = [[] for _ in range(NUM_BINS)]
for element in data:
    bin_id = int(element * NUM_BINS)  # 구간 결정
    histogram[bin_id].append(element)  # 순차적 추가
```

**단순한 GPU 병렬화의 문제점:**

- **경쟁 상태**: 여러 스레드가 같은 구간에 동시에 쓰기
- **비정렬 메모리 접근**: 스레드들이 서로 다른 메모리 위치에 접근
- **부하 불균형**: 일부 구간에 훨씬 많은 요소가 몰릴 수 있음
- **복잡한 동기화**: barrier와 atomic 연산이 필요

## 고급 방식: `block.prefix_sum()` 조율

복잡한 병렬 파티셔닝을 조율된 추출로 변환합니다:

## 완성할 코드

### `block.prefix_sum()` 방식

`block.prefix_sum()`을 사용하여 병렬 히스토그램 구간 분류를 구현합니다:

```mojo
{{#include ../../../../../problems/p27/p27.mojo:block_histogram}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p27/p27.mojo" class="filename">전체 파일 보기: problems/p27/p27.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **핵심 알고리즘 구조 (이전 퍼즐에서 적용)**

`block_sum_dot_product`와 마찬가지로 다음 핵심 변수가 필요합니다:

```mojo
global_i = block_dim.x * block_idx.x + thread_idx.x
local_i = thread_idx.x
```

함수는 **5가지 주요 단계**(총 약 15-20줄)로 구성됩니다:

1. 요소를 로드하고 구간을 결정
2. 대상 구간에 대한 이진 프레디케이트 생성
3. 프레디케이트에 `block.prefix_sum()` 실행
4. 계산된 offset을 사용하여 조건부 쓰기
5. 마지막 스레드가 총 개수를 계산

### 2. **구간 계산 (`math.floor` 사용)**

`Float32` 값을 구간으로 분류하려면:

```mojo
my_value = input_data[global_i][0]  # 내적에서처럼 SIMD 추출
bin_number = Int(floor(my_value * num_bins))
```

**경계 사례 처리**: 정확히 1.0인 값은 구간 `NUM_BINS`에 들어가지만, 실제 구간은 0부터 `NUM_BINS-1`까지입니다. `if` 문을 사용하여 최대 구간을 제한하세요.

### 3. **이진 프레디케이트 생성**

이 스레드의 요소가 target_bin에 속하는지를 나타내는 정수 변수(0 또는 1)를 만듭니다:

```mojo
var belongs_to_target: Int = 0
if (thread_has_valid_element) and (my_bin == target_bin):
    belongs_to_target = 1
```

이것이 핵심 통찰입니다: prefix sum이 이 이진 플래그에 작용하여 위치를 계산합니다!

### 4. **`block.prefix_sum()` 호출 패턴**

문서에 따르면 호출은 다음과 같습니다:

```mojo
offset = block.prefix_sum[
    dtype=DType.int32,         # 정수 프레디케이트로 작업
    block_size=tpb,            # block.sum()과 동일
    exclusive=True             # 핵심: 각 스레드 이전의 위치를 제공
](val=SIMD[DType.int32, 1](my_predicate_value))
```

**왜 exclusive인가?** 위치 5에서 프레디케이트=1인 스레드는, 자신 앞에 4개의 요소가 있었다면 output[4]에 써야 합니다.

### 5. **조건부 쓰기 패턴**

`belongs_to_target == 1`인 스레드만 기록해야 합니다:

```mojo
if belongs_to_target == 1:
    bin_output[Int(offset[0])] = my_value  # 인덱싱을 위해 SIMD를 Int로 변환
```

이것은 [Puzzle 12](../puzzle_12/layout_tensor.md)의 경계 검사 패턴과 동일하지만, 조건이 "대상 구간에 속하는지"로 바뀌었습니다.

### 6. **최종 개수 계산**

마지막 스레드(스레드 0이 아님!)가 총 개수를 계산합니다:

```mojo
if local_i == tpb - 1:  # 블록의 마지막 스레드
    total_count = offset[0] + belongs_to_target  # Inclusive = exclusive + 자신의 기여분
    count_output[0] = total_count
```

**왜 마지막 스레드인가?** 가장 높은 `offset` 값을 가지므로, `offset + 기여분`이 총 개수가 됩니다.

### 7. **데이터 타입과 변환**

이전 퍼즐의 패턴을 기억하세요:

- `LayoutTensor` 인덱싱은 SIMD를 반환: `input_data[i][0]`
- `block.prefix_sum()`은 SIMD를 반환: `offset[0]`으로 추출
- 배열 인덱싱은 `Int`가 필요: `bin_output[...]`에 `Int(offset[0])`

</div>
</details>

**block.prefix_sum() 방식 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p27 --histogram
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p27 --histogram
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p27 --histogram
```

  </div>
  <div class="tab-content">

```bash
uv run poe p27 --histogram
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
SIZE: 128
TPB: 128
NUM_BINS: 8

Input sample: 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 ...

=== Processing Bin 0 (range [ 0.0 , 0.125 )) ===
Bin 0 count: 26
Bin 0 extracted elements: 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 ...

=== Processing Bin 1 (range [ 0.125 , 0.25 )) ===
Bin 1 count: 24
Bin 1 extracted elements: 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 ...

=== Processing Bin 2 (range [ 0.25 , 0.375 )) ===
Bin 2 count: 26
Bin 2 extracted elements: 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 ...

=== Processing Bin 3 (range [ 0.375 , 0.5 )) ===
Bin 3 count: 22
Bin 3 extracted elements: 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 ...

=== Processing Bin 4 (range [ 0.5 , 0.625 )) ===
Bin 4 count: 13
Bin 4 extracted elements: 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 ...

=== Processing Bin 5 (range [ 0.625 , 0.75 )) ===
Bin 5 count: 12
Bin 5 extracted elements: 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 ...

=== Processing Bin 6 (range [ 0.75 , 0.875 )) ===
Bin 6 count: 5
Bin 6 extracted elements: 0.75 0.76 0.77 0.78 0.79

=== Processing Bin 7 (range [ 0.875 , 1.0 )) ===
Bin 7 count: 0
Bin 7 extracted elements:
```

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p27/p27.mojo:block_histogram_solution}}
```

<div class="solution-explanation">

`block.prefix_sum()` 커널은 이전 퍼즐의 개념을 기반으로 고급 병렬 조율 패턴을 보여줍니다:

## **단계별 알고리즘 분석:**

### **1단계: 요소 처리 ([Puzzle 12](../puzzle_12/layout_tensor.md) 내적과 유사)**

```
스레드 인덱싱 (익숙한 패턴):
  global_i = block_dim.x * block_idx.x + thread_idx.x  // 전역 요소 인덱스
  local_i = thread_idx.x                               // 로컬 스레드 인덱스

요소 로딩 (LayoutTensor 패턴과 동일):
  스레드 0:  my_value = input_data[0][0] = 0.00
  스레드 1:  my_value = input_data[1][0] = 0.01
  스레드 13: my_value = input_data[13][0] = 0.13
  스레드 25: my_value = input_data[25][0] = 0.25
  ...
```

### **2단계: 구간 분류 (새로운 개념)**

```
floor 연산을 사용한 구간 계산:
  스레드 0:  my_bin = Int(floor(0.00 * 8)) = 0  // 값 [0.000, 0.125) → 구간 0
  스레드 1:  my_bin = Int(floor(0.01 * 8)) = 0  // 값 [0.000, 0.125) → 구간 0
  스레드 13: my_bin = Int(floor(0.13 * 8)) = 1  // 값 [0.125, 0.250) → 구간 1
  스레드 25: my_bin = Int(floor(0.25 * 8)) = 2  // 값 [0.250, 0.375) → 구간 2
  ...
```

### **3단계: 이진 프레디케이트 생성 (필터링 패턴)**

```
target_bin=0에 대해 추출 마스크 생성:
  스레드 0:  belongs_to_target = 1  (구간 0 == 대상 0)
  스레드 1:  belongs_to_target = 1  (구간 0 == 대상 0)
  스레드 13: belongs_to_target = 0  (구간 1 != 대상 0)
  스레드 25: belongs_to_target = 0  (구간 2 != 대상 0)
  ...

이진 배열 생성: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
```

### **4단계: 병렬 prefix sum (마법이 일어나는 곳!)**

```
프레디케이트에 block.prefix_sum[exclusive=True] 적용:
입력:      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, ...]
Exclusive: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12, -, -, -, ...]
                                                      ^
                                                 중요하지 않음

핵심 통찰: 각 스레드가 출력 배열에서 자신의 쓰기 위치를 받습니다!
```

### **5단계: 조율된 추출 (조건부 쓰기)**

```
belongs_to_target=1인 스레드만 기록:
  스레드 0:  bin_output[0] = 0.00   // write_offset[0] = 0 사용
  스레드 1:  bin_output[1] = 0.01   // write_offset[1] = 1 사용
  스레드 12: bin_output[12] = 0.12  // write_offset[12] = 12 사용
  스레드 13: (기록 안 함)             // belongs_to_target = 0
  스레드 25: (기록 안 함)             // belongs_to_target = 0
  ...

결과: [0.00, 0.01, 0.02, ..., 0.12, ???, ???, ...] // 빈틈없이 채워짐!
```

### **6단계: 개수 계산 (block.sum() 패턴과 유사)**

```
마지막 스레드가 총 개수를 계산 (스레드 0이 아님!):
  if local_i == tpb - 1:  // 이 경우 스레드 127
      total = write_offset[0] + belongs_to_target  // Inclusive sum 공식
      count_output[0] = total
```

## **이 고급 알고리즘이 동작하는 이유:**

### **[Puzzle 12](../puzzle_12/layout_tensor.md) (기존 내적)과의 연결:**

- **동일한 스레드 인덱싱**: `global_i`와 `local_i` 패턴
- **동일한 경계 검사**: `if global_i < size` 검증
- **동일한 데이터 로딩**: `[0]`을 사용한 LayoutTensor SIMD 추출

### **[`block.sum()`](./block_sum.md) (이 퍼즐의 앞부분)과의 연결:**

- **동일한 블록 전체 연산**: 모든 스레드가 블록 기본 요소에 참여
- **동일한 결과 처리**: 특정 스레드(첫 번째 대신 마지막)가 최종 결과 처리
- **동일한 SIMD 변환**: 배열 인덱싱을 위한 `Int(result[0])` 패턴

### **`block.prefix_sum()`만의 고급 개념:**

- **모든 스레드가 결과를 받음**: 스레드 0만 중요한 `block.sum()`과 달리
- **조율된 쓰기 위치**: prefix sum이 경쟁 상태를 자동으로 제거
- **병렬 필터링**: 이진 프레디케이트가 고급 데이터 재구성을 가능하게 함

## **단순한 방식 대비 성능 이점:**

### **vs. Atomic 연산:**

- **경쟁 상태 없음**: prefix sum이 고유한 쓰기 위치를 제공
- **Coalesced 메모리**: 순차적 쓰기가 캐시 성능을 향상
- **직렬화 없음**: 모든 쓰기가 병렬로 수행

### **vs. 다중 패스 알고리즘:**

- **단일 kernel**: 한 번의 GPU 실행으로 히스토그램 추출 완료
- **완전 활용**: 데이터 분포에 관계없이 모든 스레드가 작업
- **최적 메모리 대역폭**: GPU 메모리 계층 구조에 최적화된 패턴

이것은 `block.prefix_sum()`이 `block.sum()` 같은 단순한 기본 요소로는 복잡하거나 불가능한 고급 병렬 알고리즘을 어떻게 가능하게 하는지 보여줍니다.

</div>
</details>

## 성능 인사이트

**`block.prefix_sum()` vs 기존 방식:**

- **알고리즘 정교함**: 고급 병렬 파티셔닝 vs 순차적 처리
- **메모리 효율**: coalesced 쓰기 vs 분산된 무작위 접근
- **동기화**: 내장 조율 vs 수동 barrier와 atomic
- **확장성**: 모든 블록 크기와 구간 수에 동작

**`block.prefix_sum()` vs `block.sum()`:**

- **범위**: 모든 스레드가 결과를 받음 vs 스레드 0만
- **용도**: 복잡한 파티셔닝 vs 단순한 집계
- **알고리즘 유형**: 병렬 scan 기본 요소 vs reduction 기본 요소
- **출력 패턴**: 스레드별 위치 vs 단일 합계

**`block.prefix_sum()`을 사용해야 할 때:**

- **병렬 필터링**: 조건에 맞는 요소 추출
- **Stream compaction**: 불필요한 요소 제거
- **병렬 파티셔닝**: 데이터를 카테고리별로 분리
- **고급 알고리즘**: 부하 분산, 정렬, 그래프 알고리즘

## 다음 단계

`block.prefix_sum()` 연산을 배웠으니, 다음으로 진행할 수 있습니다:

- **[block.broadcast()와 벡터 정규화](./block_broadcast.md)**: 블록 내 모든 스레드에 값을 공유
- **멀티 블록 알고리즘**: 더 큰 문제를 위한 여러 블록 간 조율
- **고급 병렬 알고리즘**: 정렬, 그래프 탐색, 동적 부하 분산
- **복잡한 메모리 패턴**: 블록 연산과 고급 메모리 접근의 결합

💡 **핵심 요점**: 블록 prefix sum 연산은 GPU 프로그래밍을 단순한 병렬 계산에서 고급 병렬 알고리즘으로 변환합니다. `block.sum()`이 reduction을 단순화했다면, `block.prefix_sum()`은 고성능 병렬 알고리즘에 필수적인 고급 데이터 재구성 패턴을 가능하게 합니다.
