<!-- i18n-source-commit: 1bae134f191aad7be60cc8612dcb64a50ef5ab2e -->

# `warp.prefix_sum()` 하드웨어 최적화 병렬 Scan

Warp 레벨 병렬 scan 연산에서는 `prefix_sum()`을 사용하여 복잡한 공유 메모리 알고리즘을 하드웨어 최적화 기본 요소로 대체할 수 있습니다. 이 강력한 연산을 통해 수십 줄의 공유 메모리 및 동기화 코드가 필요했을 효율적인 누적 계산, 병렬 파티셔닝, 고급 조정 알고리즘을 구현할 수 있습니다.

**핵심 통찰:** _[prefix_sum()](https://docs.modular.com/mojo/stdlib/gpu/warp/prefix_sum) 연산은 하드웨어 가속 병렬 scan을 활용하여 Warp Lane에 걸쳐 \\(O(\\log n)\\) 복잡도로 누적 연산을 수행하며, 복잡한 다단계 알고리즘을 단일 함수 호출로 대체합니다._

> **병렬 scan이란?** [병렬 scan (prefix sum)](https://en.wikipedia.org/wiki/Prefix_sum)은 데이터 요소에 걸쳐 누적 연산을 수행하는 기본적인 병렬 기본 요소입니다. 덧셈의 경우 `[a, b, c, d]`를 `[a, a+b, a+b+c, a+b+c+d]`로 변환합니다. 이 연산은 stream compaction, quicksort 파티셔닝, 병렬 정렬 같은 병렬 알고리즘에 필수적입니다.

## 핵심 개념

이 퍼즐에서 배울 내용:

- `prefix_sum()`을 활용한 **하드웨어 최적화 병렬 scan**
- **Inclusive vs exclusive prefix sum** 패턴
- 데이터 재배치를 위한 **Warp 레벨 stream compaction**
- 여러 Warp 기본 요소를 결합한 **고급 병렬 파티셔닝**
- 복잡한 공유 메모리를 대체하는 **단일 Warp 알고리즘 최적화**

이를 통해 다단계 공유 메모리 알고리즘이 우아한 단일 함수 호출로 변환되어, 명시적 동기화 없이 효율적인 병렬 scan 연산이 가능합니다.

## 1. Warp inclusive prefix sum

### 구성

- 벡터 크기: `SIZE = WARP_SIZE` (GPU에 따라 32 또는 64)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 블록 구성: `(WARP_SIZE, 1)` 블록당 스레드 수
- 데이터 타입: `DType.float32`
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)

### prefix_sum의 이점

기존 prefix sum은 복잡한 다단계 공유 메모리 알고리즘이 필요합니다. [Puzzle 14: Prefix Sum](../puzzle_14/puzzle_14.md)에서는 명시적 공유 메모리 관리로 이를 힘들게 구현했습니다:

```mojo
{{#include ../../../../../solutions/p14/p14.mojo:prefix_sum_simple_solution}}
```

**기존 방식의 문제점:**

- **메모리 오버헤드**: 공유 메모리 할당이 필요
- **다중 barrier**: 복잡한 다단계 동기화
- **복잡한 인덱싱**: 수동 stride 계산과 경계 검사
- **낮은 확장성**: 각 단계 사이에 barrier가 필요한 \\(O(\\log n)\\) 단계

`prefix_sum()`을 사용하면 병렬 scan이 간단해집니다:

```mojo
# 하드웨어 최적화 방식 - 단일 함수 호출!
current_val = input[global_i]
scan_result = prefix_sum[exclusive=False](current_val)
output[global_i] = scan_result
```

**prefix_sum의 장점:**

- **메모리 오버헤드 제로**: 하드웨어 가속 연산
- **동기화 불필요**: 단일 atomic 연산
- **하드웨어 최적화**: 전용 scan 유닛 활용
- **완벽한 확장성**: 모든 `WARP_SIZE` (32, 64 등)에서 동작

### 완성할 코드

하드웨어 최적화 `prefix_sum()` 기본 요소를 사용하여 inclusive prefix sum을 구현합니다.

**수학적 연산:** 각 Lane이 자신의 위치까지 모든 요소의 합을 포함하는 누적 합을 계산합니다:
\\[\Large \\text{output}[i] = \\sum_{j=0}^{i} \\text{input}[j]\\]

입력 데이터 `[1, 2, 3, 4, 5, ...]`를 누적 합 `[1, 3, 6, 10, 15, ...]`으로 변환하며, 각 위치에 이전 모든 요소와 자기 자신의 합이 담깁니다.

```mojo
{{#include ../../../../../problems/p26/p26.mojo:warp_inclusive_prefix_sum}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p26/p26.mojo" class="filename">전체 파일 보기: problems/p26/p26.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **prefix_sum 매개변수 이해하기**

`prefix_sum()` 함수에는 scan 유형을 제어하는 중요한 템플릿 매개변수가 있습니다.

**핵심 질문:**

- Inclusive prefix sum과 exclusive prefix sum의 차이는 무엇인가요?
- 어떤 매개변수가 이 동작을 제어하나요?
- Inclusive scan에서 각 Lane은 무엇을 출력해야 하나요?

**힌트**: 함수 시그니처를 보고 누적 연산에서 "inclusive"가 무엇을 의미하는지 생각해 보세요.

### 2. **단일 Warp 제한**

이 하드웨어 기본 요소는 단일 Warp 내에서만 동작합니다. 이 제한의 의미를 생각해 보세요.

**생각해 보세요:**

- 여러 Warp가 있으면 어떻게 되나요?
- 이 제한을 이해하는 것이 왜 중요한가요?
- 멀티 Warp 시나리오로 확장하려면 어떻게 해야 하나요?

### 3. **데이터 타입 고려사항**

`prefix_sum` 함수는 최적 성능을 위해 특정 데이터 타입을 요구할 수 있습니다.

**고려할 점:**

- 입력이 어떤 데이터 타입을 사용하나요?
- `prefix_sum`이 특정 스칼라 타입을 기대하나요?
- 필요한 경우 타입 변환을 어떻게 처리하나요?

</div>
</details>

**Warp inclusive prefix sum 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p26 --prefix-sum
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p26 --prefix-sum
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p26 --prefix-sum
```

  </div>
  <div class="tab-content">

```bash
uv run poe p26 --prefix-sum
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE:  32
output: [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0, 91.0, 105.0, 120.0, 136.0, 153.0, 171.0, 190.0, 210.0, 231.0, 253.0, 276.0, 300.0, 325.0, 351.0, 378.0, 406.0, 435.0, 465.0, 496.0, 528.0]
expected: [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0, 91.0, 105.0, 120.0, 136.0, 153.0, 171.0, 190.0, 210.0, 231.0, 253.0, 276.0, 300.0, 325.0, 351.0, 378.0, 406.0, 435.0, 465.0, 496.0, 528.0]
✅ Warp inclusive prefix sum test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p26/p26.mojo:warp_inclusive_prefix_sum_solution}}
```

<div class="solution-explanation">

이 풀이는 `prefix_sum()`이 복잡한 다단계 알고리즘을 하드웨어 최적화된 단일 함수 호출로 어떻게 대체하는지 보여줍니다.

**알고리즘 분석:**

```mojo
if global_i < size:
    current_val = input[global_i]

    # 이 한 줄이 Puzzle 14의 복잡한 공유 메모리 로직 ~30줄을 대체합니다!
    # 단, 현재 Warp (WARP_SIZE 스레드) 내에서만 동작합니다
    scan_result = prefix_sum[exclusive=False](
        rebind[Scalar[dtype]](current_val)
    )

    output[global_i] = scan_result
```

**SIMT 실행 상세 분석:**

```
입력: [1, 2, 3, 4, 5, 6, 7, 8, ...]

사이클 1: 모든 Lane이 동시에 값을 로드
  Lane 0: current_val = 1
  Lane 1: current_val = 2
  Lane 2: current_val = 3
  Lane 3: current_val = 4
  ...
  Lane 31: current_val = 32

사이클 2: prefix_sum[exclusive=False] 실행 (하드웨어 가속)
  Lane 0: scan_result = 1 (요소 0~0의 합)
  Lane 1: scan_result = 3 (요소 0~1의 합: 1+2)
  Lane 2: scan_result = 6 (요소 0~2의 합: 1+2+3)
  Lane 3: scan_result = 10 (요소 0~3의 합: 1+2+3+4)
  ...
  Lane 31: scan_result = 528 (요소 0~31의 합)

사이클 3: 결과 저장
  Lane 0: output[0] = 1
  Lane 1: output[1] = 3
  Lane 2: output[2] = 6
  Lane 3: output[3] = 10
  ...
```

**수학적 통찰:** inclusive prefix sum 연산을 구현합니다:
\\[\Large \\text{output}[i] = \\sum_{j=0}^{i} \\text{input}[j]\\]

**Puzzle 14 방식과의 비교:**

- **[Puzzle 14: Prefix Sum](../puzzle_14/puzzle_14.md)**: 공유 메모리 ~30줄 + 다중 barrier + 복잡한 인덱싱
- **Warp 기본 요소**: 하드웨어 가속의 함수 호출 1개
- **성능**: 같은 \\(O(\\log n)\\) 복잡도이지만, 전용 하드웨어에서 구현
- **메모리**: 명시적 할당 대비 공유 메모리 사용량 제로

**Puzzle 12에서의 발전:** 현대 GPU 아키텍처의 강력함을 보여줍니다 - Puzzle 12에서 신중한 수동 구현이 필요했던 것이 이제는 하드웨어 가속 기본 요소 하나로 해결됩니다. Warp 레벨 `prefix_sum()`은 구현 복잡도 제로로 같은 알고리즘적 이점을 제공합니다.

**prefix_sum이 우월한 이유:**

1. **하드웨어 가속**: 현대 GPU의 전용 scan 유닛
2. **메모리 오버헤드 제로**: 공유 메모리 할당 불필요
3. **자동 동기화**: 명시적 barrier 불필요
4. **완벽한 확장성**: 모든 `WARP_SIZE`에서 최적으로 동작

**성능 특성:**

- **Latency**: ~1-2 사이클 (하드웨어 scan 유닛)
- **대역폭**: 메모리 트래픽 제로 (레지스터 전용 연산)
- **병렬성**: `WARP_SIZE`개 Lane 모두 동시에 참여
- **확장성**: 하드웨어 최적화를 동반한 \\(O(\\log n)\\) 복잡도

**중요한 제한사항**: 이 기본 요소는 단일 Warp 내에서만 동작합니다. 멀티 Warp 시나리오에서는 Warp 간 추가 조정이 필요합니다.

</div>
</details>

## 2. Warp 파티션

### 구성

- 벡터 크기: `SIZE = WARP_SIZE` (GPU에 따라 32 또는 64)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 블록 구성: `(WARP_SIZE, 1)` 블록당 스레드 수

### 완성할 코드

`shuffle_xor`과 `prefix_sum` 기본 요소를 **모두** 사용하여 단일 Warp 병렬 파티셔닝을 구현합니다.

**수학적 연산:** 피벗 값을 기준으로 요소를 분할하여, `< pivot`인 요소는 왼쪽에, `>= pivot`인 요소는 오른쪽에 배치합니다:
\\[\Large \\text{output} = [\\text{elements} < \\text{pivot}] \\,|\\, [\\text{elements} \\geq \\text{pivot}]\\]

**고급 알고리즘:** 이 알고리즘은 두 가지 정교한 Warp 기본 요소를 결합합니다:

1. **`shuffle_xor()`**: 왼쪽 요소 개수를 세기 위한 Warp 레벨 butterfly reduction
2. **`prefix_sum()`**: 각 파티션 내 위치 계산을 위한 exclusive scan

이는 단일 Warp 내에서 여러 Warp 기본 요소를 결합하여 복잡한 병렬 알고리즘을 구현하는 강력함을 보여줍니다.

```mojo
{{#include ../../../../../problems/p26/p26.mojo:warp_partition}}
```

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **다단계 알고리즘 구조**

이 알고리즘은 여러 조정된 단계가 필요합니다. 파티셔닝에 필요한 논리적 단계를 생각해 보세요.

**고려할 핵심 단계:**

- 어떤 요소가 어느 파티션에 속하는지 어떻게 식별하나요?
- 각 파티션 내에서 위치를 어떻게 계산하나요?
- 왼쪽 파티션의 전체 크기를 어떻게 알 수 있나요?
- 최종 위치에 요소를 어떻게 기록하나요?

### 2. **프레디케이트 생성**

어느 파티션에 속하는지 판별하는 불리언 프레디케이트를 만들어야 합니다.

**생각해 보세요:**

- "이 요소는 왼쪽 파티션에 속한다"를 어떻게 표현하나요?
- "이 요소는 오른쪽 파티션에 속한다"를 어떻게 표현하나요?
- `prefix_sum`에 전달할 프레디케이트는 어떤 데이터 타입이어야 하나요?

### 3. **shuffle_xor과 prefix_sum 결합**

이 알고리즘은 두 Warp 기본 요소를 서로 다른 목적으로 사용합니다.

**고려할 점:**

- 이 맥락에서 `shuffle_xor`은 무엇에 사용되나요?
- 이 맥락에서 `prefix_sum`은 무엇에 사용되나요?
- 이 두 연산이 어떻게 함께 동작하나요?

### 4. **위치 계산**

가장 까다로운 부분은 각 요소가 출력에서 어디에 기록되어야 하는지 계산하는 것입니다.

**핵심 통찰:**

- 왼쪽 파티션 요소: 최종 위치를 무엇이 결정하나요?
- 오른쪽 파티션 요소: offset을 어떻게 올바르게 적용하나요?
- 로컬 위치와 파티션 경계를 어떻게 결합하나요?

</div>
</details>

**Warp 파티션 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p26 --partition
```

  </div>
  <div class="tab-content">

```bash
pixi run p26 --partition
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
WARP_SIZE:  32
SIZE:  32
output: HostBuffer([3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 3.0, 1.0, 2.0, 4.0, 0.0, 3.0, 1.0, 4.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0, 7.0, 8.0, 9.0, 6.0, 10.0, 11.0, 12.0, 13.0])
pivot: 5.0
✅ Warp partition test passed!
```

### 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p26/p26.mojo:warp_partition_solution}}
```

<div class="solution-explanation">

이 풀이는 여러 Warp 기본 요소 간의 고급 조정을 통해 정교한 병렬 알고리즘을 구현하는 방법을 보여줍니다.

**전체 알고리즘 분석:**

```mojo
if global_i < size:
    current_val = input[global_i]

    # 1단계: Warp 레벨 프레디케이트 생성
    predicate_left = Float32(1.0) if current_val < pivot else Float32(0.0)
    predicate_right = Float32(1.0) if current_val >= pivot else Float32(0.0)

    # 2단계: Warp 레벨 prefix sum으로 Warp 내 위치 계산
    warp_left_pos = prefix_sum[exclusive=True](predicate_left)
    warp_right_pos = prefix_sum[exclusive=True](predicate_right)

    # 3단계: shuffle_xor butterfly reduction으로 왼쪽 총 개수 구하기
    warp_left_total = predicate_left

    # Warp 전체의 합산을 위한 butterfly reduction: 모든 WARP_SIZE에 동적 대응
    offset = WARP_SIZE // 2
    while offset > 0:
        warp_left_total += shuffle_xor(warp_left_total, offset)
        offset //= 2

    # 4단계: 출력 위치에 기록
    if current_val < pivot:
        # 왼쪽 파티션: Warp 레벨 위치 사용
        output[Int(warp_left_pos)] = current_val
    else:
        # 오른쪽 파티션: 왼쪽 총 개수 + 오른쪽 위치로 offset
        output[Int(warp_left_total + warp_right_pos)] = current_val
```

**다단계 실행 추적 (8-Lane 예제, pivot=5, 값 [3,7,1,8,2,9,4,6]):**

```
초기 상태:
  Lane 0: current_val=3 (< 5)  Lane 1: current_val=7 (>= 5)
  Lane 2: current_val=1 (< 5)  Lane 3: current_val=8 (>= 5)
  Lane 4: current_val=2 (< 5)  Lane 5: current_val=9 (>= 5)
  Lane 6: current_val=4 (< 5)  Lane 7: current_val=6 (>= 5)

1단계: 프레디케이트 생성
  Lane 0: predicate_left=1.0, predicate_right=0.0
  Lane 1: predicate_left=0.0, predicate_right=1.0
  Lane 2: predicate_left=1.0, predicate_right=0.0
  Lane 3: predicate_left=0.0, predicate_right=1.0
  Lane 4: predicate_left=1.0, predicate_right=0.0
  Lane 5: predicate_left=0.0, predicate_right=1.0
  Lane 6: predicate_left=1.0, predicate_right=0.0
  Lane 7: predicate_left=0.0, predicate_right=1.0

2단계: 위치 계산을 위한 exclusive prefix sum
  warp_left_pos:  [0, 0, 1, 1, 2, 2, 3, 3]
  warp_right_pos: [0, 0, 0, 1, 1, 2, 2, 3]

3단계: 왼쪽 총 개수를 위한 butterfly reduction
  초기값: [1, 0, 1, 0, 1, 0, 1, 0]
  Reduction 후: 모든 Lane이 warp_left_total = 4를 가짐

4단계: 출력 위치에 기록
  Lane 0: current_val=3 < pivot → output[0] = 3
  Lane 1: current_val=7 >= pivot → output[4+0] = output[4] = 7
  Lane 2: current_val=1 < pivot → output[1] = 1
  Lane 3: current_val=8 >= pivot → output[4+1] = output[5] = 8
  Lane 4: current_val=2 < pivot → output[2] = 2
  Lane 5: current_val=9 >= pivot → output[4+2] = output[6] = 9
  Lane 6: current_val=4 < pivot → output[3] = 4
  Lane 7: current_val=6 >= pivot → output[4+3] = output[7] = 6

최종 결과: [3, 1, 2, 4, 7, 8, 9, 6] (< pivot | >= pivot)
```

**수학적 통찰:** 이중 Warp 기본 요소를 사용한 병렬 파티셔닝을 구현합니다:
\\[\Large \\begin{align}
\\text{left\\_pos}[i] &= \\text{prefix\\_sum}_{\\text{exclusive}}(\\text{predicate\\_left}[i]) \\\\
\\text{right\\_pos}[i] &= \\text{prefix\\_sum}_{\\text{exclusive}}(\\text{predicate\\_right}[i]) \\\\
\\text{left\\_total} &= \\text{butterfly\\_reduce}(\\text{predicate\\_left}) \\\\
\\text{final\\_pos}[i] &= \\begin{cases}
\\text{left\\_pos}[i] & \\text{if } \\text{input}[i] < \\text{pivot} \\\\
\\text{left\\_total} + \\text{right\\_pos}[i] & \\text{if } \\text{input}[i] \\geq \\text{pivot}
\\end{cases}
\\end{align}\\]

**다중 기본 요소 접근 방식이 동작하는 이유:**

1. **프레디케이트 생성**: 각 요소의 파티션 소속을 식별
2. **Exclusive prefix sum**: 각 파티션 내 상대적 위치를 계산
3. **Butterfly reduction**: 파티션 경계 (왼쪽 총 개수)를 산출
4. **조정된 기록**: 로컬 위치와 전역 파티션 구조를 결합

**알고리즘 복잡도:**

- **1단계**: \\(O(1)\\) - 프레디케이트 생성
- **2단계**: \\(O(\\log n)\\) - 하드웨어 가속 prefix sum
- **3단계**: \\(O(\\log n)\\) - `shuffle_xor`을 활용한 butterfly reduction
- **4단계**: \\(O(1)\\) - 조정된 기록
- **전체**: 우수한 상수를 가진 \\(O(\\log n)\\)

**성능 특성:**

- **통신 단계**: \\(2 \\times \\log_2(\\text{WARP\_SIZE})\\) (prefix sum + butterfly reduction)
- **메모리 효율성**: 공유 메모리 제로, 모두 레지스터 기반
- **병렬성**: 알고리즘 전체에서 모든 Lane이 활성 상태
- **확장성**: 모든 `WARP_SIZE` (32, 64 등)에서 동작

**실용적 활용:** 이 패턴의 기반이 되는 분야:

- **Quicksort 파티셔닝**: 병렬 정렬 알고리즘의 핵심 단계
- **Stream compaction**: 데이터 스트림에서 null/무효 요소 제거
- **병렬 필터링**: 복잡한 프레디케이트에 따른 데이터 분리
- **부하 분산**: 연산 요구량에 따른 작업 재분배

</div>
</details>

## 요약

`prefix_sum()` 기본 요소는 복잡한 다단계 알고리즘을 단일 함수 호출로 대체하는 하드웨어 가속 병렬 scan 연산을 가능하게 합니다. 두 가지 문제를 통해 다음을 배웠습니다:

### **핵심 Prefix Sum 패턴**

1. **Inclusive Prefix Sum** (`prefix_sum[exclusive=False]`):
   - 하드웨어 가속 누적 연산
   - 공유 메모리 코드 ~30줄을 단일 함수 호출로 대체
   - 전용 하드웨어 최적화를 동반한 \\(O(\\log n)\\) 복잡도

2. **고급 다중 기본 요소 조정** (`prefix_sum` + `shuffle_xor` 결합):
   - 단일 Warp 내 정교한 병렬 알고리즘
   - 위치 계산을 위한 exclusive scan + 총합을 위한 butterfly reduction
   - 최적의 병렬 효율성을 가진 복잡한 파티셔닝 연산

### **핵심 알고리즘 통찰**

**하드웨어 가속의 이점:**

- `prefix_sum()`이 현대 GPU의 전용 scan 유닛을 활용
- 기존 방식 대비 공유 메모리 오버헤드 제로
- 명시적 barrier 없는 자동 동기화

**다중 기본 요소 조정:**

```mojo
# 1단계: 파티션 소속을 위한 프레디케이트 생성
predicate = 1.0 if condition else 0.0

# 2단계: 로컬 위치를 위한 prefix_sum 사용
local_pos = prefix_sum[exclusive=True](predicate)

# 3단계: 전역 총합을 위한 shuffle_xor 사용
global_total = butterfly_reduce(predicate)

# 4단계: 최종 위치 결정을 위한 결합
final_pos = local_pos + partition_offset
```

**성능 이점:**

- **하드웨어 최적화**: 소프트웨어 구현 대비 전용 scan 유닛
- **메모리 효율성**: 공유 메모리 할당 대비 레지스터 전용 연산
- **확장 가능한 복잡도**: 하드웨어 가속을 동반한 \\(O(\\log n)\\)
- **단일 Warp 최적화**: `WARP_SIZE` 한도 내 알고리즘에 최적

### **실용적 활용**

이 prefix sum 패턴들의 기반이 되는 분야:

- **병렬 scan 연산**: 누적 합, 누적 곱, min/max scan
- **Stream compaction**: 병렬 필터링과 데이터 재배치
- **Quicksort 파티셔닝**: 병렬 정렬 알고리즘의 핵심 빌딩 블록
- **병렬 알고리즘**: 부하 분산, 작업 분배, 데이터 재구조화

`prefix_sum()`과 `shuffle_xor()`의 결합은 현대 GPU Warp 기본 요소가 최소한의 코드 복잡도와 최적의 성능 특성으로 정교한 병렬 알고리즘을 어떻게 구현할 수 있는지를 보여줍니다.
