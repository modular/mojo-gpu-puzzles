<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# 완성 버전

1D LayoutTensor `a`에 대해 prefix sum을 계산하고 결과를 1D LayoutTensor `output`에 저장하는 kernel을 구현하세요.

**참고:** _`a`의 크기가 블록 크기보다 큰 경우, 올바른 결과를 얻으려면 여러 블록 간 동기화가 필요합니다._

## 구성

- 배열 크기: `SIZE_2 = 15`
- 블록당 스레드 수: `TPB = 8`
- 블록 수: 2
- 공유 메모리: 블록당 `TPB`개 원소

참고:

- **다중 블록**: 입력 배열이 하나의 블록보다 클 때는 다단계 접근이 필요
- **블록 레벨 동기화**: 블록 내에서는 `barrier()`로 스레드를 동기화
- **호스트 레벨 동기화**: Mojo의 `DeviceContext`가 kernel 실행 순서를 보장하므로, kernel들은 큐에 넣은 순서대로 실행되고 이전 kernel이 끝나야 다음이 시작됩니다. 호스트에서 결과를 읽기 전에 `ctx.synchronize()`로 모든 GPU 작업 완료를 확인해야 할 수 있습니다.
- **보조 저장소**: 블록 간 통신을 위해 블록 합계를 저장할 추가 공간 사용

## 완성할 코드

멀티 블록 prefix sum을 위해 두 개의 별도 kernel 함수를 완성해야 합니다:

1. **첫 번째 kernel** (`prefix_sum_local_phase`): 각 블록 내에서 로컬 prefix sum을 계산하고 블록 합계를 저장
2. **두 번째 kernel** (`prefix_sum_block_sum_phase`): 이전 블록의 합계를 후속 블록의 원소에 더함

메인 함수가 이 kernel들 사이에 필요한 호스트 측 동기화를 처리합니다.

```mojo
{{#include ../../../../../problems/p14/p14.mojo:prefix_sum_complete}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p14/p14.mojo" class="filename">전체 파일 보기: problems/p14/p14.mojo</a>

이 퍼즐의 핵심은 [barrier](https://docs.modular.com/mojo/std/gpu/sync/sync/barrier/)가 블록 내부의 스레드만 동기화하며, 블록 간 동기화는 하지 않는다는 점을 이해하는 것입니다. 블록 간 동기화를 위해서는 디바이스에서 순차적으로 실행되는 여러 kernel을 큐에 넣어야 합니다:

```mojo
{{#include ../../../../../solutions/p14/p14.mojo:prefix_sum_complete_block_level_sync}}
```

두 kernel이 순차적으로 큐에 들어가지만, `out_tensor`는 두 kernel의 작업이 모두 끝날 때까지 호스트로 전송되지 않는다는 점에 주목하세요. Mojo의 `DeviceContext`가 단일 실행 스트림을 사용하므로, 큐에 넣은 모든 kernel이 순차적으로 실행됩니다. 호스트에서 결과를 읽기 전에 모든 GPU 작업의 완료를 명시적으로 대기하려면 `ctx.synchronize()`를 사용할 수 있습니다.

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. 기본 prefix sum 위에 쌓아 올리기

[🔰 기본 버전](./simple.md)에서 단일 블록 prefix sum 구현 방법을 보여줍니다. 이 접근법을 여러 블록에서 동작하도록 확장해야 합니다:

```
기본 버전 (단일 블록): [0,1,2,3,4,5,6,7] → [0,1,3,6,10,15,21,28]

완성 버전 (두 블록):
Block 0: [0,1,2,3,4,5,6,7] → [0,1,3,6,10,15,21,28]
Block 1: [8,9,10,11,12,13,14] → [8,17,27,38,50,63,77]
```

그런데 두 번째 블록의 값은 어떻게 처리할까요? 첫 번째 블록의 합계를 포함해야 합니다!

### 2. 2단계 접근

기본 prefix sum으로는 블록 간 동기화가 불가능하므로, 작업을 나눕니다:

1. **1단계**: 각 블록이 로컬 prefix sum을 계산 (기본 버전과 동일)
2. **2단계**: 각 블록이 이전 블록의 합계를 반영

주의: `barrier()`는 하나의 블록 내에서만 스레드를 동기화합니다. 단계 간에는 호스트 레벨 동기화가 필요합니다.

### 3. 확장 메모리 전략

블록끼리 직접 통신할 수 없으므로, 블록 합계를 저장할 곳이 필요합니다:

- 출력 버퍼 끝에 추가 메모리를 할당
- 각 블록의 마지막 스레드가 최종 합계를 이 추가 공간에 저장
- 후속 블록이 이 합계를 읽어서 자기 원소에 더함

### 4. 주요 구현 포인트

- **레이아웃 차이**: 입력과 출력의 shape이 다를 수 있음
- **경계 처리**: 항상 `global_i < size`로 배열 범위 확인
- **스레드 역할 분담**: 특정 스레드(예: 마지막 스레드)만 블록 합계를 저장
- **두 kernel 간 동기화**: 두 번째 kernel은 반드시 첫 번째 kernel이 완료된 후에 실행되어야 함

### 5. 디버깅 전략

문제가 발생하면, 1단계 이후의 중간 상태를 시각화해 보세요:

```
1단계 이후: [0,1,3,6,10,15,21,28, 8,17,27,38,50,63,77, ???,???]
```

여기서 `???`에는 2단계에서 사용될 블록 합계가 들어가야 합니다.

중간 결과를 확인하려면 먼저 디바이스의 작업 완료를 명시적으로 보장해야 한다는 점을 기억하세요.

</div>
</details>

## 코드 실행

솔루션을 테스트하려면 터미널에서 다음 명령어를 실행하세요:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p14 --complete
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p14 --complete
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p14 --complete
```

  </div>
  <div class="tab-content">

```bash
uv run poe p14 --complete
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력은 다음과 같습니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0, 55.0, 66.0, 78.0, 91.0, 105.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p14/p14.mojo:prefix_sum_complete_solution}}
```

<div class="solution-explanation">

이 솔루션은 여러 스레드 블록에 걸치는 배열을 처리하기 위해 2개의 kernel을 사용하는 멀티 블록 prefix sum을 구현합니다. 각 부분을 자세히 살펴보겠습니다:

## 블록 간 통신의 과제

GPU 프로그래밍의 근본적인 제약은 `barrier()`를 사용한 스레드 동기화가 블록 내부에서만 가능하다는 점입니다. 데이터가 여러 블록에 걸쳐 있을 때 다음과 같은 과제에 직면합니다: **블록이 부분 결과를 다른 블록에 어떻게 전달할 수 있을까?**

### 메모리 레이아웃 시각화

테스트 케이스 `SIZE_2 = 15`, `TPB = 8`의 경우:

```
Input array:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

Block 0 처리: [0, 1, 2, 3, 4, 5, 6, 7]
Block 1 처리: [8, 9, 10, 11, 12, 13, 14] (유효 원소 7개)
```

블록 합계를 위한 공간을 포함하도록 출력 버퍼를 확장합니다:

```
확장 버퍼: [데이터 값 (15개)] + [블록 합계 (2개)]
           [0...14] + [block0_sum, block1_sum]
```

이 확장 버퍼의 크기: `EXTENDED_SIZE = SIZE_2 + num_blocks = 15 + 2 = 17`

## 1단계 kernel: 로컬 prefix sum

### 로컬 단계에서의 경쟁 상태 방지

로컬 단계는 기본 버전과 동일한 명시적 동기화 패턴을 사용하여 읽기-쓰기 충돌을 방지합니다:

- **읽기 단계**: 모든 스레드가 먼저 필요한 값을 로컬 변수 `current_val`에 읽어둠
- **동기화**: `barrier()`로 모든 읽기가 완료된 후에야 쓰기가 시작되도록 보장
- **쓰기 단계**: 모든 스레드가 계산된 값을 안전하게 공유 메모리에 기록

이를 통해 병렬 reduction 중 여러 스레드가 동시에 같은 공유 메모리 위치에 접근할 때 발생할 수 있는 경쟁 상태를 방지합니다.

### Block 0 단계별 실행

1. **공유 메모리에 값 로드**:

   ```
   shared = [0, 1, 2, 3, 4, 5, 6, 7]
   ```

2. **병렬 reduction 반복** (\\(\log_2(TPB) = 3\\)회 반복):

   **반복 1** (offset=1):

   **읽기 단계**: 각 활성 스레드가 필요한 값을 읽음:

   ```
   T₁ reads shared[0] = 0    T₅ reads shared[4] = 4
   T₂ reads shared[1] = 1    T₆ reads shared[5] = 5
   T₃ reads shared[2] = 2    T₇ reads shared[6] = 6
   T₄ reads shared[3] = 3
   ```

   **동기화**: `barrier()`로 모든 읽기 완료를 보장

   **쓰기 단계**: 각 스레드가 읽은 값을 더함:

   ```
   shared[0] = 0              (변경 없음)
   shared[1] = 1 + 0 = 1
   shared[2] = 2 + 1 = 3
   shared[3] = 3 + 2 = 5
   shared[4] = 4 + 3 = 7
   shared[5] = 5 + 4 = 9
   shared[6] = 6 + 5 = 11
   shared[7] = 7 + 6 = 13
   ```

   barrier 후: `shared = [0, 1, 3, 5, 7, 9, 11, 13]`

   **반복 2** (offset=2):

   **읽기 단계**: 각 활성 스레드가 필요한 값을 읽음:

   ```
   T₂ reads shared[0] = 0    T₅ reads shared[3] = 5
   T₃ reads shared[1] = 1    T₆ reads shared[4] = 7
   T₄ reads shared[2] = 3    T₇ reads shared[5] = 9
   ```

   **동기화**: `barrier()`로 모든 읽기 완료를 보장

   **쓰기 단계**: 각 스레드가 읽은 값을 더함:

   ```
   shared[0] = 0              (변경 없음)
   shared[1] = 1              (변경 없음)
   shared[2] = 3 + 0 = 3      (변경 없음)
   shared[3] = 5 + 1 = 6
   shared[4] = 7 + 3 = 10
   shared[5] = 9 + 5 = 14
   shared[6] = 11 + 7 = 18
   shared[7] = 13 + 9 = 22
   ```

   barrier 후: `shared = [0, 1, 3, 6, 10, 14, 18, 22]`

   **반복 3** (offset=4):

   **읽기 단계**: 각 활성 스레드가 필요한 값을 읽음:

   ```
   T₄ reads shared[0] = 0    T₆ reads shared[2] = 3
   T₅ reads shared[1] = 1    T₇ reads shared[3] = 6
   ```

   **동기화**: `barrier()`로 모든 읽기 완료를 보장

   **쓰기 단계**: 각 스레드가 읽은 값을 더함:

   ```
   shared[0] = 0              (변경 없음)
   shared[1] = 1              (변경 없음)
   shared[2] = 3              (변경 없음)
   shared[3] = 6              (변경 없음)
   shared[4] = 10 + 0 = 10    (변경 없음)
   shared[5] = 14 + 1 = 15
   shared[6] = 18 + 3 = 21
   shared[7] = 22 + 6 = 28
   ```

   barrier 후: `shared = [0, 1, 3, 6, 10, 15, 21, 28]`

3. **로컬 결과를 글로벌 메모리에 기록**:

   ```
   output[0...7] = [0, 1, 3, 6, 10, 15, 21, 28]
   ```

4. **블록 합계를 보조 공간에 저장** (마지막 스레드만):

   ```
   output[15] = 28  // 위치: size + block_idx.x = 15 + 0
   ```

### Block 1 단계별 실행

1. **공유 메모리에 값 로드**:

   ```
   shared = [8, 9, 10, 11, 12, 13, 14, 미초기화]
   ```

   참고: Thread 7은 `global_i = 15 >= SIZE_2`이므로 아무것도 로드하지 않아 `shared[7]`이 미초기화 상태로 남습니다. Thread 7은 최종 출력에 참여하지 않으므로 안전합니다.

2. **병렬 reduction 반복** (\\(\log_2(TPB) = 3\\)회 반복):

   실제로 연산에 참여하는 것은 처음 7개 스레드뿐입니다. 세 번의 반복을 거치면:

   ```
   shared = [8, 17, 27, 38, 50, 63, 77, 미초기화]
   ```

3. **로컬 결과를 글로벌 메모리에 기록**:

   ```
   output[8...14] = [8, 17, 27, 38, 50, 63, 77]  // 유효 출력 7개만
   ```

4. **블록 합계를 보조 공간에 저장** (블록의 마지막 스레드만):

   ```
   output[16] = shared[7]  // Thread 7 (TPB-1)이 shared[7]의 값을 저장
   ```

   참고: Thread 7은 유효한 입력을 로드하지 않았지만, 블록 내 prefix sum 연산에는 그대로 참여합니다. `shared[7]`은 병렬 reduction을 거치며 갱신되지만, 미초기화 상태에서 시작했기 때문에 최종 값을 예측할 수 없습니다. 다만 Block 1이 마지막 블록이므로 이 블록 합계는 2단계에서 사용되지 않아 정확성에는 영향이 없습니다.

1단계 이후 출력 버퍼의 내용:

```
[0, 1, 3, 6, 10, 15, 21, 28, 8, 17, 27, 38, 50, 63, 77, 28, ???]
                                                        ^   ^
                                                블록 합계가 여기에 저장됨
```

참고: 마지막 블록 합계 (???) 는 미초기화 메모리에 기반하므로 예측할 수 없지만, 최종 결과에는 영향을 주지 않습니다.

## 호스트-디바이스 동기화: 실제로 필요한 시점

두 kernel 단계는 **명시적 동기화 없이** 순차적으로 실행됩니다:

```mojo
# 1단계: 로컬 prefix sum
ctx.enqueue_function[prefix_sum_local_phase[...], prefix_sum_local_phase[...]](...)

# 2단계: 블록 합계 더하기 (자동으로 1단계 완료를 대기)
ctx.enqueue_function[prefix_sum_block_sum_phase[...], prefix_sum_block_sum_phase[...]](...)
```

**핵심 통찰**: Mojo의 `DeviceContext`는 단일 실행 스트림(NVIDIA GPU에서는 CUDA 스트림, AMD ROCm GPU에서는 HIP 스트림)을 사용하므로, 큐에 넣은 kernel이 정확히 넣은 순서대로 실행됨을 보장합니다. kernel 간에 명시적 동기화가 필요 없습니다.

**`ctx.synchronize()`가 필요한 시점**:

```mojo
# 두 kernel 완료 후, 호스트에서 결과를 읽기 전
ctx.synchronize()  # 호스트가 GPU 완료를 대기

with out.map_to_host() as out_host:  # 이제 GPU 결과를 안전하게 읽을 수 있음
    print("out:", out_host)
```

`ctx.synchronize()` 호출의 역할:

- **호스트-디바이스 동기화**: 결과에 접근하기 전에 호스트가 모든 GPU 작업의 완료를 대기하도록 보장
- **메모리 안전성**: 연산이 끝나기 전에 GPU 메모리를 읽는 것을 방지

**실행 모델**: 블록 내부의 스레드를 동기화하는 `barrier()`와 달리, kernel 실행 순서는 Mojo의 단일 스트림 실행 모델에서 보장되며, `ctx.synchronize()`는 호스트-디바이스 간 조율을 담당합니다.

## 2단계 kernel: 블록 합계 더하기

1. **Block 0**: 변경 불필요 (이미 올바른 상태).

2. **Block 1**: 각 스레드가 Block 0의 합계를 자기 원소에 더함:

   ```
   prev_block_sum = output[size + block_idx.x - 1] = output[15] = 28
   output[global_i] += prev_block_sum
   ```

   Block 1의 값이 변환됩니다:

   ```
   Before: [8, 17, 27, 38, 50, 63, 77]
   After:  [36, 45, 55, 66, 78, 91, 105]
   ```

## 성능 및 최적화 고려 사항

### 주요 구현 상세

**로컬 단계 동기화 패턴**: 블록 내 각 반복은 엄격한 읽기 → 동기화 → 쓰기 패턴을 따릅니다:

1. `var current_val: out.element_type = 0` - 로컬 변수 초기화
2. `current_val = shared[local_i - offset]` - 읽기 단계 (조건 충족 시)
3. `barrier()` - 경쟁 상태 방지를 위한 명시적 동기화
4. `shared[local_i] += current_val` - 쓰기 단계 (조건 충족 시)
5. `barrier()` - 다음 반복 전 동기화

**블록 간 동기화**: 이 알고리즘은 두 수준의 동기화를 사용합니다:

- **블록 내부**: 로컬 prefix sum 연산 중 `barrier()`로 각 블록 내 스레드를 동기화
- **블록 간**: `DeviceContext`가 큐에 넣은 kernel을 순차 실행하여 1단계가 2단계 전에 완료되도록 보장. 결과를 읽기 전에 호스트-디바이스 동기화가 필요하면 `ctx.synchronize()`를 사용합니다.

**경쟁 상태 방지**: 로컬 단계에서 읽기와 쓰기를 명시적으로 분리하여, 병렬 reduction 중 여러 스레드가 같은 공유 메모리 위치에 동시에 접근할 때 생길 수 있는 경쟁 상태를 방지합니다.

1. **작업 효율성**: 이 구현의 작업 복잡도는 \\(O(n \log n)\\)이며, 순차 알고리즘은 \\(O(n)\\)입니다. 병렬 알고리즘에서 전형적인 공간-시간 트레이드오프입니다.

2. **메모리 오버헤드**: 블록 합계를 위한 추가 공간은 최소입니다 (블록당 원소 하나).

이 2개 kernel 접근 방식은 블록 간 통신이 필요한 GPU 알고리즘의 기본 패턴입니다. 기수 정렬, 히스토그램 계산, reduction 연산 등 다른 병렬 알고리즘에도 동일한 전략을 적용할 수 있습니다.
</div>
</details>
