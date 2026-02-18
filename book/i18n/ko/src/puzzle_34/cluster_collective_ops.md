<!-- i18n-source-commit: 43fce1182f8029e7edc50157aed0e6ebb8129d42 -->

# ☸️ 클러스터 전체 집합 연산

## 개요

이전 섹션의 기본 클러스터 조정을 바탕으로, 이 도전에서는 **클러스터 전체 집합 연산**을 구현하는 방법을 배웁니다 - [Puzzle 27](../puzzle_27/block_sum.md)에서 익힌 [`block.sum`](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum) 패턴을 **여러 스레드 블록**에 걸쳐 확장합니다.

**도전 과제**: 4개의 조정된 블록에 걸쳐 1024개 요소를 처리하고, 각 블록의 개별 reduction을 하나의 전역 결과로 합치는 클러스터 전체 reduction을 구현합니다.

**핵심 학습**: 전체 클러스터 조정을 위한 [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync)와 효율적인 최종 reduction을 위한 [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)를 배웁니다.

## 문제: 대규모 전역 합산

단일 블록은 ([Puzzle 27](../puzzle_27/puzzle_27.md)에서 배웠듯이) 스레드 수와 [Puzzle 8의 공유 메모리 용량](../puzzle_08/puzzle_08.md)에 의해 제한됩니다. [단일 블록 reduction](../puzzle_27/block_sum.md)을 넘어서는 **대규모 데이터셋**의 전역 통계(평균, 분산, 합계)를 구하려면 **클러스터 전체 집합 연산**이 필요합니다.

**과제**: 다음과 같은 클러스터 전체 합산 reduction을 구현하세요:

1. 각 블록이 로컬 reduction을 수행합니다 ([Puzzle 27의 `block.sum()`](../puzzle_27/block_sum.md)과 유사)
2. [Puzzle 29의 동기화](../puzzle_29/barrier.md)를 사용하여 블록들이 부분 결과를 합칩니다
3. 선출된 하나의 스레드가 [Warp 선출 패턴](../puzzle_24/warp_sum.md)을 사용하여 최종 전역 합계를 계산합니다

### 문제 명세

**알고리즘 흐름:**

**1단계 - 로컬 Reduction (각 블록 내부):**
\\[R_i = \sum_{j=0}^{TPB-1} input[i \times TPB + j] \quad \text{for block } i\\]

**2단계 - 전역 집계 (클러스터 전체):**
\\[\text{Global Sum} = \sum_{i=0}^{\text{CLUSTER_SIZE}-1} R_i\\]

**조정 요구사항:**

1. **로컬 reduction**: 각 블록이 트리 reduction으로 부분 합을 계산합니다
2. **클러스터 동기화**: [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync)로 모든 부분 결과가 준비되었는지 보장합니다
3. **최종 집계**: 선출된 하나의 스레드가 모든 부분 결과를 합칩니다

## 설정

- **문제 크기**: `SIZE = 1024` 요소
- **블록 설정**: `TPB = 256` 블록당 스레드 수 `(256, 1)`
- **그리드 설정**: `CLUSTER_SIZE = 4` 클러스터당 블록 수 `(4, 1)`
- **데이터 타입**: `DType.float32`
- **메모리 레이아웃**: 입력 `Layout.row_major(SIZE)`, 출력 `Layout.row_major(1)`
- **임시 저장소**: 부분 결과를 위한 `Layout.row_major(CLUSTER_SIZE)`

**예상 결과**: 수열 `0, 0.01, 0.02, ..., 10.23`의 합 = **523,776**

## 완성할 코드

```mojo
{{#include ../../../../../problems/p34/p34.mojo:cluster_collective_operations}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p34/p34.mojo" class="filename">View full file: problems/p34/p34.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### **로컬 reduction 패턴**

- [Puzzle 27의 block sum에서 사용한 트리 reduction 패턴](../puzzle_27/block_sum.md)을 활용합니다
- stride = `tpb // 2`로 시작하여 매 반복마다 절반으로 줄입니다 (고전적인 [Puzzle 12의 reduction](../puzzle_12/puzzle_12.md))
- 각 단계에서 `local_i < stride`인 스레드만 참여합니다
- reduction 단계 사이에 `barrier()`를 사용합니다 ([Puzzle 29의 barrier 개념](../puzzle_29/barrier.md))

### **클러스터 조정 전략**

- 안정적인 인덱싱을 위해 부분 결과를 `temp_storage[block_id]`에 저장합니다
- 전체 클러스터 동기화를 위해 [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync)를 사용합니다 (arrive/wait보다 강력)
- 최종 전역 집계는 하나의 스레드만 수행해야 합니다

### **효율적인 선출 패턴**

- 첫 번째 블록(`my_block_rank == 0`) 내에서 [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)를 사용합니다 ([Warp 프로그래밍](../puzzle_24/warp_sum.md)의 패턴)
- 중복 연산을 피하기 위해 하나의 스레드만 최종 합산을 수행하도록 보장합니다
- 선출된 스레드가 `temp_storage`에서 모든 부분 결과를 읽습니다 ([Puzzle 8의 공유 메모리 접근](../puzzle_08/puzzle_08.md)과 유사)

### **메모리 접근 패턴**

- 각 스레드가 경계 검사와 함께 `input[global_i]`를 읽습니다 ([Puzzle 3의 guard](../puzzle_03/puzzle_03.md))
- 블록 내부 reduction을 위해 [공유 메모리](../puzzle_08/puzzle_08.md)에 중간 결과를 저장합니다
- 블록 간 통신을 위해 부분 결과를 `temp_storage[block_id]`에 저장합니다
- 최종 결과는 `output[0]`에 기록합니다 ([블록 조정](../puzzle_27/block_sum.md)의 단일 writer 패턴)

</div>
</details>

## 클러스터 API 참조

**[`gpu.primitives.cluster`](https://docs.modular.com/mojo/stdlib/gpu/primitives/cluster/) 모듈:**

- **[`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/primitives/cluster/cluster_sync)**: 전체 클러스터 동기화 - arrive/wait 패턴보다 강력
- **[`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/primitives/cluster/elect_one_sync)**: 효율적인 조정을 위해 Warp 내에서 단일 스레드를 선출
- **[`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/primitives/cluster/block_rank_in_cluster)**: 클러스터 내 고유한 블록 식별자를 반환

## 트리 reduction 패턴

[Puzzle 27의 전통적인 내적](../puzzle_27/puzzle_27.md)에서 배운 **트리 reduction 패턴**을 떠올려 보세요:

```txt
Stride 128: [T0] += [T128], [T1] += [T129], [T2] += [T130], ...
Stride 64:  [T0] += [T64],  [T1] += [T65],  [T2] += [T66],  ...
Stride 32:  [T0] += [T32],  [T1] += [T33],  [T2] += [T34],  ...
Stride 16:  [T0] += [T16],  [T1] += [T17],  [T2] += [T18],  ...
...
Stride 1:   [T0] += [T1] → Final result at T0
```

**이제 이 패턴을 클러스터 규모로 확장합니다** - 각 블록이 하나의 부분 결과를 생성한 뒤, 블록 간에 결합합니다.

## 코드 실행

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p34 --reduction
```

  </div>
  <div class="tab-content">

```bash
uv run poe p34 --reduction
```

  </div>
</div>

**예상 출력:**

```txt
Testing Cluster-Wide Reduction
SIZE: 1024 TPB: 256 CLUSTER_SIZE: 4
Expected sum: 523776.0
Cluster reduction result: 523776.0
Expected: 523776.0
Error: 0.0
✅ Passed: Cluster reduction accuracy test
✅ Cluster-wide collective operations tests passed!
```

**성공 기준:**

- **완벽한 정확도**: 결과가 예상 합계(523,776)와 정확히 일치합니다
- **클러스터 조정**: 4개 블록 모두가 부분 합에 기여합니다
- **효율적인 최종 reduction**: 선출된 단일 스레드가 최종 결과를 계산합니다

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p34/p34.mojo:cluster_collective_operations_solution}}
```

<div class="solution-explanation">

**클러스터 집합 연산 풀이는 분산 컴퓨팅의 고전적인 패턴을 보여줍니다: 로컬 reduction → 전역 조정 → 최종 집계:**

## **1단계: 로컬 블록 reduction (전통적 트리 reduction)**

**데이터 로딩 및 초기화:**

```mojo
var my_value: Float32 = 0.0
if global_i < size:
    my_value = input[global_i][0]  # Load with bounds checking
shared_mem[local_i] = my_value     # Store in shared memory
barrier()                          # Ensure all threads complete loading
```

**트리 reduction 알고리즘:**

```mojo
var stride = tpb // 2  # Start with half the threads (128)
while stride > 0:
    if local_i < stride and local_i + stride < tpb:
        shared_mem[local_i] += shared_mem[local_i + stride]
    barrier()          # Synchronize after each reduction step
    stride = stride // 2
```

**트리 reduction 시각화 (TPB=256):**

```txt
Step 1: stride=128  [T0]+=T128, [T1]+=T129, ..., [T127]+=T255
Step 2: stride=64   [T0]+=T64,  [T1]+=T65,  ..., [T63]+=T127
Step 3: stride=32   [T0]+=T32,  [T1]+=T33,  ..., [T31]+=T63
Step 4: stride=16   [T0]+=T16,  [T1]+=T17,  ..., [T15]+=T31
Step 5: stride=8    [T0]+=T8,   [T1]+=T9,   ..., [T7]+=T15
Step 6: stride=4    [T0]+=T4,   [T1]+=T5,   [T2]+=T6,  [T3]+=T7
Step 7: stride=2    [T0]+=T2,   [T1]+=T3
Step 8: stride=1    [T0]+=T1    → Final result at shared_mem[0]
```

**부분 결과 저장:**

- 스레드 0만 기록합니다: `temp_storage[block_id] = shared_mem[0]`
- 각 블록이 자신의 합계를 `temp_storage[0]`, `temp_storage[1]`, `temp_storage[2]`, `temp_storage[3]`에 저장합니다

## **2단계: 클러스터 동기화**

**전체 클러스터 barrier:**

- [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync)는 [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)/[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)보다 **더 강력한 보장**을 제공합니다
- 어떤 블록이든 다음으로 진행하기 전에 **모든 블록이 로컬 reduction을 완료**하도록 보장합니다
- 클러스터 내 모든 블록에 걸친 하드웨어 가속 동기화입니다

## **3단계: 최종 전역 집계**

**효율적인 스레드 선출:**

```mojo
if elect_one_sync() and my_block_rank == 0:
    var total: Float32 = 0.0
    for i in range(CLUSTER_SIZE):
        total += temp_storage[i][0]  # Sum: temp[0] + temp[1] + temp[2] + temp[3]
    output[0] = total
```

**왜 이 선출 전략을 사용할까?**

- **[`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)**: Warp당 정확히 하나의 스레드를 선택하는 하드웨어 기본 요소입니다
- **`my_block_rank == 0`**: 단일 writer를 보장하기 위해 첫 번째 블록에서만 선출합니다
- **결과**: 전체 클러스터에서 단 하나의 스레드만 최종 합산을 수행합니다
- **효율성**: 1024개 전체 스레드에 걸친 중복 연산을 피합니다

## **핵심 기술 인사이트**

**3단계 reduction 계층 구조:**

1. **스레드 → Warp**: 개별 스레드가 Warp 레벨 부분 합에 기여합니다
2. **Warp → 블록**: 트리 reduction이 Warp들을 하나의 블록 결과로 합칩니다 (256 → 1)
3. **블록 → 클러스터**: 단순 루프가 블록 결과를 최종 합계로 합칩니다 (4 → 1)

**메모리 접근 패턴:**

- **입력**: 각 요소를 정확히 한 번 읽습니다 (`input[global_i]`)
- **공유 메모리**: 블록 내부 트리 reduction을 위한 고속 작업 공간
- **임시 저장소**: 저비용 블록 간 통신 (4개 값만)
- **출력**: 단일 전역 결과를 한 번 기록

**동기화 보장:**

- **`barrier()`**: 블록 내 모든 스레드가 각 트리 reduction 단계를 완료하도록 보장합니다
- **[`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync)**: **전역 barrier** - 모든 블록이 동일한 실행 지점에 도달합니다
- **단일 writer**: 선출을 통해 최종 출력에 대한 경쟁 상태를 방지합니다

**알고리즘 복잡도 분석:**

- **트리 reduction**: O(log₂ TPB) = O(log₂ 256) = 블록당 8단계
- **클러스터 조정**: O(1) 동기화 오버헤드
- **최종 집계**: O(CLUSTER_SIZE) = O(4) 단순 덧셈
- **전체**: 블록 내부는 로그, 블록 간은 선형

**확장성 특성:**

- **블록 레벨**: 로그 복잡도로 수천 개의 스레드까지 확장 가능
- **클러스터 레벨**: 선형 복잡도로 수십 개의 블록까지 확장 가능
- **메모리**: 임시 저장소 요구량이 클러스터 크기에 비례하여 선형 증가
- **통신**: 최소한의 블록 간 데이터 이동 (블록당 하나의 값)

</div>
</details>

## 집합 연산 패턴 이해하기

이 퍼즐은 분산 컴퓨팅에서 사용되는 고전적인 **2단계 reduction 패턴**을 보여줍니다:

1. **로컬 집계**: 각 처리 단위(블록)가 자신의 데이터 영역을 reduction합니다
2. **전역 조정**: 처리 단위들이 동기화하고 결과를 교환합니다
3. **최종 reduction**: 선출된 하나의 단위가 모든 부분 결과를 합칩니다

**단일 블록 방식과의 비교:**

- **기존 `block.sum()`**: 최대 256개 스레드 내에서만 동작합니다
- **클러스터 집합 연산**: 여러 블록에 걸쳐 1000개 이상의 스레드로 확장됩니다
- **동일한 정확도**: 둘 다 동일한 수학적 결과를 생성합니다
- **다른 규모**: 클러스터 방식이 더 큰 데이터셋을 처리합니다

**성능 이점**:

- **더 큰 데이터셋**: 단일 블록 용량을 초과하는 배열을 처리합니다
- **더 나은 활용률**: 더 많은 GPU 연산 유닛을 동시에 사용합니다
- **확장 가능한 패턴**: 복잡한 다단계 알고리즘의 기반이 됩니다

**다음 단계**: 최종 도전을 할 준비가 되셨나요? **[고급 클러스터 알고리즘](./advanced_cluster_patterns.md)** 으로 이동하여 [Warp 프로그래밍](../puzzle_24/warp_sum.md)+[블록 조정](../puzzle_27/block_sum.md)+클러스터 동기화를 결합한 계층적 패턴을 배워보세요. [성능 최적화 기법](../puzzle_30/profile_kernels.md)을 기반으로 합니다!
