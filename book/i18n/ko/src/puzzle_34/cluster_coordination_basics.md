<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# 멀티 블록 조정 기초

## 개요

첫 번째 **클러스터 프로그래밍 도전**에 오신 것을 환영합니다! 이 섹션에서는 SM90+ 클러스터 API를 사용한 블록 간 조정의 기본 구성 요소를 소개합니다.

**도전 과제**: **4개의 스레드 블록이 조정**하여 서로 다른 데이터 범위를 처리하고 결과를 공유 출력 배열에 저장하는 멀티 블록 히스토그램 알고리즘을 구현합니다.

**핵심 학습**: [`cluster_arrive()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive) → 처리 → [`cluster_wait()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_wait)라는 필수적인 클러스터 동기화 패턴을 배웁니다. [Puzzle 29의 barrier()](../puzzle_29/barrier.md)에서 배운 동기화 개념을 확장합니다.

## 문제: 멀티 블록 히스토그램 구간 분류

[Puzzle 27](../puzzle_27/puzzle_27.md)과 같은 기존의 단일 블록 알고리즘은 하나의 블록이 가진 스레드 용량(예: 256개 스레드) 내에 들어오는 데이터만 처리할 수 있습니다. [Puzzle 8의 공유 메모리 용량](../puzzle_08/puzzle_08.md)을 초과하는 더 큰 데이터셋의 경우, **여러 블록이 협력**해야 합니다.

**과제**: 4개 블록 각각이 서로 다른 데이터 범위를 처리하고, 고유한 블록 순위로 값을 스케일링하며, [Puzzle 29의 동기화 패턴](../puzzle_29/barrier.md)을 사용하여 다른 블록들과 조정함으로써 모든 블록의 처리가 완료된 후에야 최종 결과를 읽을 수 있도록 하는 히스토그램을 구현하세요.

### 문제 명세

**멀티 블록 데이터 분배:**

- **Block 0**: 요소 0-255를 처리, 1배 스케일링
- **Block 1**: 요소 256-511을 처리, 2배 스케일링
- **Block 2**: 요소 512-767을 처리, 3배 스케일링
- **Block 3**: 요소 768-1023을 처리, 4배 스케일링

**조정 요구사항:**

1. 각 블록은 [`cluster_arrive()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive)를 사용하여 완료를 알려야 합니다
2. 모든 블록은 [`cluster_wait()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_wait)를 사용하여 다른 블록을 기다려야 합니다
3. 최종 출력은 각 블록의 처리된 합계를 4개 요소 배열로 보여줍니다

## 설정

- **문제 크기**: `SIZE = 1024` 요소 (1D 배열)
- **블록 설정**: `TPB = 256` 블록당 스레드 수 `(256, 1)`
- **그리드 설정**: `CLUSTER_SIZE = 4` 클러스터당 블록 수 `(4, 1)`
- **데이터 타입**: `DType.float32`
- **메모리 레이아웃**: 입력 `Layout.row_major(SIZE)`, 출력 `Layout.row_major(CLUSTER_SIZE)`

**스레드 블록 분배:**

- Block 0: 스레드 0-255 → 요소 0-255
- Block 1: 스레드 0-255 → 요소 256-511
- Block 2: 스레드 0-255 → 요소 512-767
- Block 3: 스레드 0-255 → 요소 768-1023

## 완성할 코드

```mojo
{{#include ../../../../../problems/p34/p34.mojo:cluster_coordination_basics}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p34/p34.mojo" class="filename">전체 파일 보기: problems/p34/p34.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### **블록 식별 패턴**

- [`block_rank_in_cluster()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/block_rank_in_cluster)를 사용하여 클러스터 순위(0-3)를 얻습니다
- 그리드 실행에서 안정적인 블록 인덱싱을 위해 `Int(block_idx.x)`를 사용합니다
- 블록 위치에 따라 데이터 처리를 스케일링하여 고유한 결과를 만듭니다

### **공유 메모리 조정**

- `LayoutTensor[dtype, Layout.row_major(tpb), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`으로 공유 메모리를 할당합니다 ([Puzzle 8의 공유 메모리 기초](../puzzle_08/puzzle_08.md) 참고)
- `block_id + 1`로 스케일링하여 블록마다 고유한 스케일링을 적용합니다
- 입력 데이터 접근 시 경계 검사를 사용합니다 ([Puzzle 3의 가드 패턴](../puzzle_03/puzzle_03.md))

### **클러스터 동기화 패턴**

1. **처리**: 각 블록이 자신의 데이터 영역을 처리합니다
2. **신호**: [`cluster_arrive()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive)로 처리 완료를 알립니다
3. **연산**: 블록 내부 연산 (reduction, 집계)
4. **대기**: [`cluster_wait()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_wait)로 모든 블록이 완료될 때까지 대기합니다

### **블록 내부 스레드 조정**

- 클러스터 연산 전에 블록 내부 동기화를 위해 `barrier()`를 사용합니다 ([Puzzle 29의 barrier 개념](../puzzle_29/barrier.md))
- 스레드 0만 최종 블록 결과를 기록해야 합니다 ([블록 프로그래밍](../puzzle_27/block_sum.md)의 단일 writer 패턴)
- 안정적인 인덱싱을 위해 결과를 `output[block_id]`에 저장합니다

</div>
</details>

## 코드 실행

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p34 --coordination
```

  </div>
  <div class="tab-content">

```bash
uv run poe p34 --coordination
```

  </div>
</div>

**예상 출력:**

```txt
Testing Multi-Block Coordination
SIZE: 1024 TPB: 256 CLUSTER_SIZE: 4
Block coordination results:
  Block 0 : 127.5
  Block 1 : 255.0
  Block 2 : 382.5
  Block 3 : 510.0
✅ Multi-block coordination tests passed!
```

**성공 기준:**

- 4개 블록 모두 **0이 아닌 결과**를 생성합니다
- 결과가 **스케일링 패턴**을 보여줍니다: Block 1 > Block 0, Block 2 > Block 1 등
- 경쟁 상태나 조정 실패가 없어야 합니다

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p34/p34.mojo:cluster_coordination_basics_solution}}
```

<div class="solution-explanation">

**클러스터 조정 풀이는 신중하게 설계된 2단계 접근 방식을 통해 기본적인 멀티 블록 동기화 패턴을 보여줍니다:**

## **1단계: 독립적 블록 처리**

**스레드 및 블록 식별:**

```mojo
global_i = block_dim.x * block_idx.x + thread_idx.x  # Global thread index
local_i = thread_idx.x                               # Local thread index within block
my_block_rank = Int(block_rank_in_cluster())         # Cluster rank (0-3)
block_id = Int(block_idx.x)                          # Block index for reliable addressing
```

**공유 메모리 할당 및 데이터 처리:**

- 각 블록이 자체 공유 메모리 작업 공간을 할당합니다: `LayoutTensor[dtype, Layout.row_major(tpb), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`
- **스케일링 전략**: `data_scale = Float32(block_id + 1)`로 각 블록이 다르게 데이터를 처리하도록 합니다
  - Block 0: 1.0배, Block 1: 2.0배, Block 2: 3.0배, Block 3: 4.0배
- **경계 검사**: `if global_i < size:`로 범위 밖 메모리 접근을 방지합니다
- **데이터 처리**: `shared_data[local_i] = input[global_i] * data_scale`로 블록별 입력 데이터를 스케일링합니다

**블록 내부 동기화:**

- `barrier()`는 각 블록 내 모든 스레드가 데이터 로딩을 완료한 후에야 다음 단계로 진행하도록 보장합니다
- 데이터 로딩과 이후의 클러스터 조정 사이의 경쟁 상태를 방지합니다

## **2단계: 클러스터 조정**

**블록 간 신호:**

- [`cluster_arrive()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive)는 이 블록이 로컬 처리 단계를 완료했음을 알립니다
- 클러스터 하드웨어에 완료를 등록하는 **논블로킹** 연산입니다

**로컬 집계 (스레드 0만):**

```mojo
if local_i == 0:
    var block_sum: Float32 = 0.0
    for i in range(tpb):
        block_sum += shared_data[i][0]  # Sum all elements in shared memory
    output[block_id] = block_sum        # Store result at unique block position
```

- 경쟁 상태를 피하기 위해 스레드 0만 합산을 수행합니다
- `output[block_id]`에 결과를 저장하여 각 블록이 고유한 위치에 기록하도록 합니다

**최종 동기화:**

- [`cluster_wait()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_wait)는 클러스터 내 모든 블록이 작업을 완료할 때까지 대기합니다
- 이를 통해 전체 클러스터에 걸쳐 결정론적 완료 순서를 보장합니다

## **핵심 기술 인사이트**

**왜 `my_block_rank` 대신 `block_id`를 사용할까?**

- `block_idx.x`는 안정적인 그리드 실행 인덱싱을 제공합니다 (0, 1, 2, 3)
- [`block_rank_in_cluster()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/block_rank_in_cluster)는 클러스터 설정에 따라 다르게 동작할 수 있습니다
- `block_id`를 사용하면 각 블록이 고유한 데이터 영역과 출력 위치를 확보할 수 있습니다

**메모리 접근 패턴:**

- **글로벌 메모리**: 각 스레드가 `input[global_i]`를 정확히 한 번 읽습니다
- **공유 메모리**: 블록 내부 통신과 집계에 사용됩니다
- **출력 메모리**: 각 블록이 `output[block_id]`에 정확히 한 번 기록합니다

**동기화 계층 구조:**

1. **`barrier()`**: 각 블록 내 스레드를 동기화합니다 (블록 내부)
2. **[`cluster_arrive()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive)**: 다른 블록에 완료를 알립니다 (블록 간, 논블로킹)
3. **[`cluster_wait()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_wait)**: 모든 블록이 완료될 때까지 대기합니다 (블록 간, 블로킹)

**성능 특성:**

- **연산 복잡도**: 블록당 로컬 합산에 O(TPB), 클러스터 조정에 O(1)
- **메모리 대역폭**: 각 입력 요소를 한 번만 읽으며, 블록 간 통신은 최소화
- **확장성**: 패턴이 더 큰 클러스터 크기에도 최소한의 오버헤드로 확장 가능

</div>
</details>

## 패턴 이해하기

클러스터 조정의 핵심 패턴은 단순하지만 강력한 구조를 따릅니다:

1. **1단계**: 각 블록이 할당된 데이터 영역을 독립적으로 처리합니다
2. **신호**: [`cluster_arrive()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive)로 처리 완료를 알립니다
3. **2단계**: 다른 블록의 결과에 의존하는 연산을 안전하게 수행할 수 있습니다
4. **동기화**: [`cluster_wait()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_wait)로 모든 블록이 완료된 후 다음으로 진행합니다

**다음 단계**: 더 고급 조정을 배울 준비가 되셨나요? **[클러스터 전체 집합 연산](./cluster_collective_ops.md)** 으로 이동하여 [Puzzle 27의 `block.sum()` 패턴](../puzzle_27/block_sum.md)을 클러스터 규모로 확장하는 방법을 배워보세요. [Puzzle 24의 Warp 레벨 reduction](../puzzle_24/warp_sum.md)을 기반으로 합니다!
