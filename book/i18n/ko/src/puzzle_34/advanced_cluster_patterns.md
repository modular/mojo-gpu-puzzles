<!-- i18n-source-commit: 43fce1182f8029e7edc50157aed0e6ebb8129d42 -->

# 🧠 고급 클러스터 알고리즘

## 개요

이 마지막 도전에서는 [Warp 레벨 (Puzzle 24-26)](../puzzle_24/puzzle_24.md), [블록 레벨 (Puzzle 27)](../puzzle_27/puzzle_27.md), 클러스터 조정에 이르기까지 **GPU 프로그래밍 계층 구조의 모든 레벨**을 결합하여 GPU 활용률을 극대화하는 정교한 다단계 알고리즘을 구현합니다.

**도전 과제**: **Warp 레벨 최적화** (`elect_one_sync()`), **블록 레벨 집계**, **클러스터 레벨 조정**을 하나의 통합된 패턴으로 사용하는 계층적 클러스터 알고리즘을 구현합니다.

**핵심 학습**: 고급 연산 워크로드에서 사용되는 프로덕션 수준의 조정 패턴과 함께 완전한 GPU 프로그래밍 스택을 배웁니다.

## 문제: 다단계 데이터 처리 파이프라인

실제 GPU 알고리즘은 GPU 계층 구조의 서로 다른 레벨([Puzzle 24의 Warp](../puzzle_24/warp_simt.md), [Puzzle 27의 블록](../puzzle_27/block_sum.md), 클러스터)이 조정된 연산 파이프라인에서 각각 전문화된 역할을 수행하는 **계층적 조정**을 필요로 하는 경우가 많으며, 이는 [Puzzle 29의 다단계 처리](../puzzle_29/barrier.md)를 확장합니다.

**과제**: 다음과 같은 다단계 알고리즘을 구현하세요:

1. **[Warp 레벨](../puzzle_24/warp_sum.md)**: 효율적인 Warp 내부 조정을 위해 [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)를 사용합니다 ([SIMT 실행](../puzzle_24/warp_simt.md))
2. **[블록 레벨](../puzzle_27/block_sum.md)**: [공유 메모리 조정](../puzzle_08/puzzle_08.md)을 사용하여 Warp 결과를 집계합니다
3. **클러스터 레벨**: [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) / [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait) [Puzzle 29의 단계적 동기화](../puzzle_29/barrier.md)를 사용하여 블록 간 조정을 수행합니다

### 알고리즘 명세

**다단계 처리 파이프라인:**

1. **1단계 ([Warp 레벨](../puzzle_24/puzzle_24.md))**: 각 Warp가 하나의 스레드를 선출하여 32개의 연속 요소를 합산합니다
2. **2단계 ([블록 레벨](../puzzle_27/puzzle_27.md))**: 각 블록 내의 모든 Warp 합계를 집계합니다
3. **3단계 (클러스터 레벨)**: [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive) / [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)로 블록 간 조정을 수행합니다

**입력**: 테스트를 위한 `(i % 50) * 0.02` 패턴의 1024개 float 값
**출력**: 계층적 처리 효과를 보여주는 4개 블록 결과

## 설정

- **문제 크기**: `SIZE = 1024` 요소
- **블록 설정**: `TPB = 256` 블록당 스레드 수 `(256, 1)`
- **그리드 설정**: `CLUSTER_SIZE = 4` 블록 `(4, 1)`
- **Warp 크기**: `WARP_SIZE = 32` Warp당 스레드 수 (NVIDIA 표준)
- **블록당 Warp 수**: `TPB / WARP_SIZE = 8` Warp
- **데이터 타입**: `DType.float32`
- **메모리 레이아웃**: 입력 `Layout.row_major(SIZE)`, 출력 `Layout.row_major(CLUSTER_SIZE)`

**처리 분배:**

- **Block 0**: 256 스레드 → 8 Warp → 요소 0-255
- **Block 1**: 256 스레드 → 8 Warp → 요소 256-511
- **Block 2**: 256 스레드 → 8 Warp → 요소 512-767
- **Block 3**: 256 스레드 → 8 Warp → 요소 768-1023

## 완성할 코드

```mojo
{{#include ../../../../../problems/p34/p34.mojo:advanced_cluster_patterns}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p34/p34.mojo" class="filename">View full file: problems/p34/p34.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### **Warp 레벨 최적화 패턴**

- [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)를 사용하여 Warp당 하나의 스레드를 연산용으로 선출합니다 ([Warp 프로그래밍 기초](../puzzle_24/warp_sum.md))
- 선출된 스레드가 32개의 연속 요소를 처리해야 합니다 ([SIMT 실행](../puzzle_24/warp_simt.md) 활용)
- `(local_i // 32) * 32`로 Warp 시작점을 계산하여 Warp 경계를 찾습니다 ([Warp 개념](../puzzle_24/puzzle_24.md)의 Lane 인덱싱)
- Warp 결과를 [선출된 스레드 위치의 공유 메모리](../puzzle_08/puzzle_08.md)에 저장합니다

### **블록 레벨 집계 전략**

- Warp 처리 후 모든 Warp 결과를 집계합니다 ([Puzzle 27의 블록 조정](../puzzle_27/block_sum.md) 확장)
- 선출된 위치에서 읽습니다: 인덱스 0, 32, 64, 96, 128, 160, 192, 224
- `for i in range(0, tpb, 32)` 루프로 Warp 리더를 순회합니다 ([reduction 알고리즘](../puzzle_12/puzzle_12.md)의 패턴)
- 스레드 0만 최종 블록 합계를 계산해야 합니다 ([barrier 조정](../puzzle_29/barrier.md)의 단일 writer 패턴)

### **클러스터 조정 흐름**

1. **처리**: 각 블록이 계층적 Warp 최적화로 데이터를 처리합니다
2. **신호**: [`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)로 로컬 처리 완료를 알립니다
3. **저장**: 스레드 0이 블록 결과를 출력에 기록합니다
4. **대기**: [`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)로 모든 블록이 완료될 때까지 대기합니다

### **데이터 스케일링 및 경계 검사**

- `Float32(block_id + 1)`로 입력을 스케일링하여 블록별 고유 패턴을 만듭니다
- 입력을 읽기 전에 항상 `global_i < size`를 검사합니다 ([Puzzle 3의 guard](../puzzle_03/puzzle_03.md))
- 블록 내 처리 단계 사이에 `barrier()`를 사용합니다 ([동기화 패턴](../puzzle_29/barrier.md))
- 루프에서 Warp 경계 조건을 주의 깊게 처리합니다 ([Warp 프로그래밍](../puzzle_24/warp_simt.md)의 고려사항)

</div>
</details>

## 고급 클러스터 API

**[`gpu.primitives.cluster`](https://docs.modular.com/mojo/stdlib/gpu/cluster/) 모듈:**

- **[`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)**: 효율적인 연산을 위한 Warp 레벨 스레드 선출
- **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**: 단계적 클러스터 조정을 위한 완료 신호
- **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**: 모든 블록이 동기화 지점에 도달할 때까지 대기
- **[`block_rank_in_cluster()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/block_rank_in_cluster)**: 클러스터 내 고유한 블록 식별자 반환

## 계층적 조정 패턴

이 퍼즐은 **3단계 조정 계층 구조**를 보여줍니다:

### **레벨 1: Warp 조정** ([Puzzle 24](../puzzle_24/puzzle_24.md))

```txt
Warp (32 threads) → elect_one_sync() → 1 elected thread → processes 32 elements
```

### **레벨 2: 블록 조정** ([Puzzle 27](../puzzle_27/puzzle_27.md))

```txt
Block (8 warps) → aggregate warp results → 1 block total
```

### **레벨 3: 클러스터 조정** (이 퍼즐)

```txt
Cluster (4 blocks) → cluster_arrive/wait → synchronized completion
```

**결합 효과:** 1024개 스레드 → 32개 Warp 리더 → 4개 블록 결과 → 조정된 클러스터 완료

## 코드 실행

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p34 --advanced
```

  </div>
  <div class="tab-content">

```bash
uv run poe p34 --advanced
```

  </div>
</div>

**예상 출력:**

```txt
Testing Advanced Cluster Algorithms
SIZE: 1024 TPB: 256 CLUSTER_SIZE: 4
Advanced cluster algorithm results:
  Block 0 : 122.799995
  Block 1 : 247.04001
  Block 2 : 372.72
  Block 3 : 499.83997
✅ Advanced cluster patterns tests passed!
```

**성공 기준:**

- **계층적 스케일링**: 결과가 다단계 조정 효과를 보여줍니다
- **Warp 최적화**: `elect_one_sync()`가 중복 연산을 줄입니다
- **클러스터 조정**: 모든 블록이 처리를 성공적으로 완료합니다
- **성능 패턴**: 더 높은 블록 ID가 비례적으로 더 큰 결과를 생성합니다

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p34/p34.mojo:advanced_cluster_patterns_solution}}
```

<div class="solution-explanation">

**고급 클러스터 패턴 풀이는 GPU 활용률을 극대화하기 위해 Warp, 블록, 클러스터 조정을 결합하는 정교한 3단계 계층적 최적화를 보여줍니다:**

## **레벨 1: Warp 레벨 최적화 (스레드 선출)**

**데이터 준비 및 스케일링:**

```mojo
var data_scale = Float32(block_id + 1)  # Block-specific scaling factor
if global_i < size:
    shared_data[local_i] = input[global_i] * data_scale
else:
    shared_data[local_i] = 0.0  # Zero-pad for out-of-bounds
barrier()  # Ensure all threads complete data loading
```

**Warp 레벨 스레드 선출:**

```mojo
if elect_one_sync():  # Hardware elects exactly 1 thread per warp
    var warp_sum: Float32 = 0.0
    var warp_start = (local_i // 32) * 32  # Calculate warp boundary
    for i in range(32):  # Process entire warp's data
        if warp_start + i < tpb:
            warp_sum += shared_data[warp_start + i][0]
    shared_data[local_i] = warp_sum  # Store result at elected thread's position
```

**Warp 경계 계산 설명:**

- **스레드 37** (Warp 1): `warp_start = (37 // 32) * 32 = 1 * 32 = 32`
- **스레드 67** (Warp 2): `warp_start = (67 // 32) * 32 = 2 * 32 = 64`
- **스레드 199** (Warp 6): `warp_start = (199 // 32) * 32 = 6 * 32 = 192`

**선출 패턴 시각화 (TPB=256, 8 Warp):**

```txt
Warp 0 (threads 0-31):   elect_one_sync() → Thread 0   processes elements 0-31
Warp 1 (threads 32-63):  elect_one_sync() → Thread 32  processes elements 32-63
Warp 2 (threads 64-95):  elect_one_sync() → Thread 64  processes elements 64-95
Warp 3 (threads 96-127): elect_one_sync() → Thread 96  processes elements 96-127
Warp 4 (threads 128-159):elect_one_sync() → Thread 128 processes elements 128-159
Warp 5 (threads 160-191):elect_one_sync() → Thread 160 processes elements 160-191
Warp 6 (threads 192-223):elect_one_sync() → Thread 192 processes elements 192-223
Warp 7 (threads 224-255):elect_one_sync() → Thread 224 processes elements 224-255
```

## **레벨 2: 블록 레벨 집계 (Warp 리더 조정)**

**Warp 간 동기화:**

```mojo
barrier()  # Ensure all warps complete their elected computations
```

**Warp 리더 집계 (스레드 0만):**

```mojo
if local_i == 0:
    var block_total: Float32 = 0.0
    for i in range(0, tpb, 32):  # Iterate through warp leader positions
        if i < tpb:
            block_total += shared_data[i][0]  # Sum warp results
    output[block_id] = block_total
```

**메모리 접근 패턴:**

- 스레드 0이 다음 위치에서 읽습니다: `shared_data[0]`, `shared_data[32]`, `shared_data[64]`, `shared_data[96]`, `shared_data[128]`, `shared_data[160]`, `shared_data[192]`, `shared_data[224]`
- 이 위치들에는 선출된 스레드가 계산한 Warp 합계가 저장되어 있습니다
- 결과: 8개 Warp 합계 → 1개 블록 합계

## **레벨 3: 클러스터 레벨 단계적 동기화**

**단계적 동기화 접근:**

```mojo
cluster_arrive()  # Non-blocking: signal this block's completion
# ... Thread 0 computes and stores block result ...
cluster_wait()    # Blocking: wait for all blocks to complete
```

**왜 단계적 동기화를 사용할까?**

- **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**를 최종 연산 **이전에** 호출하면 작업 중첩이 가능합니다
- 다른 블록이 아직 처리 중인 동안에도 블록이 자체 결과를 계산할 수 있습니다
- **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**로 결정론적 완료 순서를 보장합니다
- 독립적인 블록 연산의 경우 [`cluster_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_sync)보다 더 효율적입니다

## **고급 패턴 특성**

**계층적 연산 축소:**

1. **256개 스레드** → **8개 선출 스레드** (블록당 32배 축소)
2. **8개 Warp 합계** → **1개 블록 합계** (블록당 8배 축소)
3. **4개 블록** → **단계적 완료** (동기화된 종료)
4. **전체 효율**: 블록당 중복 연산 256배 축소

**메모리 접근 최적화:**

- **레벨 1**: `input[global_i]`에서 coalescing된 읽기, 공유 메모리에 스케일링된 쓰기
- **레벨 2**: 선출된 스레드가 Warp 레벨 집계를 수행합니다 (256개 대신 8개 연산)
- **레벨 3**: 스레드 0이 블록 레벨 집계를 수행합니다 (8개 대신 1개 연산)
- **결과**: 계층적 reduction을 통해 메모리 대역폭 사용량을 최소화합니다

**동기화 계층 구조:**

1. **`barrier()`**: 블록 내부 스레드 동기화 (데이터 로딩 및 Warp 처리 후)
2. **[`cluster_arrive()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_arrive)**: 블록 간 신호 (논블로킹, 작업 중첩 가능)
3. **[`cluster_wait()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/cluster_wait)**: 블록 간 동기화 (블로킹, 완료 순서 보장)

**왜 "고급"인가:**

- **다단계 최적화**: Warp, 블록, 클러스터 프로그래밍 기법을 결합합니다
- **하드웨어 효율**: [`elect_one_sync()`](https://docs.modular.com/mojo/stdlib/gpu/cluster/elect_one_sync)를 활용하여 Warp 활용률을 최적화합니다
- **단계적 조정**: 고급 클러스터 API를 사용하여 유연한 동기화를 구현합니다
- **프로덕션 수준**: 실제 GPU 라이브러리에서 사용되는 패턴을 보여줍니다

**실제 성능 이점:**

- **메모리 부하 감소**: 동시에 공유 메모리에 접근하는 스레드 수가 적어집니다
- **더 나은 Warp 활용**: 선출된 스레드가 집중적인 연산을 수행합니다
- **확장 가능한 조정**: 단계적 동기화가 더 큰 클러스터 크기를 처리합니다
- **알고리즘 유연성**: 복잡한 다단계 처리 파이프라인의 기반이 됩니다

**복잡도 분석:**

- **Warp 레벨**: 선출된 스레드당 O(32) 연산 = 블록당 총 O(256)
- **블록 레벨**: 블록당 O(8) 집계 연산
- **클러스터 레벨**: 블록당 O(1) 동기화 오버헤드
- **전체**: 대규모 병렬화 이점을 가진 선형 복잡도

</div>
</details>

## 완전한 GPU 계층 구조

축하합니다! 이 퍼즐을 완료함으로써 **완전한 GPU 프로그래밍 스택**을 학습했습니다:

✅ **스레드 레벨 프로그래밍**: 개별 실행 단위
✅ **[Warp 레벨 프로그래밍](../puzzle_24/puzzle_24.md)**: 32개 스레드 SIMT 조정
✅ **[블록 레벨 프로그래밍](../puzzle_27/puzzle_27.md)**: 멀티 Warp 조정과 공유 메모리
✅ **🆕 클러스터 레벨 프로그래밍**: SM90+ API를 활용한 멀티 블록 조정
✅ 클러스터 동기화 기본 요소로 **여러 스레드 블록을 조정**
✅ 클러스터 API를 사용하여 **단일 블록 한계를 넘어 알고리즘을 확장**
✅ Warp + 블록 + 클러스터 조정을 결합한 **계층적 알고리즘을 구현**
✅ SM90+ 클러스터 프로그래밍으로 **차세대 GPU 하드웨어를 활용**

## 실전 응용

이 퍼즐의 계층적 조정 패턴은 다음 분야의 기반이 됩니다:

**고성능 컴퓨팅:**

- **멀티 그리드 기법**: 각 레벨이 서로 다른 해상도의 그리드를 처리합니다
- **도메인 분해**: 문제의 하위 도메인에 걸친 계층적 조정
- **병렬 반복법**: Warp 레벨의 로컬 연산과 클러스터 레벨의 전역 통신

**딥러닝:**

- **모델 병렬 처리**: 각 블록이 모델의 서로 다른 구성 요소를 처리합니다
- **파이프라인 병렬 처리**: 여러 트랜스포머 레이어에 걸친 단계적 처리
- **기울기 집계**: 분산 학습 노드에 걸친 계층적 reduction

**그래픽스 및 시각화:**

- **멀티 패스 렌더링**: 복잡한 시각 효과를 위한 단계적 처리
- **계층적 컬링**: 각 레벨이 서로 다른 세분도에서 컬링합니다
- **병렬 지오메트리 처리**: 조정된 변환 파이프라인

## 다음 단계

이제 최신 하드웨어에서 사용 가능한 **최첨단 GPU 프로그래밍 기법**을 배웠습니다!

**더 많은 도전을 할 준비가 되셨나요?** 다른 고급 GPU 프로그래밍 주제를 탐구하고, [Puzzle 30-32의 성능 최적화 기법](../puzzle_30/puzzle_30.md)을 복습하고, [NVIDIA 도구의 프로파일링 방법론](../puzzle_30/nvidia_profiling_basics.md)을 적용하거나, 이 클러스터 프로그래밍 패턴을 기반으로 자신만의 연산 워크로드를 구축해 보세요!
