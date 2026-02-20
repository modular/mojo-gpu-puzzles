<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# Puzzle 28: 비동기 메모리 연산과 복사 중첩

**GPU 메모리 병목 현상:** 실제 GPU 알고리즘 대부분은 좌절스러운 벽에 부딪힙니다 - 연산 능력이 아니라 **메모리 대역폭**에 의해 제한된다는 것입니다. 비싼 GPU 코어가 느린 DRAM에서 데이터가 도착하기를 기다리며 놀고 있는 것이죠.

GPU 프로그래밍에서 흔히 볼 수 있는 상황을 살펴보겠습니다:

```mojo
# 성능의 적 - 순차적 메모리 연산
load_input_tile()     # ← DRAM 대기 500 사이클
load_kernel_data()    # ← 또 100 사이클 대기
barrier()             # ← 모든 스레드가 유휴 대기
compute()             # ← 드디어 실제 연산 50 사이클
# 총: 650 사이클, 연산 활용률 겨우 7.7%!
```

**이렇게 할 수 있다면 어떨까요?**

```mojo
# 성능 개선 - 중첩 연산
launch_async_load()   # ← 백그라운드에서 500 사이클 전송 시작
load_small_data()     # ← 대기 중 유용한 작업 100 사이클
wait_and_compute()    # ← 나머지 ~400 사이클만 대기 후 연산
# 총: ~550 사이클, 45% 향상!
```

**이것이 비동기 메모리 연산의 위력입니다** - 느린 알고리즘과 GPU의 잠재력을 최대한 발휘하는 알고리즘의 차이를 만들어 냅니다.

## 왜 중요한가

이 퍼즐에서는 [Puzzle 13](../puzzle_13/puzzle_13.md)의 메모리 바운드 1D convolution을 **연산 뒤에 메모리 latency를 숨기는** 고성능 구현으로 변환합니다. 단순한 학술적 연습이 아닙니다 - 이 패턴들은 다음 분야의 핵심입니다:

- **딥러닝**: 가중치와 활성화값의 효율적 로딩
- **과학 연산**: stencil 연산에서 데이터 전송 중첩
- **이미지 처리**: 메모리 계층 구조를 통한 대규모 데이터셋 스트리밍
- **모든 메모리 바운드 알고리즘**: 대기 시간을 생산적인 작업으로 전환

## 사전 준비

시작하기 전에 다음 내용을 확실히 이해하고 있어야 합니다:

**필수 GPU 프로그래밍 개념:**

- **공유 메모리 프로그래밍** ([Puzzle 8](../puzzle_08/puzzle_08.md), [Puzzle 16](../puzzle_16/puzzle_16.md)) - matmul 패턴을 확장합니다
- **메모리 병합(coalescing)** ([Puzzle 21](../puzzle_21/puzzle_21.md)) - 최적의 비동기 전송에 필수
- **Tiled 처리** ([Puzzle 23](../puzzle_23/puzzle_23.md)) - 이 최적화의 기반

**하드웨어 이해:**

- GPU 메모리 계층 구조 (DRAM → 공유 메모리 → 레지스터)
- 스레드 블록 구성과 동기화
- 메모리 latency vs. 대역폭에 대한 기본 이해

**API 숙지:** [Mojo GPU Memory Operations](https://docs.modular.com/mojo/std/gpu/memory/)

> **⚠️ 하드웨어 호환성 참고:** 이 퍼즐은 최신 GPU 아키텍처가 필요할 수 있는 비동기 복사 연산(`copy_dram_to_sram_async`, `async_copy_wait_all`)을 사용합니다. `.async` 수정자나 지원되지 않는 연산 관련 컴파일 오류가 발생하면 해당 GPU가 이 기능을 지원하지 않는 것일 수 있습니다. 그래도 메모리 최적화 패턴을 이해하는 데 개념은 여전히 유용합니다.
>
> **GPU 컴퓨팅 능력 확인:**
>
> ```bash
> nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits
> ```
>
> - **SM_70 이상** (예: V100, T4, A10G, RTX 20+ 시리즈): 기본 비동기 복사 지원
> - **SM_80 이상** (예: A100, RTX 30+ 시리즈): 전체 비동기 복사 기능
> - **SM_90 이상** (예: H100, RTX 40+ 시리즈): 고급 TMA 연산 지원

## 학습 내용

이 퍼즐을 마치면 다음을 직접 경험하게 됩니다:

### **핵심 기법**

- **비동기 복사 기본 요소**: 백그라운드 DRAM→SRAM 전송 시작
- **Latency hiding**: 비용이 큰 메모리 연산을 유용한 연산과 중첩
- **스레드 레이아웃 최적화**: 메모리 접근 패턴을 하드웨어에 맞추기
- **파이프라인 프로그래밍**: 메모리 활용을 극대화하도록 알고리즘 구조화

### **주요 API**

[Puzzle 16의 관용적 matmul](../puzzle_16/tiled.md#solution-idiomatic-layouttensor-tiling)에서 소개한 비동기 복사 연산을 기반으로, 이제 메모리 최적화 잠재력에 집중합니다:

- **[`copy_dram_to_sram_async()`](https://docs.modular.com/mojo/kernels/layout/layout_tensor/copy_dram_to_sram_async/)**: 전용 복사 엔진을 사용하여 백그라운드 DRAM→SRAM 전송 시작
- **[`async_copy_wait_all()`](https://docs.modular.com/mojo/std/gpu/memory/memory/async_copy_wait_all/)**: 공유 메모리 접근 전 전송 완료 동기화

**Puzzle 16과 다른 점은?** Puzzle 16에서는 matmul의 깔끔한 타일 로딩을 위해 비동기 복사를 사용했다면, 이 퍼즐은 **latency hiding**에 집중합니다 - 비용이 큰 메모리 연산과 유용한 연산 작업을 중첩하도록 알고리즘을 구조화하는 것입니다.

### **성능 효과**

이 기법들은 다음과 같은 방식으로 메모리 바운드 알고리즘의 **성능을 크게 향상**시킵니다:

- **DRAM latency 숨기기**: 유휴 대기를 생산적인 연산 시간으로 전환
- **대역폭 극대화**: 최적의 메모리 접근 패턴으로 캐시 미스 방지
- **파이프라인 효율**: 메모리 전송이 병렬로 일어나는 동안 연산 유닛을 바쁘게 유지

> **비동기 복사 연산이란?** [비동기 복사 연산](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)은 GPU 블록이 다른 작업을 계속하는 동안 백그라운드에서 실행되는 메모리 전송을 시작할 수 있게 해줍니다. 이를 통해 연산과 메모리 이동을 중첩할 수 있으며, 이는 메모리 바운드 알고리즘의 근본적인 최적화 기법입니다.

💡 **성공 팁**: 이것을 **GPU 메모리를 위한 파이프라인 프로그래밍**으로 생각하세요 - 단계를 중첩하고, latency를 숨기고, 처리량을 극대화합니다. 목표는 데이터가 백그라운드에서 이동하는 동안 비싼 연산 유닛을 바쁘게 유지하는 것입니다.

## Halo 영역 이해하기

비동기 복사 연산으로 들어가기 전에, convolution과 같은 stencil 연산의 tile 기반 처리에 필수적인 **halo 영역**(ghost cell 또는 guard cell이라고도 함)을 이해하는 것이 중요합니다.

### Halo 영역이란?

**Halo 영역**은 stencil 연산에 필요한 이웃 데이터를 제공하기 위해 처리 타일의 경계를 넘어 확장되는 **추가 요소**입니다. 타일 가장자리 근처의 요소를 처리할 때, stencil 연산은 인접 타일의 데이터에 접근해야 합니다.

### Halo 영역이 필요한 이유

타일에서 5점 커널을 사용하는 1D convolution을 생각해 봅시다:

```
원본 데이터:      [... | a b c d e f g h i j k l m n o | ...]
처리 타일:              [c d e f g h i j k l m n o]
                            ^                 ^
                      왼쪽 타일에서        오른쪽 타일에서
                      이웃 필요           이웃 필요

Halo 포함:       [a b | c d e f g h i j k l m n o | p q]
                 ^^^                               ^^^
                 왼쪽 halo                      오른쪽 halo
```

**주요 특성:**

- **Halo 크기**: 일반적으로 각 측면에 `KERNEL_SIZE // 2`개 요소
- **목적**: 타일 경계에서 정확한 stencil 연산 가능
- **내용**: 이웃 타일의 데이터 복사본 또는 경계 조건
- **메모리 오버헤드**: 큰 연산 이점을 위한 적은 추가 저장 공간

### Convolution에서의 halo 영역

5점 convolution 커널 \\([k_0, k_1, k_2, k_3, k_4]\\)의 경우:

- **중심 요소**: \\(k_2\\)가 현재 처리 요소와 정렬
- **왼쪽 이웃**: \\(k_0, k_1\\)은 왼쪽 2개 요소 필요
- **오른쪽 이웃**: \\(k_3, k_4\\)은 오른쪽 2개 요소 필요
- **Halo 크기**: 각 측면에 `HALO_SIZE = 5 // 2 = 2`개 요소

**Halo 영역 없이:**

- 타일 경계 요소에서 전체 convolution을 수행할 수 없음
- 잘못된 출력이나 복잡한 경계 처리 로직이 필요
- 분산된 메모리 접근 패턴으로 성능 저하

**Halo 영역 사용 시:**

- 모든 타일 요소가 로컬 데이터를 사용하여 전체 convolution 수행 가능
- 예측 가능한 메모리 접근으로 간결하고 효율적인 연산
- 더 나은 캐시 활용과 메모리 병합

이 개념은 비동기 복사 연산을 구현할 때 특히 중요합니다. halo 영역을 올바르게 로딩하고 동기화해야 여러 타일에 걸친 정확한 병렬 연산을 보장할 수 있습니다.

## 비동기 복사 중첩을 활용한 1D convolution

**[Puzzle 13](../puzzle_13/puzzle_13.md) 기반:** 이 퍼즐은 Puzzle 13의 1D convolution을 다시 다루지만, 이번에는 비동기 복사 연산으로 메모리 latency를 연산 뒤에 숨기는 최적화를 적용합니다. 단순한 동기식 메모리 접근 대신, 하드웨어 가속을 사용하여 비용이 큰 DRAM 전송과 유용한 작업을 중첩합니다.

### 구성

- 벡터 크기: `VECTOR_SIZE = 16384` (여러 블록에 걸친 16K 요소)
- 타일 크기: `CONV_TILE_SIZE = 256` (처리 타일 크기)
- 블록 구성: 블록당 `(256, 1)` 스레드
- 그리드 구성: 그리드당 `(VECTOR_SIZE // CONV_TILE_SIZE, 1)` 블록 (64개 블록)
- 커널 크기: `KERNEL_SIZE = 5` (Puzzle 13과 동일한 간단한 1D convolution)
- 데이터 타입: `DType.float32`
- 레이아웃: `Layout.row_major(VECTOR_SIZE)` (1D row-major)

### 비동기 복사의 기회

**Puzzle 16 기반:** matmul에서 깔끔한 타일 로딩을 위해 `copy_dram_to_sram_async`를 사용하는 것을 이미 보셨습니다. 이제 고성능 메모리 바운드 알고리즘의 핵심인 **latency hiding 기능**에 집중합니다.

기존의 동기식 메모리 로딩은 전송 중 연산 유닛을 유휴 상태로 대기하게 합니다. 비동기 복사 연산은 전송과 유용한 작업의 중첩을 가능하게 합니다:

```mojo
# 동기식 접근 - 비효율적:
for i in range(CONV_TILE_SIZE):
    input_shared[i] = input[base_idx + i]  # 각 로드가 DRAM을 기다림
for i in range(KERNEL_SIZE):
    kernel_shared[i] = kernel[i]           # DRAM 추가 대기
barrier()  # 연산 시작 전 모든 스레드 대기
# ↑ 총 시간 = input_transfer_time + kernel_transfer_time

# 비동기 복사 접근 - 효율적:
copy_dram_to_sram_async[thread_layout](input_shared, input_tile)  # 백그라운드 전송 시작
# 입력이 백그라운드에서 전송되는 동안, 커널을 동기식으로 로딩
for i in range(KERNEL_SIZE):
    kernel_shared[i] = kernel[i]  # 비동기 입력 전송과 중첩
async_copy_wait_all()  # 두 연산이 모두 완료될 때만 대기
# ↑ 총 시간 = MAX(input_transfer_time, kernel_transfer_time)
```

**비동기 복사가 잘 동작하는 이유:**

- **전용 복사 엔진**: 최신 GPU는 레지스터를 우회하고 진정한 연산-메모리 중첩을 가능하게 하는 전용 하드웨어를 갖추고 있습니다 ([Puzzle 16](../puzzle_16/tiled.md#solution-idiomatic-layouttensor-tiling)에서 설명)
- **Latency hiding**: GPU 스레드가 다른 연산을 실행하는 동안 메모리 전송이 이루어집니다
- **최적의 병합**: 스레드 레이아웃이 효율적인 DRAM 접근 패턴을 보장합니다
- **리소스 활용**: 연산 유닛이 유휴 대기 대신 계속 바쁘게 동작합니다

### 완성할 코드

Puzzle 16의 matmul 구현 패턴을 따라, 비동기 복사 연산으로 메모리 전송과 연산을 중첩하는 1D convolution을 구현하세요.

**수학적 연산:** 비동기 복사를 활용하여 대규모 벡터에 대한 1D convolution을 효율적으로 계산합니다:
\\[\\text{output}[i] = \\sum_{k=0}^{\\text{KERNEL_SIZE}-1} \\text{input}[i+k-\\text{HALO_SIZE}] \\times \\text{kernel}[k]\\]

**비동기 복사 알고리즘:**

1. **비동기 타일 로딩:** 입력 데이터의 백그라운드 DRAM→SRAM 전송 시작
2. **중첩 연산:** 입력 전송 중 작은 커널 데이터 로딩
3. **동기화:** 전송 완료 대기 후 공유 메모리를 사용하여 연산

```mojo
{{#include ../../../../../problems/p28/p28.mojo:async_copy_overlap_convolution}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p28/p28.mojo" class="filename">전체 파일 보기: problems/p28/p28.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **비동기 복사 메커니즘 이해**

비동기 복사 연산은 블록이 다른 코드를 계속 실행하는 동안 백그라운드 전송을 시작합니다.

**탐구할 핵심 질문:**

- DRAM에서 공유 메모리로 어떤 데이터를 전송해야 하는가?
- 전송이 백그라운드에서 일어나는 동안 어떤 연산을 실행할 수 있는가?
- 하드웨어가 여러 동시 연산을 어떻게 조율하는가?

**스레드 레이아웃 고려사항:**

- 블록에는 `THREADS_PER_BLOCK_ASYNC = 256`개의 스레드가 있습니다
- 타일에는 `CONV_TILE_SIZE = 256`개의 요소가 있습니다
- 어떤 레이아웃 패턴이 최적의 메모리 병합을 보장하는가?

### 2. **중첩 기회 파악**

목표는 유용한 연산 뒤에 메모리 latency를 숨기는 것입니다.

**분석 접근법:**

- 어떤 연산이 순차적으로 vs. 병렬로 일어나야 하는가?
- 어떤 데이터 전송이 큰(비용이 높은) vs. 작은(비용이 낮은)가?
- 병렬 실행을 최대화하도록 알고리즘을 어떻게 구조화할 수 있는가?

**메모리 계층 구조 고려사항:**

- 큰 입력 타일: 256 요소 × 4 바이트 = 1KB 전송
- 작은 커널: 5 요소 × 4 바이트 = 20 바이트
- 어떤 전송이 비동기 최적화의 이점을 가장 많이 받는가?

### 3. **동기화 전략**

적절한 동기화는 성능을 희생하지 않으면서 정확성을 보장합니다.

**타이밍 분석:**

- 각 연산이 실제로 데이터가 준비되어야 하는 시점은 언제인가?
- 정확성을 위해 필요한 최소한의 동기화는 무엇인가?
- 데이터 의존성을 유지하면서 불필요한 정체를 어떻게 피할 수 있는가?

**경쟁 상태 방지:**

- 전송이 완료되기 전에 연산이 시작되면 어떻게 되는가?
- memory fence와 barrier가 서로 다른 메모리 연산을 어떻게 조율하는가?

</div>
</details>

**비동기 복사 중첩 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p28
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p28
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p28
```

  </div>
  <div class="tab-content">

```bash
uv run poe p28
```

  </div>
</div>

### 풀이

<details class="solution-details">
<summary><strong>상세 설명이 포함된 전체 풀이</strong></summary>

비동기 복사 중첩 풀이는 비용이 큰 DRAM 전송과 유용한 연산을 중첩하여 메모리 latency를 숨기는 방법을 보여줍니다:

```mojo
{{#include ../../../../../solutions/p28/p28.mojo:async_copy_overlap_convolution_solution}}
```

#### **단계별 분석**

**Phase 1: 비동기 복사 시작**

```mojo
# Phase 1: Launch async copy for input tile
input_tile = input.tile[CONV_TILE_SIZE](block_idx.x)
comptime load_layout = Layout.row_major(THREADS_PER_BLOCK_ASYNC)
copy_dram_to_sram_async[thread_layout=load_layout](input_shared, input_tile)
```

- **타일 생성**: `input.tile[CONV_TILE_SIZE](block_idx.x)`는 `block_idx.x * 256`에서 시작하는 256개 요소의 입력 배열 뷰를 생성합니다. Mojo의 [`tile` 메서드](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/#tile)는 경계 검사나 제로 패딩을 **수행하지 않습니다**. 범위를 벗어난 인덱스 접근은 미정의 동작을 초래합니다. 구현에서 타일 크기와 offset이 유효한 배열 범위 내에 있는지 확인해야 합니다.

- **스레드 레이아웃**: `Layout.row_major(THREADS_PER_BLOCK_ASYNC, 1)`는 블록 구성과 일치하는 `256 x 1` 레이아웃을 생성합니다. 이것은 **필수**입니다 - 최적의 병합된 메모리 접근을 위해 레이아웃이 물리적 스레드 배치와 일치해야 합니다. 레이아웃이 일치하지 않으면 스레드가 비연속적인 메모리 주소에 접근하여 병합이 깨지고 성능이 심각하게 저하됩니다.

- **비동기 복사 시작**: `copy_dram_to_sram_async`는 DRAM에서 공유 메모리로의 백그라운드 전송을 시작합니다. 하드웨어가 256개의 float(1KB)를 복사하는 동안 블록은 계속 실행됩니다.

**Phase 2: 중첩 연산**

```mojo
# Phase 2: Load kernel synchronously (small data)
if local_i < KERNEL_SIZE:
    kernel_shared[local_i] = kernel[local_i]
```

- **동시 실행**: 1KB 입력 타일이 백그라운드에서 전송되는 동안, 스레드들은 작은 20바이트 커널을 동기식으로 로딩합니다. 이 중첩이 핵심 최적화입니다.

- **크기 기반 전략**: 큰 전송(입력 타일)은 비동기 복사를, 작은 전송(커널)은 동기식 로딩을 사용합니다. 이는 복잡성과 성능 이점의 균형을 맞춥니다.

**Phase 3: 동기화**

```mojo
# Phase 3: Wait for async copy to complete
async_copy_wait_all()  # Always wait since we always do async copy
barrier()  # Sync all threads
```

- **전송 완료**: `async_copy_wait_all()`은 모든 비동기 전송이 완료될 때까지 대기합니다. `input_shared`에 접근하기 전에 반드시 필요합니다.

- **스레드 동기화**: `barrier()`는 모든 스레드가 연산으로 넘어가기 전에 완료된 전송을 확인하도록 보장합니다.

**Phase 4: 연산**

```mojo
# Phase 4: Compute convolution
global_i = block_idx.x * CONV_TILE_SIZE + local_i
if local_i < CONV_TILE_SIZE and global_i < output.shape[0]():
    var result: output.element_type = 0

    if local_i >= HALO_SIZE and local_i < CONV_TILE_SIZE - HALO_SIZE:
        # Full convolution for center elements
        for k in range(KERNEL_SIZE):
            input_idx = local_i + k - HALO_SIZE
            if input_idx >= 0 and input_idx < CONV_TILE_SIZE:
                result += input_shared[input_idx] * kernel_shared[k]
    else:
        # For boundary elements, just copy input (no convolution)
        result = input_shared[local_i]

    output[global_i] = result
```

- **빠른 공유 메모리 접근**: 모든 연산이 미리 로드된 공유 메모리 데이터를 사용하여, 연산 집약적인 convolution 루프에서 느린 DRAM 접근을 피합니다.

- **단순화된 경계 처리**: 이 구현은 타일 경계 근처 요소를 처리하기 위해 실용적인 접근 방식을 사용합니다:
  - **중심 요소** (`local_i >= HALO_SIZE`이고 `local_i < CONV_TILE_SIZE - HALO_SIZE`): 공유 메모리 데이터를 사용하여 전체 5점 convolution 적용
  - **경계 요소** (각 타일의 처음 2개와 마지막 2개 요소): 복잡한 경계 로직을 피하기 위해 convolution 없이 입력을 직접 복사

  **교육적 근거**: 이 접근 방식은 복잡한 경계 처리보다 비동기 복사 패턴 시연을 우선시합니다. `HALO_SIZE = 2`인 256개 요소 타일에서, 요소 0-1과 254-255는 입력 복사를, 요소 2-253은 전체 convolution을 사용합니다. 이를 통해 동작하는 구현을 제공하면서 메모리 최적화에 초점을 유지합니다.

#### **성능 분석**

**비동기 복사 없이 (동기식):**

```
Total Time = Input_Transfer_Time + Kernel_Transfer_Time + Compute_Time
           = Large_DRAM_transfer + Small_DRAM_transfer + convolution
           = Major_latency + Minor_latency + computation_work
```

**비동기 복사 사용 (중첩):**

```
Total Time = MAX(Input_Transfer_Time, Kernel_Transfer_Time) + Compute_Time
           = MAX(Major_latency, Minor_latency) + computation_work
           = Major_latency + computation_work
```

**성능 향상**: 더 큰 입력 전송 뒤에 더 작은 커널 전송의 latency를 숨김으로써 성능이 향상됩니다. 실제 성능 향상 폭은 전송의 상대적 크기와 사용 가능한 메모리 대역폭에 따라 달라집니다. 더 큰 중첩이 가능한 메모리 바운드 시나리오에서는 성능 향상이 훨씬 클 수 있습니다.

#### **핵심 기술적 통찰**

1. **스레드 레이아웃 매칭**: `Layout.row_major(256, 1)` 레이아웃이 블록의 `(256, 1)` 스레드 구성과 정확히 일치하여 최적의 메모리 병합을 가능하게 합니다.

2. **경쟁 상태 방지**: 적절한 순서 지정(비동기 복사 → 커널 로드 → 대기 → barrier → 연산)으로 공유 메모리를 손상시킬 수 있는 모든 경쟁 상태를 제거합니다.

3. **하드웨어 최적화**: 최신 GPU는 비동기 복사 연산을 위한 전용 하드웨어를 갖추고 있어, 메모리 유닛과 연산 유닛 사이의 진정한 병렬 처리가 가능합니다.

4. **메모리 계층 구조 활용**: 이 패턴은 데이터를 계층 구조를 통해 효율적으로 이동시킵니다: DRAM → 공유 메모리 → 레지스터 → 연산.

5. **테스트-구현 일관성**: 테스트 검증 로직은 `local_i_in_tile = i % CONV_TILE_SIZE`를 검사하여 각 요소가 convolution 결과(중심 요소)를 기대해야 하는지 입력 복사(경계 요소)를 기대해야 하는지 판별하며, 경계 처리 전략과 일치합니다. 이를 통해 단순화된 경계 접근 방식의 정확한 검증을 보장합니다.

이 풀이는 단순한 메모리 바운드 convolution을 유용한 작업 뒤에 메모리 latency를 숨기는 최적화된 구현으로 변환하여, 고성능 GPU 프로그래밍의 기본 원리를 보여줍니다.

</details>
