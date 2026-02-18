<!-- i18n-source-commit: 43fce1182f8029e7edc50157aed0e6ebb8129d42 -->

# 🧠 Warp Lane과 SIMT 실행

## Warp 프로그래밍 vs SIMD 멘탈 모델

### Warp란 무엇인가?

**Warp**는 32개(또는 64개)의 GPU 스레드가 서로 다른 데이터에 대해 **동일한 명령을 동시에 실행**하는 그룹입니다. 각 스레드가 벡터 프로세서의 "Lane" 역할을 하는 **동기화된 벡터 유닛**이라고 생각하면 됩니다.

**간단한 예시:**

```mojo
from gpu.primitives.warp import sum
# Warp 내 32개 스레드가 동시에 실행:
var my_value = input[my_thread_id]     # 각 스레드가 서로 다른 데이터를 가져옴
var warp_total = sum(my_value)         # 모든 스레드가 하나의 합계에 기여
```

무슨 일이 일어난 걸까요? 32개의 개별 스레드가 복잡한 조율을 하는 대신, **Warp**가 자동으로 동기화하여 하나의 결과를 만들어냈습니다. 이것이 바로 **SIMT(Single Instruction, Multiple Thread)** 실행입니다.

### SIMT vs SIMD 비교

CPU 벡터 프로그래밍(SIMD)에 익숙하다면, GPU Warp는 비슷하지만 핵심적인 차이가 있습니다:

| 관점 | CPU SIMD (예: AVX) | GPU Warp (SIMT) |
|--------|---------------------|------------------|
| **프로그래밍 모델** | 명시적 벡터 연산 | 스레드 기반 프로그래밍 |
| **데이터 폭** | 고정 (256/512 비트) | 유연 (32/64 스레드) |
| **동기화** | 명령 내 암시적 | Warp 내 암시적 |
| **통신** | 메모리/레지스터 경유 | shuffle 연산 경유 |
| **분기 처리** | 해당 없음 | 하드웨어 마스킹 |
| **예시** | `a + b` | `sum(thread_value)` |

**CPU SIMD 방식 (C++ intrinsics):**

```cpp
// 명시적 벡터 연산 - 8개의 float를 병렬로
__m256 result = _mm256_add_ps(a, b);   // 8쌍을 동시에 덧셈
```

**CPU SIMD 방식 (Mojo):**

```mojo
# Mojo에서 SIMD는 일급 시민 타입이므로 a, b가 SIMD 타입이면
# 덧셈이 병렬로 수행됩니다
var result = a + b # 8쌍을 동시에 덧셈
```

**GPU SIMT 방식 (Mojo):**

```mojo
# 스레드 기반 코드가 벡터 연산으로 변환됩니다
from gpu.primitives.warp import sum

var my_data = input[thread_id]         # 각 스레드가 자기 요소를 가져옴
var partial = my_data * coefficient    # 모든 스레드가 동시에 계산
var total = sum(partial)               # 하드웨어가 합산을 조율
```

### Warp를 강력하게 만드는 핵심 개념

**1. Lane 식별:** 각 스레드는 사실상 비용 없이 접근할 수 있는 "Lane ID" (0~31)를 갖습니다

```mojo
var my_lane = lane_id()  # 하드웨어 레지스터를 읽을 뿐
```

**2. 암시적 동기화:** Warp 내에서 barrier가 필요 없습니다

```mojo
# 그냥 동작합니다 - 모든 스레드가 자동으로 동기화됩니다
var sum = sum(my_contribution)
```

**3. 효율적인 통신:** 메모리 없이도 스레드 간 데이터 공유가 가능합니다

```mojo
# Lane 0의 값을 다른 모든 Lane으로 전달
var broadcasted = shuffle_idx(my_value, 0)
```

**핵심 통찰:** SIMT를 사용하면 자연스러운 스레드 코드를 작성하면서도 효율적인 벡터 연산으로 실행할 수 있어, 스레드 프로그래밍의 편리함과 벡터 처리의 성능을 모두 얻을 수 있습니다.

### GPU 실행 계층 구조에서 Warp의 위치

Warp가 전체 GPU 실행 모델과 어떻게 연결되는지 자세히 알아보려면 [GPU 스레딩 vs SIMD 개념](../puzzle_23/gpu-thread-vs-simd.md)을 참고하세요. Warp의 위치는 다음과 같습니다:

```
GPU Device
├── Grid (전체 문제)
│   ├── Block 1 (스레드 그룹, 공유 메모리)
│   │   ├── Warp 1 (32 스레드, lockstep 실행) ← 이 레벨
│   │   │   ├── Thread 1 → SIMD 연산
│   │   │   ├── Thread 2 → SIMD 연산
│   │   │   └── ... (총 32개 스레드)
│   │   └── Warp 2 (32 스레드)
│   └── Block 2 (독립적인 그룹)
```

**Warp 프로그래밍은 "Warp 레벨"에서 동작합니다** - 단일 Warp 내의 32개 스레드를 모두 조율하는 연산을 다루며, 그렇지 않으면 복잡한 공유 메모리 조율이 필요한 `sum()` 같은 강력한 기본 요소를 사용할 수 있습니다.

이 멘탈 모델은 문제가 Warp 연산에 자연스럽게 매핑되는 경우와 기존의 공유 메모리 방식이 필요한 경우를 구분하는 데 도움이 됩니다.

## Warp 프로그래밍의 하드웨어 기반

**SIMT(Single Instruction, Multiple Thread)** 실행을 이해하는 것은 효과적인 Warp 프로그래밍에 필수적입니다. 이것은 단순한 소프트웨어 추상화가 아니라, GPU 하드웨어가 실리콘 수준에서 실제로 작동하는 방식입니다.

## SIMT 실행이란?

**SIMT**란 Warp 내에서 모든 스레드가 **서로 다른 데이터**에 대해 **같은 명령**을 **동시에** 실행한다는 뜻입니다. 이는 완전히 다른 명령을 독립적으로 실행할 수 있는 CPU 스레드와 근본적으로 다릅니다.

### CPU vs GPU 실행 모델

| 관점 | CPU (MIMD) | GPU Warp (SIMT) |
|--------|------------|------------------|
| **명령 모델** | Multiple Instructions, Multiple Data | Single Instruction, Multiple Thread |
| **Core 1** | `add r1, r2` | `add r1, r2` |
| **Core 2** | `load r3, [mem]` | `add r1, r2` (동일 명령) |
| **Core 3** | `branch loop` | `add r1, r2` (동일 명령) |
| **... Core 32** | `다른 명령` | `add r1, r2` (동일 명령) |
| **실행 방식** | 독립적, 비동기 | 동기화, lockstep |
| **스케줄링** | 복잡, OS 관리 | 단순, 하드웨어 관리 |
| **데이터** | 독립적인 데이터 세트 | 서로 다른 데이터, 같은 연산 |

**GPU Warp 실행 패턴:**

- **명령**: 32개 Lane 모두 동일: `add r1, r2`
- **Lane 0**: `Data0`에 연산 → `Result0`
- **Lane 1**: `Data1`에 연산 → `Result1`
- **Lane 2**: `Data2`에 연산 → `Result2`
- **... (모든 Lane이 동시에 실행)**
- **Lane 31**: `Data31`에 연산 → `Result31`

**핵심 통찰:** 모든 Lane이 **서로 다른 데이터**에 대해 **같은 명령**을 **동시에** 실행합니다.

### SIMT가 GPU에 적합한 이유

GPU는 latency가 아닌 **처리량**에 최적화되어 있습니다. SIMT가 가능하게 하는 것들:

- **하드웨어 단순화**: 하나의 명령 디코더가 32개 또는 64개 스레드를 처리
- **실행 효율성**: Warp 내 스레드 간 복잡한 스케줄링 불필요
- **메모리 대역폭**: coalescing된 메모리 접근 패턴
- **전력 효율성**: Lane 전체에 걸쳐 제어 로직 공유

## Warp 실행 메커니즘

### Lane 번호와 식별

Warp 내 각 스레드는 0부터 `WARP_SIZE-1`까지의 **Lane ID**를 갖습니다:

```mojo
from gpu import lane_id
from gpu.primitives.warp import WARP_SIZE

# kernel 함수 내에서:
my_lane = lane_id()  # 0-31 (NVIDIA/RDNA) 또는 0-63 (CDNA) 반환
```

**핵심 통찰:** `lane_id()`는 **비용이 없습니다** - 값을 계산하는 것이 아니라 하드웨어 레지스터를 읽을 뿐입니다.

### Warp 내 동기화

SIMT의 가장 강력한 측면: **암시적 동기화**.

```mojo
# thread_idx.x < WARP_SIZE인 경우의 예시

# 1. 기존 공유 메모리 방식:
shared[thread_idx.x] = partial_result
barrier()  # 명시적 동기화 필요
var total = shared[0] + shared[1] + ... + shared[WARP_SIZE] # Sum reduction

# 2. Warp 방식:
from gpu.primitives.warp import sum

var total = sum(partial_result)  # 암시적 동기화!
```

**왜 barrier가 필요 없을까요?** 모든 Lane이 각 명령을 정확히 같은 시점에 실행하기 때문입니다. `sum()`이 시작될 때, 모든 Lane은 이미 `partial_result` 계산을 마친 상태입니다.

## Warp 분기와 수렴

### 조건 코드에서 무슨 일이 일어날까?

```mojo
if lane_id() % 2 == 0:
    # 짝수 Lane이 이 경로를 실행
    result = compute_even()
else:
    # 홀수 Lane이 이 경로를 실행
    result = compute_odd()
# 모든 Lane이 여기서 수렴
```

**하드웨어 동작 단계:**

| 단계 | 페이즈 | 활성 Lane | 대기 Lane | 효율 | 성능 비용 |
|------|-------|--------------|---------------|------------|------------------|
| **1** | 조건 평가 | 32개 Lane 전부 | 없음 | 100% | 정상 속도 |
| **2** | 짝수 Lane 분기 | Lane 0,2,4...30 (16개) | Lane 1,3,5...31 (16개) | 50% | **2배 느림** |
| **3** | 홀수 Lane 분기 | Lane 1,3,5...31 (16개) | Lane 0,2,4...30 (16개) | 50% | **2배 느림** |
| **4** | 수렴 | 32개 Lane 전부 | 없음 | 100% | 정상 속도 복귀 |

**예시 분석:**

- **2단계**: 짝수 Lane만 `compute_even()`을 실행하고 홀수 Lane은 대기
- **3단계**: 홀수 Lane만 `compute_odd()`를 실행하고 짝수 Lane은 대기
- **총 소요 시간**: `time(compute_even) + time(compute_odd)` (순차 실행)
- **분기 없는 경우**: `max(time(compute_even), time(compute_odd))` (병렬 실행)

**성능 영향:**

1. **분기**: Warp가 실행을 분리 - 일부 Lane은 활성, 나머지는 대기
2. **순차 실행**: 서로 다른 경로가 병렬이 아닌 순차적으로 실행
3. **수렴**: 모든 Lane이 다시 합류하여 함께 진행
4. **비용**: 분기가 있는 Warp는 통합 실행 대비 2배 이상의 시간 소요

### Warp 효율을 위한 모범 사례

### Warp 효율 패턴

**✅ 우수: 균일 실행 (100% 효율)**

```mojo
# 모든 Lane이 같은 작업 수행 - 분기 없음
var partial = a[global_i] * b[global_i]
var total = sum(partial)
```

*성능: 32개 Lane 모두 동시 활성*

**⚠️ 허용: 예측 가능한 분기 (~95% 효율)**

```mojo
# lane_id() 기반 분기 - 하드웨어 최적화됨
if lane_id() == 0:
    output[block_idx] = sum(partial)
```

*성능: 단일 Lane의 짧은 연산, 예측 가능한 패턴*

**🔶 주의: 구조화된 분기 (~50-75% 효율)**

```mojo
# 규칙적인 패턴은 컴파일러가 최적화 가능
if (global_i / 4) % 2 == 0:
    result = method_a()
else:
    result = method_b()
```

*성능: 예측 가능한 그룹, 일부 최적화 가능*

**❌ 회피: 데이터 의존적 분기 (~25-50% 효율)**

```mojo
# 데이터에 따라 Lane마다 다른 경로를 탈 수 있음
if input[global_i] > threshold:  # 예측 불가능한 분기
    result = expensive_computation()
else:
    result = simple_computation()
```

*성능: 무작위 분기가 Warp 효율을 떨어뜨림*

**💀 최악: 중첩된 데이터 의존적 분기 (~10-25% 효율)**

```mojo
# 예측 불가능한 분기의 다단계 중첩
if input[global_i] > threshold1:
    if input[global_i] > threshold2:
        result = very_expensive()
    else:
        result = expensive()
else:
    result = simple()
```

*성능: Warp 효율이 사실상 무너짐*

## 크로스 아키텍처 호환성

### NVIDIA vs AMD Warp 크기

```mojo
from gpu.primitives.warp import WARP_SIZE

# NVIDIA GPUs:     WARP_SIZE = 32
# AMD RDNA GPUs:   WARP_SIZE = 32 (wavefront32 모드)
# AMD CDNA GPUs:   WARP_SIZE = 64 (전통적인 wavefront64)
```

**왜 중요할까요:**

- **메모리 패턴**: 병합된 접근이 Warp 크기에 의존
- **알고리즘 설계**: reduction 트리가 Warp 크기를 고려해야 함
- **성능 확장**: AMD에서 Warp당 Lane이 2배

### 이식 가능한 Warp 코드 작성

### 아키텍처 적응 전략

**✅ 이식 가능: 항상 `WARP_SIZE` 사용**

```mojo
comptime THREADS_PER_BLOCK = (WARP_SIZE, 1)  # 자동으로 적응
comptime ELEMENTS_PER_WARP = WARP_SIZE       # 하드웨어에 맞게 확장
```

*결과: NVIDIA/AMD (32)와 AMD (64) 모두에서 최적으로 동작*

**❌ 잘못된 방식: Warp 크기를 하드코딩하지 마세요**

```mojo
comptime THREADS_PER_BLOCK = (32, 1)  # AMD GPU에서 동작 안 함!
comptime REDUCTION_SIZE = 32          # AMD에서 잘못된 값!
```

*결과: AMD에서 성능 저하, 정확성 문제 가능*

### 실제 하드웨어 영향

| GPU 아키텍처 | WARP_SIZE | Warp당 메모리 | Reduction 단계 | Lane 패턴 |
|------------------|-----------|-----------------|-----------------|--------------|
| **NVIDIA/AMD RDNA** | 32 | 128 bytes (4×32) | 5단계: 32→16→8→4→2→1 | Lane 0-31 |
| **AMD CDNA** | 64 | 256 bytes (4×64) | 6단계: 64→32→16→8→4→2→1 | Lane 0-63 |

**64 vs 32의 성능 차이:**

- **CDNA 장점**: Warp당 2배의 메모리 대역폭
- **CDNA 장점**: Warp당 2배의 연산량
- **NVIDIA/RDNA 장점**: 블록당 더 많은 Warp (더 높은 occupancy)
- **코드 이식성**: 같은 소스 코드로 양쪽 모두 최적 성능

## Warp와 메모리 접근 패턴

### 병합된 메모리 접근 패턴

**✅ 완벽: 병합된 접근 (100% 대역폭 활용)**

```mojo
# 인접 Lane → 인접 메모리 주소
var value = input[global_i]  # Lane 0→input[0], Lane 1→input[1], 등
```

**메모리 접근 패턴:**

| 접근 패턴 | NVIDIA/RDNA (32 Lane) | CDNA (64 Lane) | 대역폭 활용 | 성능 |
|----------------|-------------------|----------------|----------------------|-------------|
| **✅ Coalesced** | Lane N → 주소 4×N | Lane N → 주소 4×N | 100% | 최적 |
| | 1회 트랜잭션: 128 bytes | 1회 트랜잭션: 256 bytes | 전체 버스 폭 | 빠름 |
| **❌ Scattered** | Lane N → 임의 주소 | Lane N → 임의 주소 | ~6% | 최악 |
| | 32회 개별 트랜잭션 | 64회 개별 트랜잭션 | 대부분 유휴 버스 | **32배 느림** |

**주소 예시:**

- **Coalesced**: Lane 0→0, Lane 1→4, Lane 2→8, Lane 3→12, ...
- **Scattered**: Lane 0→1000, Lane 1→52, Lane 2→997, Lane 3→8, ...

### 공유 메모리 뱅크 충돌

**뱅크 충돌이란?**

GPU 공유 메모리가 동시 접근이 가능한 32개의 독립적인 **banks**로 나뉘어 있다고 가정합니다. **뱅크 충돌**은 Warp 내 여러 스레드가 같은 뱅크의 서로 다른 주소에 동시에 접근하려 할 때 발생합니다. 이 경우 하드웨어가 접근을 **직렬화**해야 하므로, 단일 사이클이어야 할 연산이 여러 사이클로 늘어납니다.

**핵심 개념:**

- **충돌 없음**: 각 스레드가 서로 다른 뱅크에 접근 → 모든 접근이 동시에 발생 (1 사이클)
- **뱅크 충돌**: 여러 스레드가 같은 뱅크에 접근 → 접근이 순차적으로 발생 (N개 스레드에 N 사이클)
- **Broadcast**: 모든 스레드가 같은 주소에 접근 → 하드웨어가 1 사이클로 최적화

**공유 메모리 뱅크 구성:**

| 뱅크 | 주소 (바이트 offset) | 예시 데이터 (float32) |
|------|--------------------------|------------------------|
| Bank 0 | 0, 128, 256, 384, ... | `shared[0]`, `shared[32]`, `shared[64]`, ... |
| Bank 1 | 4, 132, 260, 388, ... | `shared[1]`, `shared[33]`, `shared[65]`, ... |
| Bank 2 | 8, 136, 264, 392, ... | `shared[2]`, `shared[34]`, `shared[66]`, ... |
| ... | ... | ... |
| Bank 31 | 124, 252, 380, 508, ... | `shared[31]`, `shared[63]`, `shared[95]`, ... |

**뱅크 충돌 예시:**

| 접근 패턴 | 뱅크 사용 | 사이클 | 성능 | 설명 |
|----------------|------------|--------|-------------|-------------|
| **✅ 순차적** | `shared[thread_idx.x]` | 1 사이클 | 100% | 각 Lane이 다른 뱅크 접근 |
| | Lane 0→Bank 0, Lane 1→Bank 1, ... | | 최적 | 충돌 없음 |
| **✅ 동일 인덱스** | `shared[0]`| 1 사이클 | 100% | 모든 Lane이 같은 주소에서 broadcast |
| | 32개 Lane 전부→Bank 0 (같은 주소) | | 최적 | 충돌 없음 |
| **❌ Stride 2** | `shared[thread_idx.x * 2]` | 2 사이클 | 50% | 뱅크당 2개 Lane |
| | Lane 0,16→Bank 0; Lane 1,17→Bank 1 | | **2배 느림** | 직렬화된 접근 |
| **💀 Stride 32** | `shared[thread_idx.x * 32]` | 32 사이클 | 3% | 모든 Lane이 같은 뱅크 접근 |
| | 32개 Lane 전부→Bank 0 (다른 주소) | | **32배 느림** | 완전히 직렬화 |

## Warp 프로그래밍의 실전 활용

### Warp 연산이 가장 효과적인 경우

1. **Reduction 연산**: `sum()`, `max()` 등
2. **Broadcast 연산**: `shuffle_idx()`로 값 공유
3. **이웃 통신**: `shuffle_down()`으로 슬라이딩 윈도우
4. **Prefix 연산**: `prefix_sum()`으로 scan 알고리즘

### 성능 특성

| 연산 유형 | 기존 방식 | Warp 연산 |
|----------------|------------|-----------------|
| **Reduction (32개 요소)** | ~20개 명령 | 10개 명령 |
| **메모리 트래픽** | 높음 | 최소 |
| **동기화 비용** | 비용 높음 | 무료 |
| **코드 복잡도** | 높음 | 낮음 |

## 다음 단계

SIMT의 기반을 이해했으니, 이 개념이 어떻게 강력한 Warp 연산을 가능하게 하는지 알아볼 차례입니다. 다음 섹션에서는 `sum()`이 복잡한 reduction 패턴을 간단하고 효율적인 함수 호출로 어떻게 변환하는지 보여줍니다.

**→ 다음: [warp.sum()의 핵심](./warp_sum.md)**
