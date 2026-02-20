<!-- i18n-source-commit: 9c7176b81f278a6e8efa26c92005c139967c0c27 -->

# 충돌 없는 패턴

> **참고: 이 섹션은 NVIDIA GPU 전용입니다**
>
> 여기서 다루는 뱅크 충돌 분석과 프로파일링 기법은 NVIDIA GPU에 특화되어 있습니다. 프로파일링 명령은 NVIDIA CUDA 툴킷에 포함된 NSight Compute 도구를 사용합니다.

## 프로파일링 역량을 바탕으로

[Puzzle 30](../puzzle_30/puzzle_30.md)에서 GPU 프로파일링 기초를 배우고, [Puzzle 31](../puzzle_31/puzzle_31.md)에서 리소스 최적화를 이해했습니다. 이제 배운 탐정 기술을 새로운 성능 미스터리에 적용할 차례입니다: **공유 메모리 뱅크 충돌**.

**탐정 도전 과제:** 동일한 수학적 연산(`(input + 10) * 2`)을 수행하는 두 GPU 커널이 있습니다. 둘 다 정확히 같은 결과를 냅니다. 같은 양의 공유 메모리를 사용합니다. 점유율도 동일합니다. 그런데 하나는 공유 메모리에 **접근하는 방식** 때문에 체계적인 성능 저하를 겪습니다.

**여러분의 임무:** 지금까지 배운 프로파일링 방법론으로 이 숨겨진 성능 함정을 밝혀내고, 실제 GPU 프로그래밍에서 뱅크 충돌이 언제 중요한지 이해하세요.

## 개요

공유 메모리 뱅크 충돌은 Warp 내의 여러 스레드가 동일한 메모리 뱅크의 서로 다른 주소에 동시에 접근할 때 발생합니다. 이 탐정 사건에서는 대조적인 접근 패턴을 가진 두 커널을 살펴봅니다:

```mojo
{{#include ../../../../../problems/p32/p32.mojo:no_conflict_kernel}}
```

```mojo
{{#include ../../../../../problems/p32/p32.mojo:two_way_conflict_kernel}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p32/p32.mojo" class="filename">전체 파일 보기: problems/p32/p32.mojo</a>

**미스터리:** 이 커널들은 동일한 결과를 계산하지만 공유 메모리 접근 효율은 극적으로 다릅니다. 체계적인 프로파일링 분석을 통해 그 이유를 밝혀내는 것이 임무입니다.

## 구성

**요구 사항:**

- [Puzzle 30](../puzzle_30/puzzle_30.md)의 CUDA 툴킷과 NSight Compute가 설치된 NVIDIA GPU
- [이전 섹션](./shared_memory_bank.md)에서 다룬 공유 메모리 뱅킹 개념에 대한 이해

**커널 설정:**

```mojo
comptime SIZE = 8 * 1024      # 8K elements - focus on shared memory patterns
comptime TPB = 256            # 256 threads per block (8 warps)
comptime BLOCKS_PER_GRID = (SIZE // TPB, 1)  # 32 blocks
```

**핵심 통찰:** 글로벌 메모리 대역폭 제한이 아닌 공유 메모리 효과를 부각하기 위해 문제 크기를 의도적으로 이전 퍼즐보다 작게 설정했습니다.

## 조사 과정

### Step 1: 정확성 검증

```bash
pixi shell -e nvidia
mojo problems/p32/p32.mojo --test
```

두 커널 모두 동일한 결과를 내야 합니다. 이를 통해 뱅크 충돌이 **정확성**이 아닌 **성능**에 영향을 미친다는 것을 확인합니다.

### Step 2: 성능 기준선 벤치마크

```bash
mojo problems/p32/p32.mojo --benchmark
```

실행 시간을 기록하세요. 워크로드가 글로벌 메모리 접근에 의해 지배되기 때문에 비슷한 성능이 나올 수 있지만, 뱅크 충돌은 프로파일링 메트릭을 통해 드러납니다.

### Step 3: 프로파일링용 빌드

```bash
mojo build --debug-level=full problems/p32/p32.mojo -o problems/p32/p32_profiler
```

### Step 4: 뱅크 충돌 프로파일링

NSight Compute를 사용하여 공유 메모리 뱅크 충돌을 정량적으로 측정합니다:

```bash
# Profile no-conflict kernel
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st problems/p32/p32_profiler --no-conflict

```

그리고

```bash
# Profile two-way conflict kernel
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st problems/p32/p32_profiler --two-way
```

**기록할 핵심 메트릭:**

- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` - 로드 충돌
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` - 스토어 충돌

### Step 5: 접근 패턴 분석

프로파일링 결과를 바탕으로 수학적 접근 패턴을 분석합니다:

**충돌 없는 커널 접근 패턴:**

```mojo
# Thread mapping: thread_idx.x directly maps to shared memory index
shared_buf[thread_idx.x]  # Thread 0→Index 0, Thread 1→Index 1, etc.
# Bank mapping: Index % 32 = Bank ID
# Result: Thread 0→Bank 0, Thread 1→Bank 1, ..., Thread 31→Bank 31
```

**2-way 충돌 커널 접근 패턴:**

```mojo
# Thread mapping with stride-2 modulo operation
shared_buf[(thread_idx.x * 2) % TPB]
# For threads 0-31: Index 0,2,4,6,...,62, then wraps to 64,66,...,126, then 0,2,4..
# Bank mapping examples:
# Thread 0  → Index 0   → Bank 0
# Thread 16 → Index 32  → Bank 0  (conflict!)
# Thread 1  → Index 2   → Bank 2
# Thread 17 → Index 34  → Bank 2  (conflict!)
```

## 도전 과제: 뱅크 충돌 미스터리를 풀어보세요

**위의 조사 단계를 완료한 후, 다음 분석 질문에 답하세요:**

### 성능 분석 (Step 1-2)

1. 두 커널이 동일한 수학적 결과를 내나요?
2. 커널 간 실행 시간 차이가 있나요?
3. 접근 패턴이 다른데도 성능이 비슷할 수 있는 이유는 무엇인가요?

### 뱅크 충돌 프로파일링 (Step 4)

1. 충돌 없는 커널은 로드와 스토어에서 몇 건의 뱅크 충돌을 발생시키나요?
2. 2-way 충돌 커널은 로드와 스토어에서 몇 건의 뱅크 충돌을 발생시키나요?
3. 두 커널 간 총 충돌 횟수 차이는 얼마인가요?

### 접근 패턴 분석 (Step 5)

1. 충돌 없는 커널에서 Thread 0은 어떤 뱅크에 접근하나요? Thread 31은?
2. 2-way 충돌 커널에서 Bank 0에 접근하는 스레드는? Bank 2에 접근하는 스레드는?
3. 충돌 커널에서 같은 뱅크를 놓고 경쟁하는 스레드는 몇 개인가요?

### 뱅크 충돌 탐정 작업

1. 충돌 없는 커널은 충돌이 0인데, 2-way 충돌 커널에서는 측정 가능한 충돌이 나타나는 이유는 무엇인가요?
2. stride-2 접근 패턴 `(thread_idx.x * 2) % TPB`는 어떻게 체계적인 충돌을 만들어내나요?
3. 뱅크 충돌이 메모리 바운드 커널보다 연산 집약적 커널에서 더 중요한 이유는 무엇인가요?

### 실전 시사점

1. 뱅크 충돌이 애플리케이션 성능에 큰 영향을 미칠 것으로 예상되는 경우는 언제인가요?
2. 공유 메모리 알고리즘을 구현하기 전에 뱅크 충돌 패턴을 어떻게 예측할 수 있나요?
3. 행렬 연산과 stencil 연산에서 뱅크 충돌을 피하는 데 도움이 되는 설계 원칙은 무엇인가요?

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

**뱅크 충돌 탐정 도구 모음:**

- **NSight Compute 메트릭** - 정밀한 측정으로 충돌을 정량화
- **접근 패턴 시각화** - 스레드 인덱스를 뱅크에 체계적으로 매핑
- **수학적 분석** - 모듈로 연산으로 충돌 예측
- **워크로드 특성** - 충돌이 중요한 경우와 그렇지 않은 경우 이해

**핵심 조사 원칙:**

- **체계적으로 측정하기:** 충돌을 추측하지 말고 프로파일링 도구를 사용
- **접근 패턴 시각화하기:** 복잡한 알고리즘의 스레드-뱅크 매핑을 그려보기
- **워크로드 맥락 고려하기:** 뱅크 충돌은 연산 집약적 공유 메모리 알고리즘에서 가장 중요
- **예방적으로 사고하기:** 처음부터 충돌 없는 접근 패턴으로 알고리즘 설계

**접근 패턴 분석 방법:**

1. **스레드를 인덱스에 매핑:** 수학적 주소 계산을 이해
2. **뱅크 할당 계산:** 공식 `bank_id = (address / 4) % 32` 사용
3. **충돌 식별:** 같은 뱅크에 접근하는 스레드가 여러 개인지 확인
4. **프로파일링으로 검증:** NSight Compute 측정으로 이론적 분석 확인

**일반적인 충돌 없는 패턴:**

- **순차 접근:** `shared[thread_idx.x]` - 각 스레드가 다른 뱅크에 접근
- **Broadcast 접근:** 모든 스레드가 `shared[0]` - 하드웨어 최적화
- **2의 거듭제곱 stride:** stride-32는 뱅킹 패턴에 깔끔하게 매핑되는 경우가 많음
- **패딩된 배열:** 패딩을 추가하여 문제가 되는 접근 패턴을 이동

</div>
</details>

## 풀이

<details class="solution-details">
<summary><strong>뱅크 충돌 분석이 포함된 완전한 풀이</strong></summary>

이 뱅크 충돌 탐정 사건은 공유 메모리 접근 패턴이 GPU 성능에 어떤 영향을 미치는지, 그리고 최적화를 위한 체계적 프로파일링의 중요성을 보여줍니다.

## **프로파일링을 통한 조사 결과**

**Step 1: 정확성 검증**
두 커널 모두 동일한 수학적 결과를 냅니다:

```
✅ No-conflict kernel: PASSED
✅ Two-way conflict kernel: PASSED
✅ Both kernels produce identical results
```

**Step 2: 성능 기준선**
벤치마크 결과는 비슷한 실행 시간을 보여줍니다:

```
| name             | met (ms)           | iters |
| ---------------- | ------------------ | ----- |
| no_conflict      | 2.1930616745886655 | 547   |
| two_way_conflict | 2.1978922967032966 | 546   |
```

**핵심 통찰:** 성능이 거의 동일한 이유(~2.19ms vs ~2.20ms)는 이 워크로드가 공유 메모리 바운드가 아닌 **글로벌 메모리 바운드**이기 때문입니다. 뱅크 충돌은 실행 시간이 아닌 프로파일링 메트릭을 통해 드러납니다.

## **뱅크 충돌 프로파일링 근거**

**충돌 없는 커널 (최적 접근 패턴):**

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum    0
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum    0
```

**결과:** 로드와 스토어 모두 충돌 0건 - 완벽한 공유 메모리 효율.

**2-Way 충돌 커널 (문제 있는 접근 패턴):**

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum    256
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum    256
```

**결과:** 로드와 스토어 각각 256건의 충돌 - 체계적인 뱅킹 문제의 명확한 근거.

**총 충돌 차이:** 512건의 충돌(256 + 256)이 측정 가능한 공유 메모리 비효율을 보여줍니다.

## **접근 패턴 수학적 분석**

### 충돌 없는 커널 접근 패턴

**스레드-인덱스 매핑:**

```mojo
shared_buf[thread_idx.x]
```

**뱅크 할당 분석:**

```
Thread 0  → Index 0   → Bank 0 % 32 = 0
Thread 1  → Index 1   → Bank 1 % 32 = 1
Thread 2  → Index 2   → Bank 2 % 32 = 2
...
Thread 31 → Index 31  → Bank 31 % 32 = 31
```

**결과:** 완벽한 뱅크 분배 - 각 Warp 내에서 각 스레드가 서로 다른 뱅크에 접근하여 병렬 접근이 가능합니다.

### 2-way 충돌 커널 접근 패턴

**스레드-인덱스 매핑:**

```mojo
shared_buf[(thread_idx.x * 2) % TPB]  # TPB = 256
```

**첫 번째 Warp(스레드 0-31)의 뱅크 할당 분석:**

```
Thread 0  → Index (0*2)%256 = 0   → Bank 0
Thread 1  → Index (1*2)%256 = 2   → Bank 2
Thread 2  → Index (2*2)%256 = 4   → Bank 4
...
Thread 16 → Index (16*2)%256 = 32 → Bank 0  ← Thread 0과 충돌
Thread 17 → Index (17*2)%256 = 34 → Bank 2  ← Thread 1과 충돌
Thread 18 → Index (18*2)%256 = 36 → Bank 4  ← Thread 2와 충돌
...
```

**충돌 패턴:** 각 뱅크가 정확히 2개의 스레드를 처리하여 32개 뱅크 전체에서 체계적인 2-way 충돌이 발생합니다.

**수학적 설명:** stride-2 패턴과 모듈로 256의 조합이 반복적인 접근 패턴을 만들어냅니다:

- 스레드 0-15는 뱅크 0,2,4,...,30에 접근
- 스레드 16-31은 **동일한 뱅크** 0,2,4,...,30에 접근
- 각 뱅크 충돌마다 하드웨어 직렬화가 필요

## **이것이 중요한 이유: 워크로드 맥락 분석**

### 메모리 바운드 vs 연산 바운드 시사점

**이 워크로드의 특성:**

- **글로벌 메모리 지배적:** 각 스레드가 메모리 전송 대비 최소한의 연산만 수행
- **공유 메모리는 부차적:** 뱅크 충돌이 오버헤드를 추가하지만 전체 실행 시간을 지배하지는 않음
- **동일한 성능:** 글로벌 메모리 대역폭 포화가 공유 메모리 비효율을 가림

**뱅크 충돌이 가장 중요한 경우:**

1. **연산 집약적 공유 메모리 알고리즘** - 행렬 곱셈, stencil 연산, FFT
2. **타이트한 연산 루프** - 내부 루프 안에서 반복적인 공유 메모리 접근
3. **높은 산술 강도** - 메모리 접근당 상당한 연산량
4. **대규모 공유 메모리 작업 세트** - 공유 메모리 캐싱을 집중적으로 활용하는 알고리즘

### 실전 성능 시사점

**뱅크 충돌이 성능에 큰 영향을 미치는 애플리케이션:**

**행렬 곱셈:**

```mojo
# Problematic: All threads in warp access same column
for k in range(tile_size):
    acc += a_shared[local_row, k] * b_shared[k, local_col]  # b_shared[k, 0] conflicts
```

**Stencil 연산:**

```mojo
# Problematic: Stride access in boundary handling
shared_buf[thread_idx.x * stride]  # Creates systematic conflicts
```

**병렬 Reduction:**

```mojo
# Problematic: Power-of-2 stride patterns
if thread_idx.x < stride:
    shared_buf[thread_idx.x] += shared_buf[thread_idx.x + stride]  # Conflict potential
```

## **충돌 없는 설계 원칙**

### 예방 전략

**1. 순차 접근 패턴:**

```mojo
shared[thread_idx.x]  # Optimal - each thread different bank
```

**2. Broadcast 최적화:**

```mojo
constant = shared[0]  # All threads read same address - hardware optimized
```

**3. 패딩 기법:**

```mojo
shared = LayoutTensor[dtype, Layout.row_major(TPB + 1), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()  # Shift access patterns
```

**4. 접근 패턴 분석:**

- 구현 전에 뱅크 할당을 계산
- 모듈로 연산 사용: `bank_id = (address_bytes / 4) % 32`
- 복잡한 알고리즘의 스레드-뱅크 매핑을 시각화

### 체계적 최적화 워크플로우

**설계 단계:**

1. **접근 패턴 계획** - 스레드-메모리 매핑을 스케치
2. **뱅크 할당 계산** - 수학적 분석 활용
3. **충돌 예측** - 문제가 되는 접근 패턴 식별
4. **대안 설계** - 패딩, 전치, 또는 알고리즘 변경 고려

**구현 단계:**

1. **체계적 프로파일링** - NSight Compute 충돌 메트릭 사용
2. **영향 측정** - 구현 간 충돌 횟수 비교
3. **성능 검증** - 최적화가 종단간 성능을 개선하는지 확인
4. **패턴 문서화** - 성공적인 충돌 없는 알고리즘을 재사용을 위해 기록

## **핵심 정리: 탐정 작업에서 최적화 전문성으로**

**뱅크 충돌 조사에서 밝혀진 것:**

1. **측정이 직관보다 낫다** - 프로파일링 도구가 성능 타이밍으로는 보이지 않는 충돌을 드러냄
2. **패턴 분석이 유효하다** - 수학적 예측이 NSight Compute 결과와 정확히 일치
3. **맥락이 중요하다** - 뱅크 충돌은 연산 집약적 공유 메모리 워크로드에서 가장 중요
4. **예방이 수정보다 낫다** - 충돌 없는 패턴을 설계하는 것이 사후 최적화보다 쉬움

**보편적인 공유 메모리 최적화 원칙:**

**뱅크 충돌에 주의해야 하는 경우:**

- 데이터 재사용을 위해 공유 메모리를 사용하는 **연산 집약적 커널**
- 타이트한 루프에서 반복적으로 공유 메모리에 접근하는 **반복 알고리즘**
- 모든 사이클이 중요한 **성능 핵심 코드**
- 대역폭 바운드가 아닌 연산 바운드인 **메모리 집약적 연산**

**뱅크 충돌이 덜 중요한 경우:**

- 글로벌 메모리가 성능을 지배하는 **메모리 바운드 워크로드**
- 공유 메모리 재사용이 최소인 **단순 캐싱 시나리오**
- 반복적인 충돌 발생 연산이 없는 **일회성 접근 패턴**

**전문적 개발 방법론:**

1. **최적화 전에 프로파일링** - NSight Compute로 충돌을 정량적으로 측정
2. **접근 수학 이해** - 뱅크 할당 공식으로 문제를 예측
3. **체계적으로 설계** - 뱅킹을 사후 고려가 아닌 알고리즘 설계 단계에서 고려
4. **최적화 검증** - 충돌 감소가 실제 성능을 개선하는지 확인

이 탐정 사건은 **체계적 프로파일링이 성능 타이밍만으로는 보이지 않는 최적화 기회를 드러낸다**는 것을 보여줍니다 - 뱅크 충돌은 측정 기반 최적화가 추측보다 나은 대표적인 사례입니다.

</details>
