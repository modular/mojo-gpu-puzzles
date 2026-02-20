<!-- i18n-source-commit: 682d932e25a4853e788236c40c9d23d6e9ec64ab -->

# 📚 공유 메모리 뱅크 이해하기

## 지금까지 배운 것을 바탕으로

GPU 최적화 여정에서 이미 많은 길을 걸어왔습니다. [Puzzle 8](../puzzle_08/puzzle_08.md)에서는 공유 메모리가 글로벌 메모리보다 훨씬 빠른 블록 내부 저장소를 제공한다는 것을 배웠습니다. [Puzzle 16](../puzzle_16/puzzle_16.md)에서는 행렬 곱셈 커널이 공유 메모리를 사용하여 데이터 타일을 캐싱하고, 비용이 큰 글로벌 메모리 접근을 줄이는 방법을 확인했습니다.

하지만 공유 메모리에는 병렬 연산을 직렬화시킬 수 있는 숨겨진 성능 함정이 도사리고 있습니다: **뱅크 충돌**.

**성능 미스터리:** 겉보기에 동일한 방식으로 공유 메모리에 접근하는 두 커널을 작성할 수 있습니다 - 둘 다 같은 양의 데이터를 사용하고, 완벽한 점유율을 가지며, 경쟁 상태도 없습니다. 그런데 하나가 다른 것보다 32배 느립니다. 범인은? 스레드가 공유 메모리 뱅크에 접근하는 방식입니다.

## 공유 메모리 뱅크란?

공유 메모리를 **뱅크**라고 불리는 32개의 독립적인 메모리 유닛의 집합이라고 생각하세요. 각 뱅크는 클록 사이클당 하나의 메모리 요청을 처리할 수 있습니다. 이 뱅킹 시스템이 존재하는 근본적인 이유는 **하드웨어 병렬성** 때문입니다.

32개 스레드로 구성된 Warp가 동시에 공유 메모리에 접근해야 할 때, **각 스레드가 서로 다른 뱅크에 접근한다면** GPU는 32개의 요청을 모두 병렬로 처리할 수 있습니다. 여러 스레드가 같은 뱅크에 접근하려 하면 하드웨어는 이를 **직렬화**해야 하므로, 1사이클이면 될 연산이 여러 사이클로 늘어납니다.

### 뱅크 주소 매핑

공유 메모리의 각 4바이트 워드는 다음 공식에 따라 특정 뱅크에 배정됩니다:

```
bank_id = (byte_address / 4) % 32
```

공유 메모리의 처음 128바이트가 뱅크에 매핑되는 방식은 다음과 같습니다:

| Address Range | Bank ID | Example `float32` Elements |
|---------------|---------|---------------------------|
| 0-3 bytes     | Bank 0  | `shared[0]` |
| 4-7 bytes     | Bank 1  | `shared[1]` |
| 8-11 bytes    | Bank 2  | `shared[2]` |
| ...           | ...     | ... |
| 124-127 bytes | Bank 31 | `shared[31]` |
| 128-131 bytes | Bank 0  | `shared[32]` |
| 132-135 bytes | Bank 1  | `shared[33]` |

**핵심 통찰:** `float32` 배열에서 뱅킹 패턴은 32개 요소마다 반복되며, 이는 32개 스레드로 구성된 Warp 크기와 정확히 일치합니다. 이것은 우연이 아닙니다 - 최적의 병렬 접근을 위해 설계된 것입니다.

## 뱅크 충돌의 유형

### 충돌 없음: 이상적인 경우

Warp 내 각 스레드가 서로 다른 뱅크에 접근하면 32개의 접근이 모두 1사이클에 완료됩니다:

```mojo
# Perfect case: each thread accesses a different bank
shared[thread_idx.x]  # Thread 0→Bank 0, Thread 1→Bank 1, ..., Thread 31→Bank 31
```

**결과:** 32개 병렬 접근, 총 1사이클

### N-way 뱅크 충돌

N개의 스레드가 같은 뱅크의 서로 다른 주소에 접근하면 하드웨어가 접근을 직렬화합니다:

```mojo
# 2-way conflict: stride-2 access pattern
shared[thread_idx.x * 2]  # Thread 0,16→Bank 0; Thread 1,17→Bank 1; etc.
```

**결과:** 뱅크당 2회 접근, 총 2사이클 (효율 50%)

```mojo
# Worst case: all threads access different addresses in Bank 0
shared[thread_idx.x * 32]  # All threads→Bank 0
```

**결과:** 32회 직렬화된 접근, 총 32사이클 (효율 3%)

### Broadcast 예외

충돌 규칙에는 한 가지 중요한 예외가 있습니다: **broadcast 접근**. 모든 스레드가 **동일한 주소**를 읽으면 하드웨어가 이를 단일 메모리 접근으로 최적화합니다:

```mojo
# Broadcast: all threads read the same value
constant = shared[0]  # All threads read shared[0]
```

**결과:** 1회 접근으로 32개 스레드에 broadcast, 총 1사이클

이 최적화가 존재하는 이유는 broadcast가 흔한 패턴(상수 로딩, reduction 연산 등)이고, 하드웨어가 추가 메모리 대역폭 없이 단일 값을 모든 스레드에 복제할 수 있기 때문입니다.

## 뱅크 충돌이 중요한 이유

### 성능 영향

뱅크 충돌은 공유 메모리 접근 시간을 직접적으로 배가시킵니다:

| 충돌 유형 | 접근 시간 | 효율 | 성능 영향 |
|-----------|-----------|------|-----------|
| 충돌 없음 | 1사이클 | 100% | 기준선 |
| 2-way conflict | 2사이클 | 50% | 2배 느림 |
| 4-way conflict | 4사이클 | 25% | 4배 느림 |
| 32-way conflict | 32사이클 | 3% | **32배 느림** |

### 실전 맥락

[Puzzle 30](../puzzle_30/puzzle_30.md)에서 메모리 접근 패턴이 극적인 성능 차이를 만들어낸다는 것을 배웠습니다. 뱅크 충돌은 이 원리가 공유 메모리 수준에서 작동하는 또 다른 사례입니다.

글로벌 메모리 병합이 DRAM 대역폭 활용에 영향을 주는 것처럼, 뱅크 충돌은 공유 메모리 처리량에 영향을 줍니다. 차이는 규모에 있습니다: 글로벌 메모리 latency는 수백 사이클이지만, 공유 메모리 충돌은 접근당 몇 사이클만 추가합니다. 그러나 공유 메모리를 집중적으로 사용하는 연산 집약적 커널에서는 이 "몇 사이클"이 빠르게 누적됩니다.

### Warp 실행과의 관계

[Puzzle 24](../puzzle_24/puzzle_24.md)에서 Warp가 SIMT(Single Instruction, Multiple Thread) 방식으로 실행된다는 것을 배웠습니다. Warp가 뱅크 충돌에 부딪히면 직렬화된 메모리 접근이 완료될 때까지 **32개 스레드 모두가 대기**해야 합니다. 이 대기 시간은 충돌을 일으킨 스레드만이 아니라 Warp 전체의 진행에 영향을 미칩니다.

이는 [Puzzle 31](../puzzle_31/puzzle_31.md)의 점유율 개념과 연결됩니다: 뱅크 충돌은 Warp가 메모리 latency를 효과적으로 숨기는 것을 방해하여, 높은 점유율의 실질적인 이점을 줄일 수 있습니다.

## 뱅크 충돌 감지하기

### 시각적 패턴 인식

접근 패턴을 분석하면 뱅크 충돌을 예측할 수 있는 경우가 많습니다:

**순차 접근 (충돌 없음):**

```mojo
# Thread ID:  0  1  2  3  ...  31
# Address:    0  4  8 12  ... 124
# Bank:       0  1  2  3  ...  31  ✅ All different banks
```

**Stride-2 접근 (2-way conflict):**

```mojo
# Thread ID:  0  1  2  3  ...  15 16 17 18 ... 31
# Address:    0  8 16 24  ... 120  4 12 20 ... 124
# Bank:       0  2  4  6  ...  30  1  3  5 ...  31
# Conflict:   Banks 0,2,4... have 2 threads each  ❌
```

**Stride-32 접근 (32-way conflict):**

```mojo
# Thread ID:  0   1   2   3  ...  31
# Address:    0  128 256 384 ... 3968
# Bank:       0   0   0   0  ...   0  ❌ All threads→Bank 0
```

### NSight Compute(`ncu`)를 사용한 프로파일링

[Puzzle 30](../puzzle_30/puzzle_30.md)에서 배운 프로파일링 방법론을 바탕으로, 뱅크 충돌을 정량적으로 측정할 수 있습니다:

```bash
# Key metrics for shared memory bank conflicts
ncu --metrics=l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st your_kernel

# Additional context metrics
ncu --metrics=smsp__sass_average_branch_targets_threads_uniform.pct your_kernel
ncu --metrics=smsp__warps_issue_stalled_membar_per_warp_active.pct your_kernel
```

`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`와 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` 메트릭은 커널 실행 중 로드 및 스토어 연산의 뱅크 충돌 횟수를 직접 카운트합니다. 공유 메모리 접근 횟수와 결합하면 충돌 비율을 구할 수 있으며, 이는 핵심적인 성능 지표입니다.

## 뱅크 충돌이 가장 중요한 경우

### 연산 집약적 커널

뱅크 충돌은 다음과 같은 커널에서 가장 큰 영향을 미칩니다:

- 타이트한 루프 안에서 공유 메모리에 자주 접근하는 경우
- 공유 메모리 접근당 연산량이 적은 경우
- 커널이 메모리 바운드가 아닌 연산 바운드인 경우

**대표적인 시나리오:**

- 행렬 곱셈 내부 루프 ([Puzzle 16](../puzzle_16/puzzle_16.md)의 tiled 버전과 같은)
- 공유 메모리 캐싱을 사용하는 stencil 연산
- 병렬 reduction 연산

### 메모리 바운드 vs 연산 바운드 트레이드오프

[Puzzle 31](../puzzle_31/puzzle_31.md)에서 메모리 바운드 워크로드에서는 점유율이 덜 중요하다는 것을 보았듯이, 커널이 글로벌 메모리 대역폭에 병목이 걸리거나 산술 강도가 매우 낮은 경우에는 뱅크 충돌의 영향도 줄어듭니다.

그러나 공유 메모리를 사용하는 많은 커널은 바로 메모리 바운드에서 연산 바운드로 전환하기 **위해** 공유 메모리를 활용합니다. 이런 경우 뱅크 충돌은 애초에 공유 메모리를 도입한 이유였던 성능 향상을 달성하지 못하게 만들 수 있습니다.

## 앞으로의 방향

공유 메모리 뱅킹을 이해하면 다음과 같은 기초를 갖추게 됩니다:

1. 접근 패턴을 분석하여 코드를 작성하기 전에 **성능을 예측**
2. 체계적인 프로파일링 접근법으로 **성능 저하를 진단**
3. 높은 공유 메모리 처리량을 유지하는 **충돌 없는 알고리즘 설계**
4. 알고리즘 복잡도와 메모리 효율 사이의 **균형 잡힌 판단**

다음 섹션에서는 이 지식을 실습에 적용하여 일반적인 충돌 패턴과 해결책을 직접 다뤄봅니다 - 이론적 이해를 실전 최적화 역량으로 바꾸는 과정입니다.
