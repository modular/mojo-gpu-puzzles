<!-- i18n-source-commit: 9c7176b81f278a6e8efa26c92005c139967c0c27 -->

# 🏁 경쟁 상태 디버깅

## 개요

NVIDIA `compute-sanitizer`를 사용해 잘못된 결과를 일으키는 경쟁 상태를 식별하면서 실패하는 GPU 프로그램을 디버깅합니다. 공유 메모리 연산에서 동시성 버그를 찾는 `racecheck` 도구 사용법을 배웁니다.

공유 메모리로 여러 스레드의 값을 누적해야 하는 GPU Kernel이 있습니다. 테스트는 실패하는데, 로직은 올바른 것 같습니다. 당신의 과제는 실패를 일으키는 경쟁 상태를 찾아 수정하는 것입니다.

## 구성

```mojo
comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)  # 9개 스레드 중 4개만 활성화
comptime dtype = DType.float32
```

## 실패하는 Kernel

```mojo
{{#include ../../../../../problems/p10/p10.mojo:shared_memory_race}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p10/p10.mojo" class="filename">전체 파일 보기: problems/p10/p10.mojo</a>

## 코드 실행

```bash
pixi run p10 --race-condition
```

출력은 다음과 같습니다

```txt
out shape: 2 x 2
Running race condition example...
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
Unhandled exception caught during execution: At /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p10/p10.mojo:122:33: AssertionError: `left == right` comparison failed:
   left: 0.0
  right: 6.0
```

`compute-sanitizer`가 GPU 코드의 문제를 어떻게 찾아내는지 살펴봅시다.

## `compute-sanitizer`로 디버깅하기

### 1단계: `racecheck`로 경쟁 상태 식별

`compute-sanitizer`와 `racecheck` 도구를 사용하여 경쟁 상태를 식별합니다:

```bash
pixi run compute-sanitizer --tool racecheck mojo problems/p10/p10.mojo --race-condition
```

출력은 다음과 같습니다

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running race condition example...
========= Error: Race reported between Write access at p10_shared_memory_race_...+0x140
=========     and Read access at p10_shared_memory_race_...+0xe0 [4 hazards]
=========     and Write access at p10_shared_memory_race_...+0x140 [5 hazards]
=========
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
AssertionError: `left == right` comparison failed:
  left: 0.0
  right: 6.0
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
```

**분석**: 프로그램에 **1개의 경쟁 상태**와 **9개의 개별 위험 요소**가 있습니다:

- **4개의 read-after-write 위험** (다른 스레드가 쓰는 동안 읽기)
- **5개의 write-after-write 위험** (여러 스레드가 동시에 쓰기)

### 2단계: `synccheck`와 비교

동기화 문제가 아닌 경쟁 상태인지 확인합니다:

```bash
pixi run compute-sanitizer --tool synccheck mojo problems/p10/p10.mojo --race-condition
```

출력은 다음과 같습니다

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running race condition example...
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
AssertionError: `left == right` comparison failed:
  left: 0.0
  right: 6.0
========= ERROR SUMMARY: 0 errors
```

**핵심 통찰**: `synccheck`가 **0개의 오류**를 찾았습니다 - 교착 상태 같은 동기화 문제는 없습니다. 문제는 동기화 버그가 아닌 **경쟁 상태**입니다.

## 교착 상태 vs 경쟁 상태: 차이점 이해하기

| 측면 | 교착 상태 | Race Condition (경쟁 상태) |
|------|----------|----------|
| **증상** | 프로그램이 영원히 멈춤 | 프로그램이 잘못된 결과 생성 |
| **실행** | 완료되지 않음 | 성공적으로 완료됨 |
| **타이밍** | 결정적으로 멈춤 | 비결정적 결과 |
| **근본 원인** | 동기화 로직 오류 | 동기화되지 않은 데이터 접근 |
| **탐지 도구** | `synccheck` | `racecheck` |
| **예시** | [Puzzle 09: 세 번째 사례](../puzzle_09/third_case.md) barrier 교착 상태 | 공유 메모리 `+=` 연산 |

**우리 사례에서:**

- **프로그램 완료됨** → 교착 상태 없음 (스레드가 멈추지 않음)
- **잘못된 결과** → 경쟁 상태 (스레드들이 서로의 데이터를 손상)
- **도구 확인** → `synccheck`는 0개 오류, `racecheck`는 9개 위험 요소 보고

**디버깅에서 이 구분이 중요한 이유:**

- **교착 상태 디버깅**: barrier 배치, 조건부 동기화, 스레드 조율에 집중
- **경쟁 상태 디버깅**: 공유 메모리 접근 패턴, atomic 연산 (_역주: 중간 상태 없이 완전히 실행되거나 전혀 실행되지 않는 연산_), 데이터 의존성에 집중

## 도전 과제

이 도구들을 활용하여 실패하는 Kernel을 수정하세요.

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 위험 요소 분석

`shared_sum[0] += a[row, col]` 연산이 위험한 이유는 실제로 **세 개의 별도 메모리 연산**이기 때문입니다:

1. `shared_sum[0]` **읽기**
2. 읽은 값에 `a[row, col]` **더하기**
3. 결과를 `shared_sum[0]`에 다시 **쓰기**

4개의 활성 스레드(위치 (0,0), (0,1), (1,0), (1,1))에서 이 연산들이 겹칠 수 있습니다:

- **스레드 타이밍 중첩** → 여러 스레드가 같은 초기값(0.0)을 읽음
- **업데이트 손실** → 각 스레드가 `0.0 + 자신의_값`을 써서 다른 스레드의 작업을 덮어씀
- **Non-atomic 연산** → `+=` 복합 대입은 GPU 공유 메모리에서 atomic하지 않음 (_역주: 실행 도중 다른 스레드가 끼어들 수 있어 중간 상태가 노출됨_)

**정확히 9개의 위험 요소가 나오는 이유:**

- 각 스레드가 read-modify-write를 시도
- 4개 스레드 × 스레드당 2-3개 위험 요소 = 총 9개 위험 요소
- `compute-sanitizer`가 모든 충돌하는 메모리 접근 쌍을 추적

### 경쟁 상태 디버깅 팁

1. **데이터 경쟁에는 racecheck 사용**: 공유 메모리 위험 요소와 데이터 손상 탐지
2. **교착 상태에는 synccheck 사용**: 동기화 버그(barrier 문제, 교착 상태) 탐지
3. **공유 메모리 접근에 집중**: 공유 변수에 대한 동기화되지 않은 `+=`, `=` 연산 찾기
4. **패턴 식별**: read-modify-write 연산이 흔한 경쟁 상태 원인
5. **Barrier 배치 확인**: barrier는 충돌 연산 **이전에** 배치해야 함, 이후가 아님

**디버깅에서 이 구분이 중요한 이유:**

- **교착 상태 디버깅**: barrier 배치, 조건부 동기화, 스레드 조율에 집중
- **경쟁 상태 디버깅**: 공유 메모리 접근 패턴, atomic 연산, 데이터 의존성에 집중

**피해야 할 흔한 경쟁 상태 패턴:**

- 여러 스레드가 같은 공유 메모리 위치에 쓰기
- 동기화되지 않은 read-modify-write 연산 (`+=`, `++` 등)
- 경쟁 상태 이전이 아닌 이후에 barrier 배치

</div>
</details>

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p10/p10.mojo:shared_memory_race_solution}}
```

<div class="solution-explanation">

### 무엇이 잘못되었는지 이해하기

#### 경쟁 상태 문제 패턴

원래 실패하는 코드에는 이 핵심적인 줄이 있었습니다:

```mojo
shared_sum[0] += a[row, col]  # 경쟁 상태!
```

이 한 줄이 4개의 유효한 스레드 사이에서 여러 위험 요소를 만듭니다:

1. **스레드 (0,0)이 읽음** `shared_sum[0]` (값: 0.0)
2. **스레드 (0,1)이 읽음** `shared_sum[0]` (값: 0.0) ← **Read-after-write 위험!**
3. **스레드 (0,0)이 씀** `0.0 + 0`
4. **스레드 (1,0)이 씀** `0.0 + 2` ← **Write-after-write 위험!**

#### 테스트가 실패한 이유

- `+=` 연산 중 여러 스레드가 서로의 쓰기를 손상시킴
- `+=` 연산이 중단되어 업데이트 손실 발생
- 예상 합계 6.0 (0+1+2+3)이지만, 경쟁 상태로 인해 0.0이 됨
- `barrier()`가 너무 늦게 옴 - 경쟁 상태가 이미 발생한 후

#### 경쟁 상태란?

**경쟁 상태**는 여러 스레드가 공유 데이터에 동시에 접근하고, 결과가 예측 불가능한 스레드 실행 타이밍에 따라 달라질 때 발생합니다.

**주요 특성:**

- **비결정적 동작**: 같은 코드가 다른 실행에서 다른 결과를 낼 수 있음
- **타이밍 의존적**: 결과가 어떤 스레드가 "경쟁에서 이기는지"에 따라 달라짐
- **재현하기 어려움**: 특정 조건이나 하드웨어에서만 나타날 수 있음

#### GPU 특유의 위험성

**대규모 병렬 처리의 영향:**

- **Warp 수준 손상**: 경쟁 상태가 전체 Warp(32개 스레드)에 영향을 줄 수 있음
- **메모리 병합(coalescing) 문제**: 경쟁으로 효율적인 메모리 접근 패턴이 깨질 수 있음
- **Kernel 전체 실패**: 공유 메모리 손상이 전체 GPU Kernel에 영향을 줄 수 있음

**하드웨어 차이:**

- **다른 GPU 아키텍처**: 경쟁 상태가 GPU 모델마다 다르게 나타날 수 있음
- **메모리 계층**: L1 캐시, L2 캐시, 글로벌 메모리가 각각 다른 경쟁 동작을 보일 수 있음
- **Warp 스케줄링**: 다른 스레드 스케줄링이 다른 경쟁 상태 시나리오를 노출시킬 수 있음

### 전략: 단일 Writer 패턴

핵심은 공유 메모리에 대한 동시 쓰기를 없애는 것입니다:

1. **Single writer**: 하나의 스레드(위치 (0,0))만 모든 누적 작업 수행
2. **로컬 누적**: 위치 (0,0) 스레드가 로컬 변수를 사용해 반복적인 공유 메모리 접근을 피함
3. **단일 공유 메모리 쓰기**: 단일 쓰기 연산으로 write-write 경쟁 제거
4. **Barrier 동기화**: writer가 완료된 후에야 다른 스레드가 읽도록 보장
5. **다중 읽기**: 모든 스레드가 안전하게 최종 결과를 읽음

#### 단계별 솔루션 분석

**1단계: 스레드 식별**

```mojo
if row == 0 and col == 0:
```

직접 좌표 검사로 위치 (0,0)의 스레드를 식별합니다.

**2단계: 단일 스레드 누적**

```mojo
if row == 0 and col == 0:
    local_sum = Scalar[dtype](0.0)
    for r in range(size):
        for c in range(size):
            local_sum += rebind[Scalar[dtype]](a[r, c])
    shared_sum[0] = local_sum  # 단일 쓰기 연산
```

위치 (0,0)의 스레드만 모든 누적 작업을 수행하여 경쟁 상태를 제거합니다.

**3단계: 동기화 barrier**

```mojo
barrier()  # 스레드 (0,0)이 완료한 후 다른 스레드가 읽도록 보장
```

모든 스레드가 위치 (0,0)의 스레드가 누적을 마칠 때까지 기다립니다.

**4단계: 안전한 병렬 읽기**

```mojo
if row < size and col < size:
    output[row, col] = shared_sum[0]
```

동기화 후 모든 스레드가 안전하게 결과를 읽을 수 있습니다.

### 효율성에 관한 중요 사항

**이 솔루션은 효율성보다 정확성을 우선합니다**. 경쟁 상태는 제거하지만, 위치 (0,0) 스레드만 누적에 사용하는 것은 GPU 성능에 **최적이 아닙니다** - 대규모 병렬 장치에서 사실상 직렬 계산을 하는 셈입니다.

**이어서 [Puzzle 11: Pooling](../../puzzle_11/puzzle_11.md)에서**: **모든 스레드**를 활용해 고성능 합산 연산을 수행하면서도 경쟁 상태를 피하는 효율적인 병렬 reduction 알고리즘 (_역주: 여러 값을 트리 구조로 병렬 합산하여 O(log n) 단계에 결과를 얻는 기법_)을 배웁니다. 이 퍼즐은 **정확성 우선**의 기초를 가르칩니다 - 경쟁 상태를 피하는 방법을 이해하고 나면, Puzzle 11에서 **정확성과 성능 모두**를 달성하는 방법을 보게 됩니다.

### 검증

```bash
pixi run compute-sanitizer --tool racecheck mojo solutions/p10/p10.mojo --race-condition
```

**예상 출력:**

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running race condition example...
out: HostBuffer([6.0, 6.0, 6.0, 6.0])
expected: HostBuffer([6.0, 6.0, 6.0, 6.0])
✅ Race condition test PASSED! (racecheck will find hazards)
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

**✅ 성공:** 테스트가 통과하고 경쟁 상태가 탐지되지 않았습니다!

</div>
</details>
