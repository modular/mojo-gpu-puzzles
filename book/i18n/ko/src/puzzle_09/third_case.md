<!-- i18n-source-commit: 1cd13cbe87682d50679d452938efab4cc79ddb78 -->

# 🕵 탐정 수사: 세 번째 사례

## 개요

[메모리 크래시](./first_case.md)와 [로직 버그](./second_case.md) 디버깅을 익혔습니다. 이제 GPU 디버깅의 최종 보스에 도전합니다: 프로그램이 무한정 멈춰버리는 **barrier 교착 상태**. 오류 메시지도, 잘못된 결과도 없이 - 그저 끝없는 침묵만 있습니다.

**디버깅 여정의 완결:**

- **[첫 번째 사례](./first_case.md)**: 프로그램 크래시 → 오류 신호 추적 → 메모리 버그 발견
- **[두 번째 사례](./second_case.md)**: 잘못된 결과 출력 → 패턴 분석 → 로직 버그 발견
- **[세 번째 사례]**: 프로그램 무한 정지 → 스레드 상태 조사 → 조율 버그 발견

이 고급 디버깅 챌린지에서는 공유 메모리, LayoutTensor 연산, barrier 동기화가 얽힌 **스레드 조율 실패**를 조사하는 방법을 배웁니다 - 이전 사례들에서 익힌 체계적인 조사 기술을 총동원합니다.

**사전 준비**: [Mojo GPU 디버깅의 핵심](./essentials.md), [탐정 수사: 첫 번째 사례](./first_case.md), [탐정 수사: 두 번째 사례](./second_case.md)를 먼저 완료해서 CUDA-GDB 워크플로우, 변수 검사의 한계, 체계적인 디버깅 접근법을 이해하세요. 아래 설정 명령을 실행했는지 확인하세요:

```bash
pixi run -e nvidia setup-cuda-gdb
```

## 핵심 개념

이번 디버깅 챌린지에서 배울 내용:

- **Barrier 교착 상태 탐지**: 스레드들이 동기화 지점에서 영원히 기다리게 되는 상황 식별하기
- **공유 메모리 조율**: LayoutTensor를 사용한 스레드 협력 패턴 이해하기
- **조건부 실행 분석**: 일부 스레드가 다른 코드 경로를 탈 때 디버깅하기
- **스레드 조율 디버깅**: CUDA-GDB로 다중 스레드 동기화 실패 분석하기

## 코드 실행

먼저 전체 코드를 보지 않고 kernel만 살펴봅시다:

```mojo
{{#include ../../../../../problems/p09/p09.mojo:third_crash}}
```

버그를 직접 경험하려면 터미널에서 다음 명령을 실행하세요 (`pixi` 전용):

```bash
pixi run -e nvidia p09 --third-case
```

다음과 같은 출력이 나타납니다 - **프로그램이 무한정 멈춥니다**:

```txt
Third Case: Advanced collaborative filtering with shared memory...
WARNING: This may hang - use Ctrl+C to stop if needed

Input array: [1, 2, 3, 4]
Applying collaborative filter using shared memory...
Each thread cooperates with neighbors for smoothing...
Waiting for GPU computation to complete...
[HANGS FOREVER - Use Ctrl+C to stop]
```

⚠️ **경고**: 이 프로그램은 멈춰서 완료되지 않습니다. `Ctrl+C`로 중단하세요.

## 과제: 탐정 수사

**도전**: 프로그램이 정상적으로 시작되지만 GPU 연산 중에 멈춰서 결과를 반환하지 않습니다. 코드를 보지 않은 상태에서, 이 교착 상태를 조사하기 위한 체계적인 접근법은 무엇일까요?

**생각해볼 점:**

- GPU kernel이 영영 완료되지 않게 만드는 원인은 무엇일까요?
- 스레드 조율 문제를 어떻게 조사하시겠습니까?
- 오류 메시지 없이 프로그램이 그냥 "멈춰버릴" 때 어떤 디버깅 전략이 통할까요?
- 스레드들이 제대로 협력하지 않을 수도 있다면 어떻게 디버깅할까요?
- 체계적 조사([첫 번째 사례](./first_case.md))와 실행 흐름 분석([두 번째 사례](./second_case.md))을 결합해서 조율 실패를 어떻게 디버깅할 수 있을까요?

다음 명령으로 시작해 보세요:

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --third-case
```

### GDB 명령어 단축키 (빠른 디버깅)

**이 단축키들**을 사용하면 디버깅 세션 속도를 높일 수 있습니다:

| 단축 | 전체 | 사용 예시 |
|------|------|-----------|
| `r` | `run` | `(cuda-gdb) r` |
| `n` | `next` | `(cuda-gdb) n` |
| `c` | `continue` | `(cuda-gdb) c` |
| `b` | `break` | `(cuda-gdb) b 62` |
| `p` | `print` | `(cuda-gdb) p thread_id` |
| `q` | `quit` | `(cuda-gdb) q` |

**아래 모든 디버깅 명령은 효율성을 위해 단축키를 사용합니다!**

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. **소리 없는 멈춤 조사** - 오류 메시지 없이 프로그램이 멈춰버릴 때, GPU 의 어떤 기본 요소가 무한 대기를 일으킬 수 있을까요?
2. **스레드 상태 검사** - `info cuda threads`로 서로 다른 스레드들이 어디서 멈췄는지 확인하세요
3. **조건부 실행 분석** - 어떤 스레드가 어떤 코드 경로를 실행하는지 확인하세요 (모든 스레드가 같은 경로를 따르나요?)
4. **동기화 지점 조사** - 스레드들이 조율해야 할 수도 있는 지점을 찾으세요
5. **스레드 분기 탐지** - 모든 스레드가 같은 프로그램 위치에 있나요, 아니면 일부는 다른 곳에 있나요?
6. **조율 기본 요소 분석** - 모든 스레드가 같은 동기화 연산에 참여하지 않으면 어떻게 될까요?
7. **실행 흐름 추적** - 각 스레드가 조건문을 통해 어떤 경로를 따라가는지 추적하세요
8. **스레드 ID 영향 분석** - 서로 다른 스레드 ID가 어떤 코드 경로를 실행할지 어떻게 영향을 미치나요?

</div>
</details>

<details class="solution-details">
<summary><strong>💡 조사 과정과 해결책</strong></summary>

<div class="solution-explanation">

## CUDA-GDB로 단계별 조사

### 1단계: 실행과 초기 설정

#### Step 1: 디버거 실행

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --third-case
```

#### Step 2: 정지 현상 분석

디버깅에 들어가기 전에 알고 있는 정보를 정리합니다:

```txt
기대값: 프로그램이 완료되고 필터링된 결과 표시
실제: "Waiting for GPU computation to complete..."에서 멈춤
```

**🔍 초기 가설**: GPU kernel이 교착 상태에 빠짐 - 어떤 동기화 기본 요소가 스레드들을 영원히 대기시키고 있습니다.

### 2단계: Kernel 진입

#### Step 3: 실행 및 kernel 진입 관찰

```bash
(cuda-gdb) r
Starting program: .../mojo run problems/p09/p09.mojo --third-case

Third Case: Advanced collaborative filtering with shared memory...
WARNING: This may hang - use Ctrl+C to stop if needed

Input array: [1, 2, 3, 4]
Applying collaborative filter using shared memory...
Each thread cooperates with neighbors for smoothing...
Waiting for GPU computation to complete...

[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

CUDA thread hit application kernel entry function breakpoint, p09_collaborative_filter_Orig6A6AcB6A6A_1882ca334fc2d34b2b9c4fa338df6c07<<<(1,1,1),(4,1,1)>>> (
    output=..., a=...)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:56
56          a: LayoutTensor[mut=False, dtype, vector_layout],
```

**🔍 주요 관찰**:

- **Grid**: (1,1,1) - 단일 블록
- **Block**: (4,1,1) - 총 4개 스레드 (0, 1, 2, 3)
- **현재 스레드**: (0,0,0) - 스레드 0 디버깅 중
- **함수**: 공유 메모리 연산을 사용하는 collaborative_filter

#### Step 4: 초기화 과정 탐색

```bash
(cuda-gdb) n
55          output: LayoutTensor[mut=True, dtype, vector_layout],
(cuda-gdb) n
58          thread_id = thread_idx.x
(cuda-gdb) n
66          ].stack_allocation()
(cuda-gdb) n
69          if thread_id < SIZE - 1:
(cuda-gdb) p thread_id
$1 = 0
```

**✅ 스레드 0 상태**: `thread_id = 0`, 조건 `0 < 3` 검사 직전 → **True**

#### Step 5: 1단계 추적

```bash
(cuda-gdb) n
70              shared_workspace[thread_id] = rebind[Scalar[dtype]](a[thread_id])
(cuda-gdb) n
69          if thread_id < SIZE - 1:
(cuda-gdb) n
71          barrier()
```

**1단계 완료**: 스레드 0이 초기화를 실행하고 첫 번째 barrier에 도달했습니다.

### 3단계: 결정적인 barrier 조사

#### Step 6: 첫 번째 barrier 검사

```bash
(cuda-gdb) n
74          if thread_id < SIZE - 1:
(cuda-gdb) info cuda threads
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (3,0,0)     4 0x00007fffd3272180 /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo    74
```

**✅ 정상**: 4개 스레드 모두 74번 줄(첫 번째 barrier 통과 후)에 있습니다. 첫 번째 barrier는 정상 작동했습니다.

**🔍 결정적 지점**: 이제 또 다른 조건문이 있는 2단계에 진입합니다.

#### Step 7: 2단계 추적 - 스레드 0 관점

```bash
(cuda-gdb) n
76              if thread_id > 0:
```

**스레드 0 분석**: `0 < 3` → **True** → 스레드 0이 2단계 블록에 진입

```bash
(cuda-gdb) n
78              barrier()
```

**스레드 0 경로**: `0 > 0` → **False** → 스레드 0이 내부 연산은 건너뛰지만 78번 줄의 barrier에 도달

**결정적 순간**: 스레드 0이 이제 78번 줄의 barrier에서 대기 중입니다.

```bash
(cuda-gdb) n # <-- 실행하면 프로그램이 멈춥니다!
[HANGS HERE - 프로그램이 이 지점을 넘어가지 못함]
```

#### Step 8: 다른 스레드 조사

```bash
(cuda-gdb) cuda thread (1,0,0)
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (1,0,0), device 0, sm 0, warp 0, lane 1]
78              barrier()
(cuda-gdb) p thread_id
$2 = 1
(cuda-gdb) info cuda threads
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (2,0,0)     3 0x00007fffd3273aa0 /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo    78
   (0,0,0)   (3,0,0)     (0,0,0)      (3,0,0)     1 0x00007fffd3273b10 /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo    81
```

**결정적 증거 발견**:

- **스레드 0, 1, 2**: 78번 줄에서 모두 대기 중 (조건 블록 안의 barrier)
- **스레드 3**: 81번 줄에 있음 (조건 블록을 지나쳤고, barrier에 도달한 적 없음!)

#### Step 9: 스레드 3의 실행 경로 분석

**🔍 info 출력으로 본 스레드 3 분석**:

- **스레드 3**: 81번 줄에 위치 (PC: 0x00007fffd3273b10)
- **2단계 조건**: `thread_id < SIZE - 1` → `3 < 3` → **False**
- **결과**: 스레드 3은 2단계 블록(74-78번 줄)에 **진입하지 않음**
- **결과**: 스레드 3은 78번 줄의 barrier에 **도달한 적 없음**
- **현재 상태**: 스레드 3은 81번 줄(마지막 barrier)에 있고, 스레드 0,1,2는 78번 줄에서 갇혀 있음

### 4단계: 근본 원인 분석

#### Step 10: 교착 상태 메커니즘 식별

```mojo
# 2단계: 협력적 처리
if thread_id < SIZE - 1:        # ← 스레드 0, 1, 2만 이 블록에 진입
    # 이웃과 협력 필터 적용
    if thread_id > 0:
        shared_workspace[thread_id] += shared_workspace[thread_id - 1] * 0.5
    barrier()                   # ← 교착 상태: 4개 중 3개 스레드만 여기에 도달!
```

**💀 교착 상태 메커니즘**:

1. **스레드 0**: `0 < 3` → **True** → 블록 진입 → **barrier에서 대기** (69번 줄)
2. **스레드 1**: `1 < 3` → **True** → 블록 진입 → **barrier에서 대기** (69번 줄)
3. **스레드 2**: `2 < 3` → **True** → 블록 진입 → **barrier에서 대기** (69번 줄)
4. **스레드 3**: `3 < 3` → **False** → **블록에 진입 안 함** → **72번 줄로 계속 진행**

**결과**: 3개 스레드가 4번째 스레드를 영원히 기다리지만, 스레드 3은 그 barrier에 절대 도착하지 않습니다.

### 5단계: 버그 확인과 해결책

#### Step 11: 근본적인 barrier 규칙 위반

**GPU Barrier 규칙**: 동기화가 완료되려면 스레드 블록의 모든 스레드가 같은 barrier에 도달해야 합니다.

**무엇이 잘못되었나**:

```mojo
# ❌ 잘못된 방법: 조건문 안에 barrier
if thread_id < SIZE - 1:    # 모든 스레드가 진입하지 않음
    # ... 연산 ...
    barrier()               # 일부 스레드만 여기에 도달

# ✅ 올바른 방법: 조건문 밖에 barrier
if thread_id < SIZE - 1:    # 모든 스레드가 진입하지 않음
    # ... 연산 ...
 barrier()                  # 모든 스레드가 여기에 도달
```

**수정 방법**: barrier를 조건 블록 밖으로 이동:

```mojo
fn collaborative_filter(
    output: LayoutTensor[mut=True, dtype, vector_layout],
    a: LayoutTensor[mut=False, dtype, vector_layout],
):
    thread_id = thread_idx.x
    shared_workspace = LayoutTensor[
        dtype,
        Layout.row_major(SIZE-1),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    # 1단계: 공유 작업공간 초기화 (모든 스레드 참여)
    if thread_id < SIZE - 1:
        shared_workspace[thread_id] = rebind[Scalar[dtype]](a[thread_id])
    barrier()

    # 2단계: 협력적 처리
    if thread_id < SIZE - 1:
        if thread_id > 0:
            shared_workspace[thread_id] += shared_workspace[thread_id - 1] * 0.5
    # ✅ 수정: barrier를 조건문 밖으로 이동해서 모든 스레드가 도달하도록
    barrier()

    # 3단계: 최종 동기화와 출력
    barrier()

    if thread_id < SIZE - 1:
        output[thread_id] = shared_workspace[thread_id]
    else:
        output[thread_id] = rebind[Scalar[dtype]](a[thread_id])
```

## 핵심 디버깅 교훈

**Barrier 교착 상태 탐지**:

1. **`info cuda threads` 사용** - 어떤 스레드가 어느 줄에 있는지 보여줌
2. **스레드 상태 분기 찾기** - 일부 스레드가 다른 프로그램 위치에 있음
3. **조건부 실행 경로 추적** - 모든 스레드가 같은 barrier에 도달하는지 확인
4. **Barrier 도달 가능성 검증** - 다른 스레드들이 도달하는 barrier를 건너뛰는 스레드가 없는지 확인

**실무 GPU 디버깅의 현실**:

- **교착 상태는 소리 없는 살인자** - 오류 메시지 없이 프로그램이 그냥 멈춤
- **스레드 조율 디버깅은 인내가 필요** - 각 스레드 경로를 체계적으로 분석해야 함
- **조건부 barrier가 교착 상태의 1순위 원인** - 모든 스레드가 같은 동기화 지점에 도달하는지 항상 확인
- **CUDA-GDB 스레드 검사가 필수** - 스레드 조율 실패를 볼 수 있는 유일한 방법

**고급 GPU 동기화**:

- **Barrier 규칙**: 블록의 **모든** 스레드가 **같은** barrier에 도달해야 함
- **조건부 실행의 함정**: 어떤 if문이든 스레드 분기를 일으킬 수 있음
- **공유 메모리 조율**: 올바른 동기화를 위해 barrier 배치에 주의 필요
- **LayoutTensor가 교착 상태를 막아주지 않음**: 고수준 추상화라도 올바른 동기화는 여전히 필요

**💡 핵심 통찰**: Barrier 교착 상태는 GPU 버그 중 디버깅하기 가장 어려운 유형에 속합니다:

- **오류가 보이지 않음** - 그저 무한 대기
- **다중 스레드 분석 필요** - 스레드 하나만 봐서는 디버깅할 수 없음
- **조용한 실패 모드** - 정확성 버그가 아닌 성능 문제처럼 보임
- **복잡한 스레드 조율** - 모든 스레드에 걸쳐 실행 경로를 추적해야 함

CUDA-GDB로 스레드 상태를 분석하고, 분기된 실행 경로를 식별하고, barrier 도달 가능성을 검증하는 이 디버깅 방식은 실무 GPU 개발자들이 운영 시스템에서 교착 상태 문제에 맞닥뜨렸을 때 쓰는 방법과 정확히 같습니다.

</div>
</details>

## 다음 단계: GPU 디버깅 스킬 완성

**GPU 디버깅 삼부작을 완료했습니다!**

### 완성된 GPU 디버깅 무기고

**[첫 번째 사례](./first_case.md)에서 - 크래시 디버깅:**

- ✅ 오류 메시지를 가이드 삼아 **체계적인 크래시 조사**
- ✅ 포인터 주소 검사를 통한 **메모리 버그 탐지**
- ✅ 메모리 관련 문제를 위한 **CUDA-GDB 기초**

**[두 번째 사례](./second_case.md)에서 - 로직 버그 디버깅:**

- ✅ 뚜렷한 증상 없이 **알고리즘 오류 조사**
- ✅ 잘못된 결과를 근본 원인까지 추적하는 **패턴 분석 기법**
- ✅ 변수 검사가 안 될 때 **실행 흐름 디버깅**

**[세 번째 사례](./third_case.md)에서 - 조율 디버깅:**

- ✅ 스레드 조율 실패를 위한 **barrier 교착 상태 조사**
- ✅ 고급 CUDA-GDB 기법을 사용한 **다중 스레드 상태 분석**
- ✅ 복잡한 병렬 프로그램을 위한 **동기화 검증**

### 전문가의 GPU 디버깅 방법론

실무 GPU 개발자들이 사용하는 체계적인 접근법을 익혔습니다:

1. **증상 읽기** - 크래시인가? 잘못된 결과인가? 무한 정지인가?
2. **가설 수립** - 메모리 문제? 로직 오류? 조율 문제?
3. **증거 수집** - 버그 유형에 맞춰 CUDA-GDB를 전략적으로 활용
4. **체계적으로 테스트** - 목표 지향적 조사를 통해 각 가설 검증
5. **근본 원인 추적** - 증거의 연결 고리를 따라 원천까지

**업적 달성**: 이제 가장 흔한 세 가지 GPU 프로그래밍 문제를 디버깅할 수 있습니다:

- **메모리 크래시** ([첫 번째 사례](./first_case.md)) - null 포인터, 범위 밖 접근
- **로직 버그** ([두 번째 사례](./second_case.md)) - 알고리즘 오류, 잘못된 결과
- **조율 교착 상태** ([세 번째 사례](./third_case.md)) - barrier 동기화 실패
