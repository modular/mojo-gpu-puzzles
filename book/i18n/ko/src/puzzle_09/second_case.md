<!-- i18n-source-commit: 0b65eaf4dea76e1eb968b4fc2ca14e563df205a2 -->

# 🔍 탐정 수사: 두 번째 사례

## 개요

[첫 번째 사례에서 익힌 크래시 디버깅 스킬](./first_case.md)을 바탕으로, 이번에는 전혀 다른 유형의 도전을 마주합니다: 크래시 없이 잘못된 결과를 내는 **로직 버그**입니다.

**디버깅 관점의 전환:**
- **[첫 번째 사례](./first_case.md)**: 명확한 크래시 신호(`CUDA_ERROR_ILLEGAL_ADDRESS`)가 조사를 안내함
- **두 번째 사례**: 크래시도 없고 에러 메시지도 없음 - 탐정처럼 파헤쳐야 하는 미묘하게 잘못된 결과만 있음

이번 중급 디버깅 챌린지에서는 `LayoutTensor` 연산을 사용하는 **알고리즘 오류**를 조사합니다. 프로그램은 성공적으로 실행되지만 잘못된 출력을 내는데, 실제 개발에서 훨씬 흔하면서도 까다로운 디버깅 시나리오입니다.

**사전 준비**: [Mojo GPU 디버깅의 핵심](./essentials.md)과 [탐정 수사: 첫 번째 사례](./first_case.md)를 먼저 완료해서 CUDA-GDB 워크플로우와 체계적인 디버깅 기법을 익혀두세요. 아래 명령을 실행했는지 확인하세요:

```bash
pixi run -e nvidia setup-cuda-gdb
```

## 핵심 개념

이번 디버깅 챌린지에서 배울 내용:

- **LayoutTensor 디버깅**: 구조화된 데이터 접근 패턴 조사하기
- **로직 버그 탐지**: 크래시하지 않는 알고리즘 오류 찾기
- **반복 경계 분석**: 반복 횟수 문제 이해하기
- **결과 패턴 분석**: 출력 데이터로 근본 원인까지 거슬러 올라가기

## 코드 실행

먼저 전체 코드를 보지 않고 kernel만 살펴봅시다:

```mojo
{{#include ../../../../../problems/p09/p09.mojo:second_crash}}
```

버그를 직접 경험하려면 터미널에서 다음 명령을 실행하세요 (`pixi` 전용):

```bash
pixi run -e nvidia p09 --second-case
```

다음과 같은 출력이 나타납니다 - **크래시 없이 잘못된 결과**:

```txt
This program computes sliding window sums for each position...

Input array: [0, 1, 2, 3]
Computing sliding window sums (window size = 3)...
Each position should sum its neighbors: [left + center + right]
stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
Unhandled exception caught during execution: At open-source/max/mojo/stdlib/stdlib/gpu/host/device_context.mojo:2082:17: CUDA call failed: CUDA_ERROR_INVALID_IMAGE (device kernel image is invalid)
To get more accurate error information, set MODULAR_DEVICE_CONTEXT_SYNC_MODE=true.
/home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/nvidia/bin/mojo: error: execution exited with a non-zero result: 1
```

## 과제: 탐정 수사

**도전**: 프로그램은 크래시 없이 실행되지만 일정한 패턴으로 잘못된 결과를 냅니다. 코드를 보지 않은 상태에서, 이 로직 버그를 조사하기 위한 체계적인 접근 방식은 무엇일까요?

**생각해 볼 점:**

- 잘못된 결과에서 어떤 패턴이 보이나요?
- 제대로 돌지 않는 것 같은 반복문은 어떻게 조사할 건가요?
- 변수를 직접 검사할 수 없을 때 어떤 디버깅 전략이 효과적일까요?
- 조사를 안내해 줄 크래시 신호가 없을 때, [첫 번째 사례](./first_case.md)의 체계적인 조사 방법을 어떻게 적용할 수 있을까요?

다음 명령으로 시작해 보세요:

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --second-case
```

### GDB 명령어 단축키 (빠른 디버깅)

**이 단축키들**을 사용하면 디버깅 세션 속도를 높일 수 있습니다:

| 단축 | 전체 | 사용 예시 |
|------|------|-----------|
| `r` | `run` | `(cuda-gdb) r` |
| `n` | `next` | `(cuda-gdb) n` |
| `c` | `continue` | `(cuda-gdb) c` |
| `b` | `break` | `(cuda-gdb) b 39` |
| `p` | `print` | `(cuda-gdb) p thread_id` |
| `q` | `quit` | `(cuda-gdb) q` |

**아래 모든 디버깅 명령어는 효율을 위해 이 단축키를 사용합니다!**

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. **패턴 분석부터** - 기대값과 실제 결과의 관계를 살펴보세요 (차이에 어떤 수학적 패턴이 있나요?)
2. **실행 흐름에 집중** - 변수에 접근할 수 없으면 반복 횟수를 세어보세요
3. **단순한 브레이크포인트 사용** - 최적화된 코드에서는 복잡한 디버깅 명령이 실패하기 쉽습니다
4. **수학적 추론** - 각 스레드가 접근해야 하는 것과 실제로 접근하는 것을 따져보세요
5. **누락된 데이터 조사** - 결과가 일관되게 기대보다 작다면, 무엇이 빠졌을까요?
6. **호스트 출력 검증** - 최종 결과에서 버그의 패턴이 드러나는 경우가 많습니다
7. **알고리즘 경계 분석** - 반복문이 올바른 개수의 요소를 처리하는지 확인하세요
8. **작동하는 케이스와 교차 검증** - 스레드 3은 정확하게 작동하는데 다른 것들은 왜 안 될까요?

</div>
</details>

<details class="solution-details">
<summary><strong>💡 조사 과정과 해결책</strong></summary>

<div class="solution-explanation">

## CUDA-GDB로 단계별 조사

### 1단계: 실행과 초기 분석

#### Step 1: 디버거 실행

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --second-case
```

#### Step 2: 증상부터 분석

디버거로 들어가기 전에, 이미 알고 있는 것을 정리합니다:

```txt
실제 결과: [0.0, 1.0, 3.0, 5.0]
기대값: [1.0, 3.0, 6.0, 5.0]
```

**🔍 패턴 인식**:

- 스레드 0: 0.0 얻음, 기대값 1.0 → 1.0 누락
- 스레드 1: 1.0 얻음, 기대값 3.0 → 2.0 누락
- 스레드 2: 3.0 얻음, 기대값 6.0 → 3.0 누락
- 스레드 3: 5.0 얻음, 기대값 5.0 → ✅ 정확

**초기 가설**: 각 스레드가 일부 데이터를 누락하고 있는데, 스레드 3만 정확하게 작동합니다.

### 2단계: Kernel 진입

#### Step 3: 브레이크포인트 진입 확인

실제 디버깅 세션에서는 다음과 같이 진행됩니다:

```bash
(cuda-gdb) r
Starting program: .../mojo run problems/p09/p09.mojo --second-case

This program computes sliding window sums for each position...
Input array: [0, 1, 2, 3]
Computing sliding window sums (window size = 3)...
Each position should sum its neighbors: [left + center + right]

[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

CUDA thread hit application kernel entry function breakpoint, p09_process_sliding_window_...
   <<<(1,1,1),(4,1,1)>>> (output=..., input=...)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:30
30          input: LayoutTensor[mut=False, dtype, vector_layout],
```

#### Step 4: 메인 로직으로 이동

```bash
(cuda-gdb) n
29          output: LayoutTensor[mut=True, dtype, vector_layout],
(cuda-gdb) n
32          thread_id = thread_idx.x
(cuda-gdb) n
38          for offset in range(ITER):
```

#### Step 5: 변수 접근성 테스트 - 중요한 발견

```bash
(cuda-gdb) p thread_id
$1 = 0
```

**✅ 좋음**: Thread ID에 접근 가능합니다.

```bash
(cuda-gdb) p window_sum
Cannot access memory at address 0x0
```

**❌ 문제**: `window_sum`에 접근할 수 없습니다.

```bash
(cuda-gdb) p a[0]
Attempt to take address of value not located in memory.
```

**❌ 문제**: LayoutTensor 직접 인덱싱이 작동하지 않습니다.

```bash
(cuda-gdb) p a.ptr[0]
$2 = {0}
(cuda-gdb) p a.ptr[0]@4
$3 = {{0}, {1}, {2}, {3}}
```

**🎯 돌파구**: `a.ptr[0]@4`로 전체 입력 배열을 볼 수 있습니다! 이것이 LayoutTensor 데이터를 검사하는 방법입니다.

### 3단계: 핵심 반복문 조사

#### Step 6: 반복문 모니터링 설정

```bash
(cuda-gdb) b 42
Breakpoint 1 at 0x7fffd326ffd0: file problems/p09/p09.mojo, line 42.
(cuda-gdb) c
Continuing.

CUDA thread hit Breakpoint 1, p09_process_sliding_window_...
   <<<(1,1,1),(4,1,1)>>> (output=..., input=...)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:42
42              idx = thread_id + offset - 1
```

**🔍 이제 반복문 본문 안에 있습니다. 직접 반복 횟수를 세어봅시다.**

#### Step 7: 첫 번째 반복 (offset = 0)

```bash
(cuda-gdb) n
43              if 0 <= idx < SIZE:
(cuda-gdb) n
41          for offset in range(ITER):
```

**첫 번째 반복 완료**: 반복문이 42번 줄 → 43번 줄 → 41번 줄로 돌아왔습니다. 반복문이 계속됩니다.

#### Step 8: 두 번째 반복 (offset = 1)

```bash
(cuda-gdb) n

CUDA thread hit Breakpoint 1, p09_process_sliding_window_...
42              idx = thread_id + offset - 1
(cuda-gdb) n
43              if 0 <= idx < SIZE:
(cuda-gdb) n
44                  value = rebind[Scalar[dtype]](input[idx])
(cuda-gdb) n
45                  window_sum += value
(cuda-gdb) n
43              if 0 <= idx < SIZE:
(cuda-gdb) n
41          for offset in range(ITER):
```

**두 번째 반복 완료**: 이번에는 if 블록(44-45번 줄)을 통과했습니다.

#### Step 9: 세 번째 반복 테스트

```bash
(cuda-gdb) n
47          output[thread_id] = window_sum
```

**결정적 발견**: 반복문이 2번만 돌고 종료되었습니다! 42번 줄의 브레이크포인트에 다시 걸리지 않고 47번 줄로 바로 넘어갔습니다.

**결론**: 반복문이 정확히 **2번** 돌고 종료되었습니다.

#### Step 10: Kernel 실행 완료와 컨텍스트 손실

```bash
(cuda-gdb) n
31      fn process_sliding_window(
(cuda-gdb) n
[Switching to Thread 0x7ffff7cc0e00 (LWP 110927)]
0x00007ffff064f84a in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
(cuda-gdb) p output.ptr[0]@4
No symbol "output" in current context.
(cuda-gdb) p offset
No symbol "offset" in current context.
```

**🔍 컨텍스트 손실**: Kernel 실행이 끝나면 kernel 변수에 더 이상 접근할 수 없습니다. 정상적인 동작입니다.

### 4단계: 근본 원인 분석

#### Step 11: 관찰된 실행에서 알고리즘 분석

디버깅 세션에서 관찰한 것:

1. **반복 횟수**: 2번만 반복 (offset = 0, offset = 1)
2. **기대값**: 크기 3의 슬라이딩 윈도우는 3번 반복해야 함 (offset = 0, 1, 2)
3. **누락**: 세 번째 반복 (offset = 2)

각 스레드가 계산해야 할 것:

- **스레드 0**: window_sum = input[-1] + input[0] + input[1] = (경계) + 0 + 1 = 1.0
- **스레드 1**: window_sum = input[0] + input[1] + input[2] = 0 + 1 + 2 = 3.0
- **스레드 2**: window_sum = input[1] + input[2] + input[3] = 1 + 2 + 3 = 6.0
- **스레드 3**: window_sum = input[2] + input[3] + input[4] = 2 + 3 + (경계) = 5.0

#### Step 12: 스레드 0의 실제 실행 추적

2번만 반복할 경우 (offset = 0, 1):

**반복 1 (offset = 0)**:

- `idx = thread_id + offset - 1 = 0 + 0 - 1 = -1`
- `if 0 <= idx < SIZE:` → `if 0 <= -1 < 4:` → **False**
- 합산 연산 건너뜀

**반복 2 (offset = 1)**:

- `idx = thread_id + offset - 1 = 0 + 1 - 1 = 0`
- `if 0 <= idx < SIZE:` → `if 0 <= 0 < 4:` → **True**
- `window_sum += input[0]` → `window_sum += 0`

**누락된 반복 3 (offset = 2)**:

- `idx = thread_id + offset - 1 = 0 + 2 - 1 = 1`
- `if 0 <= idx < SIZE:` → `if 0 <= 1 < 4:` → **True**
- `window_sum += input[1]` → `window_sum += 1` ← **이 연산이 실행되지 않음**

**결과**: 스레드 0은 `window_sum = 0 + 1 = 1` 대신 `window_sum = 0`을 얻습니다

### 5단계: 버그 확인

문제 코드를 보면:

```mojo
comptime ITER = 2                       # ← 버그: 3이어야 함!

for offset in range(ITER):           # ← 2번만 반복: [0, 1]
    idx = Int(thread_id) + offset - 1     # ← offset = 2 누락
    if 0 <= idx < SIZE:
        value = rebind[Scalar[dtype]](a[idx])
        window_sum += value
```

**🎯 근본 원인 확인**: 크기 3의 슬라이딩 윈도우를 위해 `ITER = 2`가 `ITER = 3`이어야 합니다.

**수정 방법**: 소스 코드에서 `comptime ITER = 2`를 `comptime ITER = 3`으로 변경합니다.

## 핵심 디버깅 교훈

**변수에 접근할 수 없을 때**:

1. **실행 흐름에 집중** - 브레이크포인트가 몇 번 걸리는지, 반복이 몇 번 도는지 세어보세요
2. **수학적 추론 사용** - 일어나야 할 일과 실제로 일어나는 일을 따져보세요
3. **패턴 분석** - 잘못된 결과가 조사를 이끌도록 하세요
4. **교차 검증** - 여러 데이터 포인트에 대해 가설을 테스트하세요

**전문적인 GPU 디버깅의 현실**:

- 컴파일러 최적화 때문에 **변수 검사가 실패하는 경우가 많습니다**
- **실행 흐름 분석**이 데이터 검사보다 더 신뢰할 수 있습니다
- **호스트 출력 패턴**이 중요한 디버깅 단서를 제공합니다
- **소스 코드 추론**이 제한된 디버거 기능을 보완합니다

**LayoutTensor 디버깅**:

- LayoutTensor 추상화를 사용해도 근본적인 알고리즘 버그는 그대로 드러납니다
- 텐서 내용을 검사하려 하기보다 알고리즘 로직에 집중하세요
- 체계적인 추론으로 각 스레드가 접근해야 하는 것과 실제로 접근하는 것을 추적하세요

**💡 핵심 통찰**: 이런 유형의 off-by-one (_역주: 경계값이 1만큼 어긋나는 오류_) 반복문 버그는 GPU 프로그래밍에서 매우 흔합니다. 여기서 배운 체계적인 접근법 - 제한된 디버거 정보에 수학적 분석과 패턴 인식을 결합하는 것 - 은 도구에 한계가 있을 때 전문 GPU 개발자들이 디버깅하는 방식 그대로입니다.

</div>
</details>

## 다음 단계: 로직 버그에서 데드락으로

**로직 버그 디버깅을 익혔습니다!** 이제 할 수 있습니다:

- ✅ 크래시나 뚜렷한 증상 없이도 **알고리즘 오류 조사**
- ✅ **패턴 분석**으로 잘못된 결과에서 근본 원인까지 추적
- ✅ 실행 흐름 분석으로 **변수 접근이 제한된 상황에서 디버깅**
- ✅ 디버거 도구에 한계가 있을 때 **수학적 추론 적용**

### 마지막 도전: [탐정 수사: 세 번째 사례](./third_case.md)

**그런데 프로그램이 크래시하지도 않고 끝나지도 않는다면요?** **그냥 영원히 멈춰버린다면요?**

[세 번째 사례](./third_case.md)는 궁극의 디버깅 도전을 제시합니다:

- ❌ **크래시 메시지 없음** (첫 번째 사례처럼)
- ❌ **잘못된 결과 없음** (두 번째 사례처럼)
- ❌ **완료 자체가 없음** - 그냥 무한히 멈춤
- ✅ 고급 스레드 조정 분석이 필요한 **조용한 데드락**

**새롭게 익히게 될 스킬:**

- **Barrier 데드락 탐지** - 병렬 스레드에서 조정 실패 찾기
- **멀티 스레드 상태 분석** - 모든 스레드를 동시에 검사하기
- **동기화 디버깅** - 스레드 협력 실패 이해하기

**디버깅 진화:**

1. **첫 번째 사례**: 크래시 신호 따라가기 → 메모리 버그 찾기
2. **두 번째 사례**: 결과 패턴 분석하기 → 로직 버그 찾기
3. **세 번째 사례**: 스레드 상태 조사하기 → 조정 버그 찾기

이전 두 사례에서 배운 체계적인 조사 스킬 - 가설 수립, 증거 수집, 패턴 분석 - 은 가장 어려운 GPU 문제를 디버깅할 때 핵심이 됩니다: 조정이 어긋나 영원히 서로를 기다리는 스레드들.
