<!-- i18n-source-commit: 3711cbf0f794001097a09faed0075ed062023569 -->

# 🧐 탐정 수사: 첫 번째 사례

## 개요

이번 퍼즐에서는 크래시가 발생하는 GPU 프로그램이 주어집니다. 소스 코드를 보지 않고 `(cuda-gdb)` 디버깅 도구만으로 문제를 찾아내야 합니다. 디버깅 스킬을 발휘해 미스터리를 풀어보세요!

**사전 준비**: [Mojo GPU 디버깅의 핵심](./essentials.md)을 먼저 완료해서 CUDA-GDB 설정과 기본 디버깅 명령어를 익혀두세요. 아래 명령을 실행했는지 확인하세요:

```bash
pixi run -e nvidia setup-cuda-gdb
```

이 명령은 시스템의 CUDA 설치를 자동으로 감지하고 GPU 디버깅에 필요한 링크를 설정합니다.

## 핵심 개념

이번 디버깅 챌린지에서 배울 내용:

- **체계적인 디버깅**: 오류 메시지를 단서 삼아 근본 원인 찾기
- **오류 분석**: 크래시 메시지와 스택 추적(stack trace) 해석하기
- **가설 수립**: 문제에 대한 합리적인 추측 세우기
- **디버깅 워크플로우**: 단계별 조사 과정 익히기

## 코드 실행

먼저 전체 코드를 보지 않고 커널만 살펴봅시다:

```mojo
{{#include ../../../../../problems/p09/p09.mojo:first_crash}}
```

버그를 직접 경험하려면 터미널에서 다음 명령을 실행하세요 (`pixi` 전용):

```bash
pixi run -e nvidia p09 --first-case
```

프로그램이 크래시하면 다음과 같은 출력이 나타납니다:

```txt
First Case: Try to identify what's wrong without looking at the code!

stack trace was not collected. Enable stack trace collection with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`
Unhandled exception caught during execution: At open-source/max/mojo/stdlib/stdlib/gpu/host/device_context.mojo:2082:17: CUDA call failed: CUDA_ERROR_INVALID_IMAGE (device kernel image is invalid)
To get more accurate error information, set MODULAR_DEVICE_CONTEXT_SYNC_MODE=true.
/home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/nvidia/bin/mojo: error: execution exited with a non-zero result: 1
```

## 과제: 탐정 수사

**도전**: 코드를 보지 않은 상태에서, 이 크래시를 조사하기 위한 디버깅 전략은 무엇일까요?

다음 명령으로 시작해 보세요:

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --first-case
```

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. **크래시 메시지를 꼼꼼히 읽기** - `CUDA_ERROR_ILLEGAL_ADDRESS`는 GPU가 잘못된 메모리에 접근하려 했다는 뜻입니다
2. **브레이크포인트 정보 확인** - CUDA-GDB가 멈출 때 표시되는 함수 파라미터를 살펴보세요
3. **모든 포인터를 체계적으로 검사** - `print`로 각 포인터 파라미터를 확인하세요
4. **수상한 주소 찾기** - 유효한 GPU 주소는 보통 큰 16진수입니다 (`0x0`은 무엇을 의미할까요?)
5. **메모리 접근 테스트** - 각 포인터로 데이터에 접근해서 어느 것이 실패하는지 확인하세요
6. **체계적으로 접근** - 탐정처럼 증거를 따라가며 증상에서 근본 원인까지 추적하세요
7. **유효한 패턴과 그렇지 않은 패턴 비교** - 한 포인터가 작동하고 다른 건 안 된다면, 문제가 있는 쪽에 집중하세요

</div>
</details>

<details class="solution-details">
<summary><strong>💡 조사 과정과 해결책</strong></summary>

<div class="solution-explanation">

## CUDA-GDB로 단계별 조사

### 디버거 실행

```bash
pixi run -e nvidia mojo debug --cuda-gdb --break-on-launch problems/p09/p09.mojo --first-case
```

### 브레이크포인트 정보 확인

CUDA-GDB가 멈추면 바로 유용한 단서가 나타납니다:

```
(cuda-gdb) run
CUDA thread hit breakpoint, p09_add_10_... (output=0x302000000, a=0x0)
    at /home/ubuntu/workspace/mojo-gpu-puzzles/problems/p09/p09.mojo:31
31          i = thread_idx.x
```

**🔍 첫 번째 단서**: 함수 시그니처에 `(output=0x302000000, a=0x0)`이 보입니다

- `output`은 유효한 GPU 메모리 주소를 가지고 있습니다
- `a`는 `0x0` - null 포인터입니다!

### 체계적인 변수 검사

```
(cuda-gdb) next
32          output[i] = a[i] + 10.0
(cuda-gdb) print i
$1 = 0
(cuda-gdb) print output
$2 = (!pop.scalar<f32> * @register) 0x302000000
(cuda-gdb) print a
$3 = (!pop.scalar<f32> * @register) 0x0
```

**증거 수집**:

- ✅ 스레드 인덱스 `i=0`은 유효합니다
- ✅ 결과 포인터 `0x302000000`은 올바른 GPU 주소입니다
- ❌ 입력 포인터 `0x0`은 null입니다

### 문제 확인

```
(cuda-gdb) print a[i]
Cannot access memory at address 0x0
```

**결정적 증거**: null 주소의 메모리에 접근할 수 없습니다 - 바로 이것이 크래시의 원인입니다!

## 근본 원인 분석

**문제점**: 이제 `--first-crash`의 [코드](../../../../../problems/p09/p09.mojo)를 보면, 호스트 코드가 GPU 메모리를 제대로 할당하지 않고 null 포인터를 만들고 있습니다:

```mojo
 input_buf = ctx.enqueue_create_buffer[dtype](0)  # 0개의 요소를 가진 `DeviceBuffer`를 생성합니다. 요소가 0개이므로 메모리가 할당되지 않아 NULL 포인터가 됩니다!
```

**왜 크래시가 발생하는가**:

1. `ctx.enqueue_create_buffer[dtype](0)`은 0개 요소를 가진 `DeviceBuffer`를 생성합니다.
2. 할당할 요소가 없으니 null 포인터를 반환합니다.
3. 이 null 포인터가 GPU 커널로 전달됩니다.
4. 커널이 `a[i]`에 접근하려 할 때 null을 역참조 → `CUDA_ERROR_ILLEGAL_ADDRESS`

## 수정 방법

Null 포인터 생성을 적절한 버퍼 할당으로 교체합니다:

```mojo
# 잘못된 방법: Null 포인터 생성
input_buf = ctx.enqueue_create_buffer[dtype](0)

# 올바른 방법: 안전한 처리를 위해 실제 GPU 메모리를 할당하고 초기화
input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
input_buf.enqueue_fill(0)
```

## 핵심 디버깅 교훈

**패턴 인식**:

- `0x0` 주소는 항상 null 포인터입니다
- 유효한 GPU 주소는 큰 16진수입니다 (예: `0x302000000`)

**디버깅 전략**:

1. **크래시 메시지 읽기** - 대체로 문제 유형에 대한 힌트를 줍니다
2. **함수 파라미터 확인** - CUDA-GDB가 브레이크포인트 진입 시 보여줍니다
3. **모든 포인터 검사** - 주소를 비교해서 null이나 잘못된 것을 찾습니다
4. **메모리 접근 테스트** - 수상한 포인터를 역참조해 봅니다
5. **할당 지점까지 추적** - 문제의 포인터가 어디서 생성되었는지 찾습니다

**💡 핵심 통찰**: 이런 유형의 null 포인터 버그는 GPU 프로그래밍에서 매우 흔합니다. 여기서 배운 체계적인 CUDA-GDB 조사 방법은 다른 많은 GPU 메모리 문제, 경쟁 상태, 커널 크래시를 디버깅할 때도 그대로 적용됩니다.

</div>
</details>

## 다음 단계: 크래시에서 조용한 버그로

**크래시 디버깅을 익혔습니다!** 이제 할 수 있습니다:

- 오류 메시지를 단서로 **GPU 크래시를 체계적으로 조사**
- 포인터 주소 검사를 통해 **null 포인터 버그 식별**
- 메모리 관련 디버깅에 **CUDA-GDB를 효과적으로 사용**

### 다음 도전: [탐정 수사: 두 번째 사례](./second_case.md)

**그런데 프로그램이 크래시하지 않는다면요?** 완벽하게 실행되지만 **잘못된 결과**가 나온다면?

[두 번째 사례](./second_case.md)는 전혀 다른 유형의 디버깅 도전입니다:

- 길잡이가 되어줄 **크래시 메시지가 없습니다**
- 조사할 **뚜렷한 포인터 문제도 없습니다**
- 문제를 가리키는 **스택 추적도 없습니다**
- 체계적인 조사가 필요한 **잘못된 결과만** 있습니다

**새롭게 익히게 될 스킬:**

- **로직 버그 탐지** - 크래시 없이 알고리즘 오류 찾기
- **패턴 분석** - 잘못된 출력에서 근본 원인까지 거슬러 올라가기
- **실행 흐름 디버깅** - 최적화 때문에 변수 검사가 안 될 때 대처하기

여기서 배운 체계적인 조사 방법 - 단서 읽기, 가설 세우기, 체계적으로 테스트하기 - 은 앞으로 마주할 더 미묘한 로직 오류를 디버깅하는 기초가 됩니다.
