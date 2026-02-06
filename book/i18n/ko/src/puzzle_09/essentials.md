<!-- i18n-source-commit: a6f8350359da1569bc39e376bc26580246e61653 -->

# 📚 Mojo GPU 디버깅의 핵심

GPU 디버깅의 세계에 오신 것을 환영합니다! Puzzle 1-8을 통해 GPU 프로그래밍 개념을 배웠으니, 이제 모든 GPU 프로그래머에게 가장 중요한 기술을 배울 준비가 되었습니다: **문제가 발생했을 때 디버깅하는 방법**.

GPU 디버깅은 처음에는 어려워 보일 수 있습니다. 수천 개의 스레드가 병렬로 실행되고, 다양한 메모리 공간이 있으며, 하드웨어별 동작도 다루어야 합니다. 하지만 적절한 도구와 워크플로우만 있으면 GPU 코드 디버깅도 체계적으로 다룰 수 있습니다.

이 가이드에서는 **CPU 호스트 코드**(GPU 작업을 설정하는 부분)와 **GPU 커널 코드**(병렬 연산이 실행되는 부분) 모두를 디버깅하는 방법을 배웁니다. 실제 예제, 실제 디버거 출력, 그리고 여러분의 프로젝트에 바로 적용할 수 있는 단계별 워크플로우를 사용합니다.

**참고**: 다음 내용은 범용 IDE 호환성을 위해 명령줄 디버깅에 초점을 맞춥니다. VS Code 디버깅을 선호한다면 [Mojo 디버깅 문서](https://docs.modular.com/mojo/tools/debugging)에서 VS Code 전용 설정과 워크플로우를 참조하세요.

## GPU 디버깅이 다른 이유

도구로 들어가기 전에, GPU 디버깅이 특별한 이유를 살펴보겠습니다:

- **전통적인 CPU 디버깅**: 단일 스레드, 순차 실행, 단순한 메모리 모델
- **GPU 디버깅**: 수천 개의 스레드, 병렬 실행, 여러 메모리 공간, 경쟁 상태

이는 다음을 할 수 있는 전문 도구가 필요하다는 의미입니다:

- 서로 다른 GPU 스레드 간 전환
- 스레드별 변수와 메모리 검사
- 병렬 실행의 복잡성 처리
- CPU 설정 코드와 GPU 커널 코드 모두 디버깅

## 디버깅 도구 모음

Mojo의 GPU 디버깅 기능은 현재 NVIDIA GPU로 제한됩니다. [Mojo 디버깅 문서](https://docs.modular.com/mojo/tools/debugging)에 따르면 Mojo 패키지에는 다음이 포함됩니다:

- CPU 측 디버깅을 위한 Mojo 플러그인이 포함된 **LLDB 디버거**
- GPU 커널 디버깅을 위한 **CUDA-GDB 통합**
- 범용 IDE 호환성을 위한 `mojo debug`를 통한 **명령줄 인터페이스**

GPU 전용 디버깅에 대해서는 [Mojo GPU 디버깅 가이드](https://docs.modular.com/mojo/tools/gpu-debugging)에서 추가 기술 세부 사항을 제공합니다.

이 아키텍처는 익숙한 디버깅 명령어와 GPU 전용 기능, 두 가지 장점을 모두 제공합니다.

## 디버깅 워크플로우: 문제에서 해결까지

GPU 프로그램이 크래시하거나, 잘못된 결과를 내거나, 예상치 못한 동작을 할 때 다음의 체계적인 접근법을 따르세요:

1. **디버깅을 위한 코드 준비** (최적화 비활성화, 디버그 심볼 추가)
2. **적절한 디버거 선택** (CPU 호스트 코드 vs GPU 커널 디버깅)
3. **전략적 브레이크포인트 설정** (문제가 의심되는 위치에)
4. **실행 및 검사** (코드를 단계별로 실행하며 변수 검사)
5. **패턴 분석** (메모리 접근, 스레드 동작, 경쟁 상태)

이 워크플로우는 Puzzle 01의 간단한 배열 연산이든 Puzzle 08의 복잡한 공유 메모리 코드든 상관없이 작동합니다.

## Step 1: 디버깅을 위한 코드 준비

**🥇 철칙**: _최적화된_ 코드는 절대 디버깅하지 마세요. 최적화는 명령어 순서를 바꾸고, 변수를 제거하고, 함수를 인라인화하여 디버깅을 거의 불가능하게 만듭니다.

### 디버그 정보로 빌드하기

디버깅용 Mojo 프로그램을 빌드할 때는 항상 디버그 심볼을 포함하세요:

```bash
# 전체 디버그 정보로 빌드
mojo build -O0 -g your_program.mojo -o your_program_debug
```

**이 플래그들이 하는 일:**

- `-O0`: 모든 최적화를 비활성화하여 원래 코드 구조를 보존
- `-g`: 디버거가 머신 코드를 Mojo 소스에 매핑할 수 있도록 디버그 심볼 포함
- `-o`: 쉬운 식별을 위해 명명된 출력 파일 생성

### 이것이 중요한 이유

디버그 심볼 없이는 디버깅 세션이 이렇게 보입니다:

```
(lldb) print my_variable
error: use of undeclared identifier 'my_variable'
```

디버그 심볼이 있으면 다음과 같이 됩니다:

```
(lldb) print my_variable
(int) $0 = 42
```

## Step 2: 디버깅 접근법 선택

여기서 GPU 디버깅이 흥미로워집니다. **네 가지 다른 조합** 중에서 선택할 수 있으며, 적절한 것을 고르면 시간을 절약할 수 있습니다:

### 네 가지 디버깅 조합

**빠른 참조:**

```bash
# 1. JIT + LLDB: 소스에서 직접 CPU 호스트 코드 디버깅
pixi run mojo debug your_gpu_program.mojo

# 2. JIT + CUDA-GDB: 소스에서 직접 GPU 커널 디버깅
pixi run mojo debug --cuda-gdb --break-on-launch your_gpu_program.mojo

# 3. 바이너리 + LLDB: 미리 컴파일된 바이너리에서 CPU 호스트 코드 디버깅
pixi run mojo build -O0 -g your_gpu_program.mojo -o your_program_debug
pixi run mojo debug your_program_debug

# 4. 바이너리 + CUDA-GDB: 미리 컴파일된 바이너리에서 GPU 커널 디버깅
pixi run mojo debug --cuda-gdb --break-on-launch your_program_debug
```

### 각 접근법을 언제 사용할까

**학습과 빠른 실험용:**

- **JIT 디버깅** 사용 - 빌드 단계가 필요 없어 더 빠르게 반복 가능

**본격적인 디버깅 세션용:**

- **바이너리 디버깅** 사용 - 더 예측 가능하고 깔끔한 디버거 출력

**CPU 측 문제용** (버퍼 할당, 호스트 메모리, 프로그램 로직):

- **LLDB 모드** 사용 - `main()` 함수와 설정 코드 디버깅에 적합

**GPU 커널 문제용** (스레드 동작, GPU 메모리, 커널 크래시):

- **CUDA-GDB 모드** 사용 - 개별 GPU 스레드를 검사하는 유일한 방법

장점은 다양하게 조합해서 사용할 수 있다는 점입니다. JIT + LLDB로 설정 코드를 디버깅한 다음, JIT + CUDA-GDB로 전환해서 실제 커널을 디버깅할 수 있습니다.

---

## CUDA-GDB로 GPU 커널 디버깅 이해하기

이제 GPU 커널 디버깅입니다 - 디버깅 도구 모음에서 가장 강력하면서도 복잡한 부분입니다.

`--cuda-gdb`를 사용하면 Mojo는 NVIDIA의 [CUDA-GDB 디버거](https://docs.nvidia.com/cuda/cuda-gdb/index.html)와 통합됩니다. 이것은 단순한 디버거가 아닙니다 - GPU 컴퓨팅의 병렬 멀티스레드 세계를 위해 특별히 설계되었습니다.

### CUDA-GDB가 특별한 이유

**일반 GDB**는 한 번에 하나의 스레드를 디버깅하며 순차 코드를 단계별로 실행합니다.
**CUDA-GDB**는 수천 개의 GPU 스레드를 동시에 디버깅하며, 각각이 서로 다른 명령어를 실행할 수 있습니다.

이는 다음을 할 수 있다는 의미입니다:

- **GPU 커널 내부에 브레이크포인트 설정** - 어떤 스레드든 브레이크포인트에 도달하면 실행을 일시 정지
- **GPU 스레드 간 전환** - 같은 순간에 서로 다른 스레드가 무엇을 하는지 검사
- **스레드별 데이터 검사** - 같은 변수가 스레드마다 다른 값을 가지는 것을 확인
- **메모리 접근 패턴 디버깅** - 범위 초과 접근, 경쟁 상태, 메모리 손상 포착 (이런 문제 감지에 대해서는 Puzzle 10에서 더 자세히)
- **병렬 실행 분석** - 스레드들이 어떻게 상호작용하고 동기화하는지 이해

### 이전 퍼즐의 개념과 연결

Puzzle 1-8에서 배운 GPU 프로그래밍 개념을 기억하시나요? CUDA-GDB로 런타임에 모든 것을 검사할 수 있습니다:

#### 스레드 계층 구조 디버깅

Puzzle 1-8에서 다음과 같은 코드를 작성했습니다:

```mojo
# Puzzle 1에서: 기본 스레드 인덱싱
i = thread_idx.x  # 각 스레드가 고유한 인덱스를 얻음

# Puzzle 7에서: 2D 스레드 인덱싱
row = thread_idx.y  # 2D 스레드 그리드
col = thread_idx.x
```

CUDA-GDB로 **이 스레드 좌표들이 실제로 동작하는 것을 볼 수 있습니다**:

```gdb
(cuda-gdb) info cuda threads
```

출력:

```
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (3,0,0)     4 0x00007fffcf26fed0 /home/ubuntu/workspace/mojo-gpu-puzzles/solutions/p01/p01.mojo    13
```

그리고 특정 스레드로 이동해서 무엇을 하는지 볼 수 있습니다:

```gdb
(cuda-gdb) cuda thread (1,0,0)
```

출력:

```
[Switching to CUDA thread (1,0,0)]
```

정말 강력한 기능입니다 - 말 그대로 **병렬 알고리즘이 여러 스레드에서 실행되는 것을 직접 지켜볼 수 있습니다**.

#### 메모리 공간 디버깅

다양한 유형의 GPU 메모리에 대해 배운 Puzzle 8을 기억하시나요? CUDA-GDB로 모든 것을 검사할 수 있습니다:

```gdb
# 글로벌 메모리 검사 (Puzzle 1-5의 배열들)
(cuda-gdb) print input_array[0]@4
$1 = {{1}, {2}, {3}, {4}}   # Mojo 스칼라 형식

# 로컬 변수를 사용해 공유 메모리 검사 (thread_idx.x는 작동하지 않음)
(cuda-gdb) print shared_data[i]   # thread_idx.x 대신 로컬 변수 'i' 사용
$2 = {42}
```

디버거는 각 스레드가 메모리에서 정확히 무엇을 보는지 보여줍니다. 이는 경쟁 상태나 메모리 접근 버그를 잡기에 완벽합니다.

#### 전략적 브레이크포인트 배치

CUDA-GDB 브레이크포인트는 병렬 실행과 함께 작동하기 때문에 일반 브레이크포인트보다 훨씬 강력합니다:

```gdb
# 어떤 스레드든 커널에 진입할 때 중단
(cuda-gdb) break add_kernel

# 특정 스레드에 대해서만 중단 (문제 격리에 좋음)
(cuda-gdb) break add_kernel if thread_idx.x == 0

# 메모리 접근 위반 시 중단
(cuda-gdb) watch input_array[thread_idx.x]

# 특정 데이터 조건에서 중단
(cuda-gdb) break add_kernel if input_array[thread_idx.x] > 100.0
```

이를 통해 수천 개 스레드의 출력에 파묻히지 않고 정확히 관심 있는 스레드와 조건에 집중할 수 있습니다.

---

## 환경 준비하기

디버깅을 시작하기 전에 개발 환경이 제대로 구성되어 있는지 확인하세요. 이전 퍼즐들을 진행해왔다면 대부분 이미 설정되어 있을 것입니다!

**참고**: `pixi` 없이는 [NVIDIA 공식 리소스](https://developer.nvidia.com/cuda-toolkit)에서 CUDA Toolkit을 수동으로 설치하고, 드라이버 호환성을 관리하고, 환경 변수를 구성하고, 컴포넌트 간 버전 충돌을 처리해야 합니다. `pixi`는 모든 CUDA 의존성, 버전, 환경 구성을 자동으로 관리하여 이 복잡성을 제거합니다.

### `pixi`가 디버깅에 중요한 이유

**문제점**: GPU 디버깅은 CUDA 툴킷, GPU 드라이버, Mojo 컴파일러, 디버거 컴포넌트 간의 정밀한 조율이 필요합니다. 버전 불일치는 "디버거를 찾을 수 없음" 오류로 이어질 수 있습니다.

**해결책**: `pixi`를 사용하면 이 모든 컴포넌트가 조화롭게 작동합니다. `pixi run mojo debug --cuda-gdb`를 실행하면 pixi가 자동으로:

- CUDA 툴킷 경로 설정
- 올바른 GPU 드라이버 로드
- Mojo 디버깅 플러그인 구성
- 환경 변수를 일관되게 관리

### 설정 확인

모든 것이 작동하는지 확인해 봅시다:

```bash
# 1. GPU 하드웨어 접근 가능 여부 확인
pixi run nvidia-smi
# GPU와 드라이버 버전이 표시되어야 함

# 2. CUDA-GDB 통합 설정 (GPU 디버깅에 필요)
pixi run setup-cuda-gdb
# 시스템 CUDA-GDB 바이너리를 conda 환경에 링크

# 3. Mojo 디버거 사용 가능 여부 확인
pixi run mojo debug --help
# --cuda-gdb를 포함한 디버깅 옵션이 표시되어야 함

# 4. CUDA-GDB 통합 테스트
pixi run cuda-gdb --version
# NVIDIA CUDA-GDB 버전 정보가 표시되어야 함
```

이 명령어 중 하나라도 실패하면 `pixi.toml` 구성을 다시 확인하고 CUDA 툴킷 기능이 활성화되어 있는지 확인하세요.

**중요**: conda의 `cuda-gdb` 패키지는 래퍼 스크립트만 제공하기 때문에 `pixi run setup-cuda-gdb` 명령이 필요합니다. 이 명령은 시스템 CUDA 설치에서 실제 CUDA-GDB 바이너리를 자동 감지하고 conda 환경에 링크하여 전체 GPU 디버깅 기능을 활성화합니다.

**이 명령이 하는 일:**

스크립트는 여러 일반적인 위치에서 CUDA를 자동 감지합니다:

- `$CUDA_HOME` 환경 변수
- `/usr/local/cuda` (Ubuntu/Debian 기본값)
- `/opt/cuda` (ArchLinux 및 기타 배포판)
- 시스템 PATH (`which cuda-gdb` 통해)

구현 세부 사항은 [`scripts/setup-cuda-gdb.sh`](https://github.com/modular/mojo-gpu-puzzles/blob/main/scripts/setup-cuda-gdb.sh)를 참조하세요.

**WSL 사용자를 위한 특별 참고사항**: Part II에서 사용할 두 가지 디버그 도구(cuda-gdb와 compute-sanitizer)는 WSL에서 CUDA 애플리케이션 디버깅을 지원하지만, 레지스트리 키 `HKEY_LOCAL_MACHINE\SOFTWARE\NVIDIA Corporation\GPUDebugger\EnableInterface`를 추가하고 `(DWORD) 1`로 설정해야 합니다. 지원되는 플랫폼과 OS별 동작에 대한 자세한 내용은 [cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html#supported-platforms)와 [compute-sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#operating-system-specific-behavior)를 참조하세요.

---

## 실습 튜토리얼: 첫 GPU 디버깅 세션

이론도 좋지만 직접 경험하는 것만 한 게 없습니다. Puzzle 01 - 여러분이 잘 아는 간단한 "배열 각 요소에 10 더하기" 커널을 사용해서 실제 프로그램을 디버깅해 봅시다.

**왜 Puzzle 01인가?** 다음 이유로 완벽한 디버깅 튜토리얼입니다:

- **충분히 단순해서** 무엇이 _일어나야 하는지_ 이해할 수 있음
- 실제 커널 실행이 있는 **진짜 GPU 코드**
- CPU 설정 코드와 GPU 커널 코드 **모두 포함**
- **짧은 실행 시간**으로 빠른 반복 가능

이 튜토리얼이 끝나면 네 가지 디버깅 접근법 모두로 같은 프로그램을 디버깅하고, 실제 디버거 출력을 보고, 매일 사용할 필수 디버깅 명령어를 배우게 됩니다.

### 디버깅 접근법 학습 경로

Puzzle 01을 예제로 [네 가지 디버깅 조합](#네-가지-디버깅-조합)을 탐색합니다. **학습 경로**: JIT + LLDB(가장 쉬움)로 시작해서 CUDA-GDB(가장 강력함)로 진행합니다.

**⚠️ GPU 디버깅 시 중요사항**:

- `--break-on-launch` 플래그는 CUDA-GDB 접근법에서 **필수**
- **미리 컴파일된 바이너리** (접근법 3 & 4)는 디버깅을 위해 `i` 같은 로컬 변수를 보존
- **JIT 컴파일** (접근법 1 & 2)은 대부분의 로컬 변수를 최적화로 제거
- 본격적인 GPU 디버깅에는 **접근법 4** (바이너리 + CUDA-GDB) 사용

## 튜토리얼 Step 1: LLDB로 CPU 디버깅

가장 일반적인 디버깅 시나리오로 시작합시다: **프로그램이 크래시하거나 예상치 못한 동작을 해서 `main()` 함수에서 무슨 일이 일어나는지 봐야 할 때**.

**미션**: Puzzle 01의 CPU 측 설정 코드를 디버깅하여 Mojo가 GPU 메모리를 초기화하고 커널을 실행하는 방법을 파악합니다.

### 디버거 실행

JIT 컴파일로 LLDB 디버거를 시작합니다:

```bash
# 한 단계로 p01.mojo를 컴파일하고 디버깅
pixi run mojo debug solutions/p01/p01.mojo
```

LLDB 프롬프트가 보입니다: `(lldb)`. 이제 디버거 안에서 프로그램 실행을 검사할 준비가 되었습니다!

### 첫 디버깅 명령어들

Puzzle 01이 실행될 때 무슨 일이 일어나는지 추적해 봅시다. **보여드린 대로 정확히 이 명령어들을 입력**하고 출력을 관찰하세요:

**Step 1: main 함수에 브레이크포인트 설정**

```bash
(lldb) br set -n main
```

출력:

```
Breakpoint 1: where = mojo`main, address = 0x00000000027d7530
```

디버거가 main 함수를 찾았고 거기서 실행을 일시 정지합니다.

**Step 2: 프로그램 시작**

```bash
(lldb) run
```

출력:

```
Process 186951 launched: '/home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/default/bin/mojo' (x86_64)
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = breakpoint 1.1
    frame #0: 0x0000555557d2b530 mojo`main
mojo`main:
->  0x555557d2b530 <+0>: pushq  %rbp
    0x555557d2b531 <+1>: movq   %rsp, %rbp
    ...
```

프로그램이 브레이크포인트에서 멈췄습니다. 현재 **어셈블리 코드**를 보고 있는데 이는 정상입니다 - 디버거가 고수준 Mojo 소스에 도달하기 전에 저수준 머신 코드에서 시작합니다.

**Step 3: 시작 과정 탐색**

```bash
# 명령어 하나를 단계별 실행 시도
(lldb) next
```

출력:

```
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = instruction step over
    frame #0: 0x0000555557d2b531 mojo`main + 1
mojo`main:
->  0x555557d2b531 <+1>: movq   %rsp, %rbp
    0x555557d2b534 <+4>: pushq  %r15
    ...
```

어셈블리를 단계별로 실행하는 것은 지루할 수 있습니다. 더 관련 있는 부분으로 진행합시다.

**Step 4: Mojo 소스 코드에 도달하기 위해 계속**

```bash
# 시작 어셈블리를 건너뛰어 실제 코드로 이동
(lldb) continue
```

출력:

```
Process 186951 resuming
Process 186951 stopped and restarted: thread 1 received signal: SIGCHLD
2 locations added to breakpoint 1
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = breakpoint 1.3
    frame #0: 0x00007fff5c01e841 JIT(0x7fff5c075000)`stdlib::builtin::_startup::__mojo_main_prototype(argc=([0] = 1), argv=0x00007fffffffa858) at _startup.mojo:95:4
```

Mojo의 런타임이 초기화 중입니다. `_startup.mojo`는 Mojo의 내부 시작 코드를 나타냅니다. `SIGCHLD` 시그널은 정상입니다 - Mojo가 내부 프로세스를 관리하는 방식입니다.

**Step 5: 실제 코드로 계속**

```bash
# 한 번 더 continue해서 p01.mojo 코드에 도달!
(lldb) continue
```

출력:

```
Process 186951 resuming
Process 186951 stopped
* thread #1, name = 'mojo', stop reason = breakpoint 1.2
    frame #0: 0x00007fff5c014040 JIT(0x7fff5c075000)`p01::main(__error__=<unavailable>) at p01.mojo:24:23
   21
   22
   23   def main():
-> 24       with DeviceContext() as ctx:
   25           out = ctx.enqueue_create_buffer[dtype](SIZE)
   26           out.enqueue_fill(0)
   27           a = ctx.enqueue_create_buffer[dtype](SIZE)
```

이제 실제 Mojo 소스 코드를 볼 수 있습니다. 주목할 점:

- p01.mojo 파일의 **21-27번 줄**
- **현재 줄 24**: `with DeviceContext() as ctx:`
- **JIT 컴파일**: `JIT(0x7fff5c075000)`은 Mojo가 코드를 즉석에서 컴파일했음을 나타냄

**Step 6: 프로그램 완료**

```bash
# 프로그램을 완료까지 실행
(lldb) continue
```

출력:

```
Process 186951 resuming
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
Process 186951 exited with status = 0 (0x00000000)
```

### 배운 내용

🎓 **축하합니다!** 첫 GPU 프로그램 디버깅 세션을 완료했습니다. 무슨 일이 있었는지 살펴보겠습니다:

**거쳐온 디버깅 여정:**

1. **어셈블리로 시작** - 저수준 디버깅에서는 정상적인 현상이며, 디버거가 머신 수준에서 어떻게 작동하는지 보여줌
2. **Mojo 시작 과정 탐색** - Mojo에 내부 초기화 코드가 있음을 학습
3. **소스 코드 도달** - 구문 강조가 된 실제 p01.mojo 21-27번 줄 확인
4. **JIT 컴파일 관찰** - Mojo가 코드를 즉석에서 컴파일하는 것을 관찰
5. **성공적인 실행 확인** - 프로그램이 예상된 출력을 생성함을 확인

**LLDB 디버깅이 제공하는 것:**

- ✅ **CPU 측 가시성**: `main()` 함수, 버퍼 할당, 메모리 설정 확인
- ✅ **소스 코드 검사**: 줄 번호가 있는 실제 Mojo 코드 보기
- ✅ **변수 검사**: 호스트 측 변수(CPU 메모리) 값 확인
- ✅ **프로그램 흐름 제어**: 설정 로직을 줄 단위로 단계별 실행
- ✅ **오류 조사**: 장치 설정, 메모리 할당 등의 크래시 디버깅

**LLDB가 할 수 없는 것:**

- ❌ **GPU 커널 검사**: `add_10` 함수 실행 내부로 진입 불가능
- ❌ **스레드 수준 디버깅**: 개별 GPU 스레드 동작 확인 불가
- ❌ **GPU 메모리 접근**: GPU 스레드가 보는 데이터 검사 불가
- ❌ **병렬 실행 분석**: 경쟁 상태나 동기화 디버깅 불가

**LLDB 디버깅을 사용할 때:**

- GPU 코드가 실행되기 전에 프로그램이 크래시할 때
- 버퍼 할당이나 메모리 설정 문제
- 프로그램 초기화와 흐름 이해
- Mojo 애플리케이션이 어떻게 시작되는지 학습
- 빠른 프로토타이핑과 코드 변경 실험

**핵심 통찰**: LLDB는 **호스트 측 디버깅**에 완벽합니다 - GPU 실행 전후에 CPU에서 일어나는 모든 것. 실제 GPU 커널 디버깅에는 다음 접근법이 필요합니다...

## 튜토리얼 Step 2: 바이너리 디버깅

JIT 디버깅을 배웠으니 이제 프로덕션 환경에서 사용하는 **전문적인 접근법**을 탐색합시다.

**시나리오**: 여러 파일이 있는 복잡한 애플리케이션을 디버깅하거나 같은 프로그램을 반복적으로 디버깅해야 합니다. 먼저 바이너리를 빌드하면 더 많은 제어와 빠른 디버깅 반복이 가능합니다.

### 디버그 바이너리 빌드

**Step 1: 디버그 정보로 컴파일**

```bash
# 디버그 빌드 생성 (명확한 명명에 주목)
pixi run mojo build -O0 -g solutions/p01/p01.mojo -o solutions/p01/p01_debug
```

**여기서 일어나는 일:**

- 🔧 **`-O0`**: 최적화 비활성화 (정확한 디버깅에 반드시 필요)
- 🔍 **`-g`**: 머신 코드를 소스 코드에 매핑하는 디버그 심볼 포함
- 📁 **`-o p01_debug`**: 명확하게 이름 지은 디버그 바이너리 생성

**Step 2: 바이너리 디버깅**

```bash
# 미리 빌드된 바이너리 디버깅
pixi run mojo debug solutions/p01/p01_debug
```

### 무엇이 다른가 (그리고 더 나은가)

**시작 비교:**

| JIT 디버깅 | 바이너리 디버깅 |
|-----------|----------------|
| 한 단계로 컴파일 + 디버깅 | 한 번 빌드, 여러 번 디버깅 |
| 느린 시작 (컴파일 오버헤드) | 빠른 시작 |
| 컴파일 메시지가 디버그 출력과 섞임 | 깔끔한 디버거 출력 |
| 디버깅 중 생성되는 디버그 심볼 | 고정된 디버그 심볼 |

**같은 LLDB 명령어**(`br set -n main`, `run`, `continue`)를 실행하면 다음과 같은 차이를 느낄 수 있습니다:

- **빠른 시작** - 컴파일 지연 없음
- **깔끔한 출력** - JIT 컴파일 메시지 없음
- **더 예측 가능** - 디버그 심볼이 실행 간에 변하지 않음
- **전문적인 워크플로우** - 프로덕션 디버깅이 이렇게 작동함

---

## 튜토리얼 Step 3: GPU 커널 디버깅

지금까지는 **CPU 호스트 코드** - 설정, 메모리 할당, 초기화를 디버깅했습니다. 하지만 병렬 연산이 일어나는 실제 **GPU 커널**은 어떨까요?

**문제점**: `add_10` 커널은 잠재적으로 수천 개의 스레드가 동시에 실행되는 GPU에서 실행됩니다. LLDB는 GPU의 병렬 실행 환경에 접근할 수 없습니다.

**해결책**: CUDA-GDB - GPU 스레드, GPU 메모리, 병렬 실행을 이해하는 전문 디버거입니다.

### CUDA-GDB가 필요한 이유

GPU 디버깅이 근본적으로 다른 이유를 이해합시다:

**CPU 디버깅 (LLDB):**

- 순차적으로 실행되는 단일 스레드
- 추적할 콜 스택이 하나뿐
- 단순한 메모리 모델
- 변수가 단일 값을 가짐

**GPU 디버깅 (CUDA-GDB):**

- 병렬로 실행되는 수천 개의 스레드
- 여러 콜 스택 (스레드당 하나)
- 복잡한 메모리 계층 구조 (글로벌, 공유, 로컬, 레지스터)
- 같은 변수가 스레드마다 다른 값을 가짐

**실제 예**: `add_10` 커널에서 `thread_idx.x` 변수는 **각 스레드마다 다른 값**을 가집니다 - 스레드 0은 `0`을, 스레드 1은 `1`을 보는 식입니다. CUDA-GDB만이 이 병렬 현실을 보여줄 수 있습니다.

### CUDA-GDB 디버거 실행

**Step 1: GPU 커널 디버깅 시작**

접근법을 선택하세요:

```bash
# 이미 실행했는지 확인 (한 번이면 충분)
pixi run setup-cuda-gdb

# JIT + CUDA-GDB 사용 (위의 접근법 2)
pixi run mojo debug --cuda-gdb --break-on-launch solutions/p01/p01.mojo
```

학습과 빠른 반복에 적합한 **JIT + CUDA-GDB 접근법**을 사용합니다.

**Step 2: 실행하고 GPU 커널 진입 시 자동 정지**

CUDA-GDB 프롬프트는 이렇게 보입니다: `(cuda-gdb)`. 프로그램을 시작합니다:

```gdb
# 프로그램 실행 - GPU 커널이 실행될 때 자동으로 정지
(cuda-gdb) run
```

출력:

```
Starting program: /home/ubuntu/workspace/mojo-gpu-puzzles/.pixi/envs/default/bin/mojo...
[Thread debugging using libthread_db enabled]
...
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0)]

CUDA thread hit application kernel entry function breakpoint, p01_add_10_UnsafePointer...
   <<<(1,1,1),(4,1,1)>>> (output=0x302000000, a=0x302000200) at p01.mojo:16
16          i = thread_idx.x
```

**성공! GPU 커널 내부에서 자동으로 정지했습니다!** `--break-on-launch` 플래그가 커널 실행을 감지했고 이제 `i = thread_idx.x`가 실행되는 16번 줄에 있습니다.

**중요**: `break add_10`처럼 수동으로 브레이크포인트를 설정할 **필요 없습니다** - 커널 진입 브레이크포인트는 자동입니다. GPU 커널 함수는 CUDA-GDB에서 맹글링된 이름(`p01_add_10_UnsafePointer...` 같은)을 가지지만, 이미 커널 안에 있으므로 바로 디버깅을 시작할 수 있습니다.

**Step 3: 병렬 실행 탐색**

```gdb
# 브레이크포인트에서 일시 정지된 모든 GPU 스레드 보기
(cuda-gdb) info cuda threads
```

출력:

```
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC                                                       Filename  Line
kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)      (3,0,0)     4 0x00007fffd326fb70 /home/ubuntu/workspace/mojo-gpu-puzzles/solutions/p01/p01.mojo    16
```

완벽합니다! Puzzle 01의 **모든 4개 병렬 GPU 스레드**를 보여줍니다:

- **`*`가 현재 스레드 표시**: `(0,0,0)` - 디버깅 중인 스레드
- **스레드 범위**: `(0,0,0)`에서 `(3,0,0)`까지 - 블록의 모든 4개 스레드
- **Count**: `4` - 코드의 `THREADS_PER_BLOCK = 4`와 일치
- **같은 위치**: 모든 스레드가 `p01.mojo`의 16번 줄에서 일시 정지

**Step 4: 커널을 단계별 실행하고 변수 검사**

```gdb
# 'next'로 코드 단계별 실행 ('step'은 내부로 들어감)
(cuda-gdb) next
```

출력:

```
p01_add_10_UnsafePointer... at p01.mojo:17
17          output[i] = a[i] + 10.0
```

```gdb
# 로컬 변수는 미리 컴파일된 바이너리에서 작동!
(cuda-gdb) print i
```

출력:

```
$1 = 0                    # 이 스레드의 인덱스 (thread_idx.x 값 캡처)
```

```gdb
# GPU 내장 변수는 작동하지 않지만 필요 없음
(cuda-gdb) print thread_idx.x
```

출력:

```
No symbol "thread_idx" in current context.
```

```gdb
# 로컬 변수를 사용해 스레드별 데이터 접근
(cuda-gdb) print a[i]     # 이 스레드의 입력: a[0]
```

출력:

```
$2 = {0}                  # 입력 값 (Mojo 스칼라 형식)
```

```gdb
(cuda-gdb) print output[i] # 연산 전 이 스레드의 출력
```

출력:

```
$3 = {0}                  # 아직 0 - 연산이 아직 실행되지 않음!
```

```gdb
# 연산 줄 실행
(cuda-gdb) next
```

출력:

```
13      fn add_10(         # 연산 후 함수 시그니처 줄로 이동
```

```gdb
# 이제 결과 확인
(cuda-gdb) print output[i]
```

출력:

```
$4 = {10}                 # 이제 계산된 결과 표시: 0 + 10 = 10
```

```gdb
# 함수 파라미터는 여전히 사용 가능
(cuda-gdb) print a
```

출력:

```
$5 = (!pop.scalar<f32> * @register) 0x302000200
```

**Step 5: 병렬 스레드 간 이동**

```gdb
# 다른 스레드로 전환해서 실행 확인
(cuda-gdb) cuda thread (1,0,0)
```

출력:

```
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (1,0,0), device 0, sm 0, warp 0, lane 1]
13      fn add_10(         # 스레드 1도 함수 시그니처에 있음
```

```gdb
# 스레드의 로컬 변수 확인
(cuda-gdb) print i
```

출력:

```
$5 = 1                    # 스레드 1의 인덱스 (스레드 0과 다름!)
```

```gdb
# 이 스레드가 처리하는 것 검사
(cuda-gdb) print a[i]     # 이 스레드의 입력: a[1]
```

출력:

```
$6 = {1}                  # 스레드 1의 입력 값
```

```gdb
# 스레드 1의 연산은 이미 완료 (병렬 실행!)
(cuda-gdb) print output[i] # 이 스레드의 출력: output[1]
```

출력:

```
$7 = {11}                 # 1 + 10 = 11 (이미 계산됨)
```

```gdb
# 최고의 기법: 모든 스레드 결과를 한 번에 보기
(cuda-gdb) print output[0]@4
```

출력:

```
$8 = {{10}, {11}, {12}, {13}}     # 모든 4개 스레드의 결과를 한 명령어로!
```

```gdb
(cuda-gdb) print a[0]@4
```

출력:

```
$9 = {{0}, {1}, {2}, {3}}         # 비교를 위한 모든 입력 값
```

```gdb
# 너무 많이 진행하면 CUDA 컨텍스트를 잃습니다
(cuda-gdb) next
```

출력:

```
[Switching to Thread 0x7ffff7e25840 (LWP 306942)]  # 호스트 스레드로 복귀
0x00007fffeca3f831 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
```

```gdb
(cuda-gdb) print output[i]
```

출력:

```
No symbol "output" in current context.  # GPU 컨텍스트를 잃음!
```

**이 디버깅 세션의 핵심 통찰:**

- 🤯 **병렬 실행은 진짜입니다** - 스레드 (1,0,0)으로 전환하면 이미 연산이 완료되어 있습니다!
- **각 스레드는 서로 다른 데이터를 가집니다** - `i=0` vs `i=1`, `a[i]={0}` vs `a[i]={1}`, `output[i]={10}` vs `output[i]={11}`
- **배열 검사가 강력합니다** - `print output[0]@4`로 모든 스레드의 결과를 확인할 수 있습니다: `{{10}, {11}, {12}, {13}}`
- **GPU 컨텍스트는 깨지기 쉽습니다** - 너무 많이 진행하면 호스트 스레드로 돌아가고 GPU 변수를 잃습니다

이것이 바로 병렬 컴퓨팅의 본질입니다: **같은 코드, 스레드마다 다른 데이터, 동시 실행.**

### CUDA-GDB로 배운 내용

**미리 컴파일된 바이너리**로 GPU 커널 실행 디버깅을 완료했습니다. 다음은 실제로 작동하는 기능들입니다:

**습득한 GPU 디버깅 능력:**

- ✅ **GPU 커널 자동 디버깅** - `--break-on-launch`가 커널 진입 시점에서 정지합니다
- ✅ **GPU 스레드 간 이동** - `cuda thread`로 컨텍스트를 전환합니다
- ✅ **로컬 변수 접근** - `-O0 -g`로 컴파일된 바이너리에서 `print i`가 작동합니다
- ✅ **스레드별 데이터 검사** - 각 스레드가 서로 다른 `i`, `a[i]`, `output[i]` 값을 보여줍니다
- ✅ **모든 스레드 결과 보기** - `print output[0]@4`로 `{{10}, {11}, {12}, {13}}`을 한 번에 표시합니다
- ✅ **GPU 코드 단계별 실행** - `next`가 연산을 실행하고 결과를 보여줍니다
- ✅ **병렬 실행 확인** - 스레드가 동시에 실행됩니다 (전환하면 다른 스레드는 이미 계산 완료)
- ✅ **함수 파라미터 접근** - `output`과 `a` 포인터를 검사할 수 있습니다
- ❌ **GPU 내장 변수 사용 불가** - `thread_idx.x`, `blockIdx.x` 등은 작동하지 않습니다 (하지만 로컬 변수는 작동합니다!)
- 📊 **Mojo 스칼라 형식** - 값이 `10.0` 대신 `{10}`으로 표시됩니다
- ⚠️ **깨지기 쉬운 GPU 컨텍스트** - 너무 많이 진행하면 GPU 변수 접근을 잃습니다

**핵심 통찰:**

- **미리 컴파일된 바이너리** (`mojo build -O0 -g`)는 필수입니다 - 로컬 변수가 보존됩니다
- **`@N`을 사용한 배열 검사** - 모든 병렬 결과를 한 번에 보는 가장 효율적인 방법입니다
- **GPU 내장 변수는 없습니다** - 하지만 `i` 같은 로컬 변수가 필요한 정보를 담고 있습니다
- **Mojo는 `{value}` 형식을 사용합니다** - 스칼라가 `10.0` 대신 `{10}`으로 표시됩니다
- **단계별 실행에 주의하세요** - GPU 컨텍스트를 잃고 호스트 스레드로 돌아가기 쉽습니다

**실제 디버깅 기법들**

이제 실제 GPU 프로그래밍에서 마주치게 될 실용적인 디버깅 시나리오를 살펴봅시다:

#### 기법 1: 스레드 경계 확인

```gdb
# 모든 4개 스레드가 올바르게 계산했는지 확인
(cuda-gdb) print output[0]@4
```

출력:

```
$8 = {{10}, {11}, {12}, {13}}    # 모든 4개 스레드가 올바르게 계산
```

```gdb
# 유효 범위를 넘어 확인하여 범위 초과 문제 감지
(cuda-gdb) print output[0]@5
```

출력:

```
$9 = {{10}, {11}, {12}, {13}, {0}}  # 요소 4는 초기화되지 않음 (좋음!)
```

```gdb
# 입력과 비교하여 연산 검증
(cuda-gdb) print a[0]@4
```

출력:

```
$10 = {{0}, {1}, {2}, {3}}       # 입력 값: 0+10=10, 1+10=11 등
```

**이것이 중요한 이유**: 범위 초과 접근은 GPU 크래시의 가장 흔한 원인입니다. 이런 디버깅 단계로 일찍 발견할 수 있습니다.

#### 기법 2: 스레드 구성 이해

```gdb
# 스레드가 블록으로 어떻게 구성되는지 보기
(cuda-gdb) info cuda blocks
```

출력:

```
  BlockIdx To BlockIdx Count   State
kernel 0
*  (0,0,0)     (0,0,0)     1 running
```

```gdb
# 현재 블록의 모든 스레드 보기
(cuda-gdb) info cuda threads
```

출력은 어떤 스레드가 활성 상태인지, 정지되었는지, 오류가 있는지 보여줍니다.

**이것이 중요한 이유**: 스레드 블록 구성을 이해하면 동기화와 공유 메모리 문제를 디버깅하는 데 도움이 됩니다.

#### 기법 3: 메모리 접근 패턴 분석

```gdb
# GPU 메모리 주소 확인:
(cuda-gdb) print a               # 입력 배열 GPU 포인터
```

출력:

```
$9 = (!pop.scalar<f32> * @register) 0x302000200
```

```gdb
(cuda-gdb) print output          # 출력 배열 GPU 포인터
```

출력:

```
$10 = (!pop.scalar<f32> * @register) 0x302000000
```

```gdb
# 로컬 변수를 사용해 메모리 접근 패턴 확인:
(cuda-gdb) print a[i]            # 각 스레드가 'i'를 사용해 자신의 요소에 접근
```

출력:

```
$11 = {0}                        # 스레드의 입력 데이터
```

**이것이 중요한 이유**: 메모리 접근 패턴은 성능과 정확성에 영향을 미칩니다. 잘못된 패턴은 경쟁 상태나 크래시를 초래합니다.

#### 기법 4: 결과 검증 및 완료

```gdb
# 커널 실행을 단계별로 실행한 후 최종 결과 확인
(cuda-gdb) print output[0]@4
```

출력:

```
$11 = {10.0, 11.0, 12.0, 13.0}    # 완벽! 각 요소가 10 증가
```

```gdb
# 프로그램을 정상적으로 완료
(cuda-gdb) continue
```

출력:

```
...프로그램 출력이 성공 표시...
```

```gdb
# 디버거 종료
(cuda-gdb) exit
```

설정부터 결과까지 GPU 커널 실행 디버깅을 완료했습니다.

## GPU 디버깅 여정: 핵심 통찰

포괄적인 GPU 디버깅 튜토리얼을 완료했습니다. 병렬 컴퓨팅에 대해 발견한 내용입니다:

### 병렬 실행에 대한 깊은 통찰

1. **스레드 인덱싱의 실제**: `thread_idx.x`가 병렬 스레드마다 다른 값(0, 1, 2, 3...)을 갖는 것을 이론이 아닌 **직접 확인**했습니다

2. **메모리 접근 패턴 파악**: 각 스레드가 `a[thread_idx.x]`에서 읽고 `output[thread_idx.x]`에 쓰며, 충돌 없이 완벽한 데이터 병렬성을 만들어냅니다

3. **병렬 실행의 이해**: 수천 개의 스레드가 **동일한 커널 코드**를 동시에 실행하면서 각각 **서로 다른 데이터 요소**를 처리합니다

4. **GPU 메모리 계층 구조**: 배열은 글로벌 GPU 메모리에 있어 모든 스레드가 접근할 수 있지만, 스레드별 인덱싱을 사용합니다

### 모든 퍼즐에 적용되는 디버깅 기법

**Puzzle 01부터 Puzzle 08, 그리고 그 이후까지** 보편적으로 적용되는 기법을 습득했습니다:

- CPU 측 문제(장치 설정, 메모리 할당)는 **LLDB로 시작**합니다
- GPU 커널 문제(스레드 동작, 메모리 접근)는 **CUDA-GDB로 전환**합니다
- 특정 스레드나 데이터 조건에 집중하려면 **조건부 브레이크포인트**를 사용합니다
- 병렬 실행 패턴을 이해하려면 **스레드 간 이동**을 활용합니다
- 경쟁 상태와 범위 초과 오류를 잡으려면 **메모리 접근 패턴**을 확인합니다

**확장성**: 이 기법들은 다음 모든 상황에서 동일하게 작동합니다:

- **Puzzle 01**: 간단한 덧셈을 하는 4개 요소 배열
- **Puzzle 08**: 스레드 동기화가 필요한 복잡한 공유 메모리 연산
- **프로덕션 코드**: 정교한 알고리즘을 사용하는 백만 개 요소 배열

---

## 필수 디버깅 명령어 참조

디버깅 워크플로우를 배웠으니, 일상적인 디버깅 세션에서 쓸 **빠른 참조 가이드**를 드립니다. 이 섹션을 북마크하세요!

### GDB 명령어 약어 (시간 절약!)

**가장 많이 사용하는 단축키**로 더 빠른 디버깅:

| 약어 | 전체 명령어 | 기능 |
|-----|-----------|------|
| `r` | `run` | 프로그램 시작/실행 |
| `c` | `continue` | 실행 재개 |
| `n` | `next` | 스텝 오버 (같은 레벨) |
| `s` | `step` | 함수 내부로 진입 |
| `b` | `break` | 브레이크포인트 설정 |
| `p` | `print` | 변수 값 출력 |
| `l` | `list` | 소스 코드 표시 |
| `q` | `quit` | 디버거 종료 |

**예시:**

```bash
(cuda-gdb) r                    # 'run' 대신
(cuda-gdb) b 39                 # 'break 39' 대신
(cuda-gdb) p thread_id          # 'print thread_id' 대신
(cuda-gdb) n                    # 'next' 대신
(cuda-gdb) c                    # 'continue' 대신
```

**⚡ Pro 팁**: 약어를 사용하면 디버깅 속도가 3-5배 빨라집니다!

## LLDB 명령어 (CPU 호스트 코드 디버깅)

**언제 사용**: 장치 설정, 메모리 할당, 프로그램 흐름, 호스트 측 크래시 디버깅

### 실행 제어

```bash
(lldb) run                   # 프로그램 실행
(lldb) continue              # 실행 재개 (별칭: c)
(lldb) step                  # 함수 내부로 진입 (소스 레벨)
(lldb) next                  # 함수 건너뛰기 (소스 레벨)
(lldb) finish                # 현재 함수에서 나가기
```

### 브레이크포인트 관리

```bash
(lldb) br set -n main        # main 함수에 브레이크포인트 설정
(lldb) br set -n function_name     # 어떤 함수에든 브레이크포인트 설정
(lldb) br list               # 모든 브레이크포인트 표시
(lldb) br delete 1           # 브레이크포인트 #1 삭제
(lldb) br disable 1          # 브레이크포인트 #1 임시 비활성화
```

### 변수 검사

```bash
(lldb) print variable_name   # 변수 값 표시
(lldb) print pointer[offset]        # 포인터 역참조
(lldb) print array[0]@4      # 첫 4개 배열 요소 표시
```

## CUDA-GDB 명령어 (GPU 커널 디버깅)

**언제 사용**: GPU 커널, 스레드 동작, 병렬 실행, GPU 메모리 문제 디버깅

### GPU 상태 검사

```bash
(cuda-gdb) info cuda threads    # 모든 GPU 스레드와 상태 표시
(cuda-gdb) info cuda blocks     # 모든 스레드 블록 표시
(cuda-gdb) cuda kernel          # 활성 GPU 커널 나열
```

### 스레드 탐색 (가장 강력한 기능!)

```bash
(cuda-gdb) cuda thread (0,0,0)  # 특정 스레드 좌표로 전환
(cuda-gdb) cuda block (0,0)     # 특정 블록으로 전환
(cuda-gdb) cuda thread          # 현재 스레드 좌표 표시
```

### 스레드별 변수 검사

```bash
# 로컬 변수와 함수 파라미터:
(cuda-gdb) print i              # 로컬 스레드 인덱스 변수
(cuda-gdb) print output         # 함수 파라미터 포인터
(cuda-gdb) print a              # 함수 파라미터 포인터
```

### GPU 메모리 접근

```bash
# 로컬 변수를 사용한 배열 검사 (실제로 작동하는 것):
(cuda-gdb) print array[i]       # 로컬 변수를 사용한 스레드별 배열 접근
(cuda-gdb) print array[0]@4     # 여러 요소 보기: {{val1}, {val2}, {val3}, {val4}}
```

### 고급 GPU 디버깅

```bash
# 메모리 감시
(cuda-gdb) watch array[i]     # 메모리 변경 시 중단
(cuda-gdb) rwatch array[i]    # 메모리 읽기 시 중단
```

---

## 빠른 참조: 디버깅 결정 트리

**🤔 어떤 유형의 문제를 디버깅하고 있나요?**

### GPU 코드 실행 전에 프로그램이 크래시

→ **LLDB 디버깅 사용**

```bash
pixi run mojo debug your_program.mojo
```

### GPU 커널이 잘못된 결과 생성

→ **조건부 브레이크포인트와 함께 CUDA-GDB 사용**

```bash
pixi run mojo debug --cuda-gdb --break-on-launch your_program.mojo
```

### 성능 문제나 경쟁 상태

→ **재현성을 위해 바이너리 디버깅 사용**

```bash
pixi run mojo build -O0 -g your_program.mojo -o debug_binary
pixi run mojo debug --cuda-gdb --break-on-launch debug_binary
```

---

## GPU 디버깅의 핵심을 배웠습니다

GPU 디버깅 기초에 대한 포괄적인 튜토리얼을 완료했습니다. 다음은 달성한 내용입니다:

### 습득한 기술

**다중 레벨 디버깅 지식**:

- ✅ LLDB로 **CPU 호스트 디버깅** - 장치 설정, 메모리 할당, 프로그램 흐름 디버깅
- ✅ CUDA-GDB로 **GPU 커널 디버깅** - 병렬 스레드, GPU 메모리, 경쟁 상태 디버깅
- ✅ **JIT vs 바이너리 디버깅** - 상황에 맞는 접근법 선택
- ✅ pixi로 **환경 관리** - 일관되고 신뢰할 수 있는 디버깅 설정 보장

**실제 병렬 프로그래밍 통찰**:

- **스레드의 실제 동작 확인** - 병렬 스레드마다 `thread_idx.x`가 다른 값을 갖는 것을 직접 목격했습니다
- **메모리 계층 구조 이해** - 글로벌 GPU 메모리, 공유 메모리, 스레드 로컬 변수를 디버깅했습니다
- **스레드 탐색 학습** - 수천 개의 병렬 스레드 사이를 효율적으로 이동했습니다

### 이론에서 실전으로

GPU 디버깅에 대해 읽기만 한 것이 아니라 **경험했습니다**:

- **실제 코드 디버깅**: 실제 GPU 실행으로 Puzzle 01의 `add_10` 커널을 디버깅했습니다
- **실제 디버거 출력 확인**: LLDB 어셈블리, CUDA-GDB 스레드 상태, 메모리 주소를 직접 확인했습니다
- **전문 도구 사용**: 프로덕션 GPU 개발에서 사용하는 것과 동일한 CUDA-GDB를 사용했습니다
- **실제 시나리오 해결**: 범위 초과 접근, 경쟁 상태, 커널 실행 실패 문제를 다뤘습니다

### 디버깅 도구 모음

**빠른 결정 가이드** (항상 가까이 두세요!):

| 문제 유형 | 도구 | 명령어 |
|----------|------|--------|
| **GPU 전에 프로그램 크래시** | LLDB | `pixi run mojo debug program.mojo` |
| **GPU 커널 문제** | CUDA-GDB | `pixi run mojo debug --cuda-gdb --break-on-launch program.mojo` |
| **경쟁 상태** | CUDA-GDB + 스레드 탐색 | `(cuda-gdb) cuda thread (0,0,0)` |

**필수 명령어** (일상 디버깅용):

```bash
# GPU 스레드 검사
(cuda-gdb) info cuda threads          # 모든 스레드 보기
(cuda-gdb) cuda thread (0,0,0)        # 스레드 전환
(cuda-gdb) print i                    # 로컬 스레드 인덱스 (thread_idx.x 등가)

# 스마트 브레이크포인트 (GPU 내장 변수가 작동하지 않으므로 로컬 변수 사용)
(cuda-gdb) break kernel if i == 0      # 스레드 0에 집중
(cuda-gdb) break kernel if array[i] > 100  # 데이터 조건에 집중

# 메모리 디버깅
(cuda-gdb) print array[i]              # 로컬 변수를 사용한 스레드별 데이터
(cuda-gdb) print array[0]@4            # 배열 세그먼트: {{val1}, {val2}, {val3}, {val4}}
```

---

### 요약

GPU 디버깅에는 수천 개의 병렬 스레드, 복잡한 메모리 계층 구조, 전문 도구가 관여합니다. 이제 다음을 갖추게 되었습니다:

- 어떤 GPU 프로그램에도 적용할 수 있는 **체계적인 워크플로우**
- LLDB와 CUDA-GDB **전문 도구**에 대한 친숙함
- 실제 병렬 코드를 디버깅한 **실전 경험**
- 복잡한 상황을 처리하기 위한 **실용적인 전략**
- GPU 디버깅 과제를 해결할 **기초**

---

## 추가 자료

- [Mojo 디버깅 문서](https://docs.modular.com/mojo/tools/debugging)
- [Mojo GPU 디버깅 가이드](https://docs.modular.com/mojo/tools/gpu-debugging)
- [NVIDIA CUDA-GDB 사용자 가이드](https://docs.nvidia.com/cuda/cuda-gdb/index.html)
- [CUDA-GDB 명령어 참조](https://docs.nvidia.com/cuda/cuda-gdb/index.html#command-reference)

**참고**: GPU 디버깅에는 인내심과 체계적인 조사가 필요합니다. 이 퍼즐에서 다룬 워크플로우와 명령어는 실제 애플리케이션에서 마주치게 될 복잡한 GPU 문제를 디버깅하는 기초가 됩니다.
