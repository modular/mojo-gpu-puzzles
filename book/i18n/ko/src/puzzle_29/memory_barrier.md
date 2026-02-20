<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# Double-Buffered Stencil 연산

> **🔬 세밀한 동기화: mbarrier vs barrier()**
>
> 이 퍼즐은 이전 퍼즐에서 사용한 기본 [`barrier()`](https://docs.modular.com/mojo/std/gpu/sync/sync/barrier/) 함수보다 훨씬 강력한 제어를 제공하는 **명시적 memory barrier API**를 소개합니다.
>
> **기본 `barrier()`의 한계:**
>
> - **일회성 사용**: 상태 추적 없이 단일 동기화 지점만 제공
> - **블록 전체 전용**: 블록의 모든 스레드가 동시에 참여해야 함
> - **재사용 불가**: 매 barrier() 호출이 새로운 동기화 이벤트를 생성
> - **세밀도 부족**: 메모리 순서와 타이밍에 대한 제한적 제어
> - **정적 조정**: 스레드 참여 패턴의 변화에 적응 불가
>
> **고급 [`mbarrier API`](https://docs.modular.com/mojo/std/gpu/sync/sync/)의 기능:**
>
> - **정밀한 제어**: [`mbarrier_init()`](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_init)로 특정 스레드 수를 지정하여 재사용 가능한 barrier 객체를 설정
> - **상태 추적**: [`mbarrier_arrive()`](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_arrive)로 개별 스레드 완료를 알리고 도착 횟수를 유지
> - **유연한 대기**: [`mbarrier_test_wait()`](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_test_wait)로 특정 완료 상태를 기다릴 수 있음
> - **재사용 가능한 객체**: 동일한 barrier를 여러 반복에 걸쳐 재초기화하고 재사용 가능
> - **다중 barrier**: 서로 다른 동기화 지점(초기화, 반복, 마무리)에 서로 다른 barrier 객체 사용
> - **하드웨어 최적화**: GPU 하드웨어 동기화 기본 요소에 직접 매핑하여 더 나은 성능
> - **메모리 의미론**: 메모리 가시성과 순서 보장에 대한 명시적 제어
>
> **반복 알고리즘에서 왜 중요한가:**
> Double-buffering 패턴에서는 버퍼 교체 단계 간의 **정밀한 조정**이 필요합니다. 기본 `barrier()`로는 다음에 필요한 세밀한 제어를 제공할 수 없습니다:
>
> - **버퍼 역할 교대**: buffer_A에 대한 모든 쓰기가 완료된 후에야 buffer_A에서 읽기 시작되도록 보장
> - **반복 경계**: 단일 kernel 내에서 여러 동기화 지점 조율
> - **상태 관리**: 어떤 스레드가 어떤 처리 단계를 완료했는지 추적
> - **성능 최적화**: 재사용 가능한 barrier 객체를 통해 동기화 오버헤드 최소화
>
> 이 퍼즐은 반복법, 시뮬레이션 프레임워크, 고성능 이미지 처리 파이프라인 등 실제 GPU 컴퓨팅 애플리케이션에서 사용되는 **동기화 패턴**을 보여줍니다.

## 개요

Double-buffered 공유 메모리를 사용하여 반복 stencil 연산을 수행하는 kernel을 구현합니다. 반복 간 안전한 버퍼 교체를 보장하기 위해 명시적 memory barrier로 조정합니다. Stencil 연산은 배열의 각 요소 값을 이웃 요소들의 고정된 패턴을 기반으로 계산하는 연산 패턴입니다.

**참고:** _버퍼 역할이 교대합니다: `buffer_A`와 `buffer_B`가 매 반복마다 읽기와 쓰기 연산을 교대하며, mbarrier 동기화가 버퍼 교체 전에 모든 스레드의 쓰기 완료를 보장합니다._

**알고리즘 아키텍처:** 이 퍼즐은 두 개의 공유 메모리 버퍼가 여러 반복에 걸쳐 읽기와 쓰기 대상의 역할을 교대하는 **double-buffering 패턴**을 구현합니다. 데이터를 한 번만 처리하는 단순한 stencil 연산과 달리, 이 접근 방식은 버퍼 전환 중 경쟁 상태를 방지하기 위한 세심한 memory barrier 조정과 함께 반복적 개선을 수행합니다.

**파이프라인 개념:** 알고리즘은 반복적 stencil 개선을 통해 데이터를 처리합니다. 각 반복은 하나의 버퍼에서 읽고 다른 버퍼에 쓰며, 버퍼들은 매 반복마다 역할을 교대하여 데이터 손상 없이 연속 처리를 가능하게 하는 핑퐁 패턴을 만듭니다.

**데이터 의존성과 동기화:** 각 반복은 이전 반복의 완성된 결과에 의존합니다:

- **반복 N → 반복 N+1**: 현재 반복이 다음 반복이 소비하는 개선된 데이터를 생성
- **버퍼 조정**: 읽기와 쓰기 버퍼가 매 반복마다 역할을 교환
- **Memory barrier가 경쟁 상태를 방지**: 새로 기록된 버퍼에서 읽기를 시작하기 전에 모든 쓰기가 완료되도록 보장

구체적으로, double-buffered stencil은 세 가지 수학 연산으로 구성된 반복적 스무딩 알고리즘을 구현합니다:

**반복 패턴 - 버퍼 교대:**

\\[\\text{Iteration } i: \\begin{cases}
\\text{Read from buffer\_A, Write to buffer\_B} & \\text{if } i \\bmod 2 = 0 \\\\
\\text{Read from buffer\_B, Write to buffer\_A} & \\text{if } i \\bmod 2 = 1
\\end{cases}\\]

**Stencil 연산 - 3점 평균:**

\\[S^{(i+1)}[j] = \\frac{1}{N_j} \\sum_{k=-1}^{1} S^{(i)}[j+k] \\quad \\text{where } j+k \\in [0, 255]\\]

여기서 \\(S^{(i)}[j]\\)는 반복 \\(i\\) 이후 위치 \\(j\\)에서의 stencil 값이고, \\(N_j\\)는 유효한 이웃 수입니다.

**Memory Barrier 조정:**

\\[\\text{mbarrier\_arrive}() \\Rightarrow \\text{mbarrier\_test\_wait}() \\Rightarrow \\text{buffer swap} \\Rightarrow \\text{next iteration}\\]

**최종 출력 선택:**

\\[\\text{Output}[j] = \\begin{cases}
\\text{buffer\_A}[j] & \\text{if STENCIL\_ITERATIONS } \\bmod 2 = 0 \\\\
\\text{buffer\_B}[j] & \\text{if STENCIL\_ITERATIONS } \\bmod 2 = 1
\\end{cases}\\]

## 핵심 개념

이 퍼즐에서는 다음을 배웁니다:

- 반복 알고리즘을 위한 double-buffering 패턴 구현
- [mbarrier API](https://docs.modular.com/mojo/std/gpu/sync/sync/)를 사용한 명시적 memory barrier 조정
- 반복에 걸쳐 교대하는 읽기/쓰기 버퍼 역할 관리

핵심 통찰은 읽기와 쓰기 연산 사이의 경쟁 상태가 적절히 동기화되지 않으면 데이터를 손상시킬 수 있는 반복 알고리즘에서 버퍼 교체를 안전하게 조율하는 방법을 이해하는 것입니다.

**왜 중요한가:** 대부분의 GPU 튜토리얼은 단순한 단일 패스 알고리즘을 보여주지만, 실제 애플리케이션에서는 데이터에 대한 다중 패스를 수행하는 **반복적 개선**이 필요한 경우가 많습니다. Double-buffering은 각 반복이 이전 반복의 완성된 결과에 의존하는 반복법, 이미지 처리 필터, 시뮬레이션 업데이트 같은 알고리즘에 필수적입니다.

**이전 퍼즐과 현재의 동기화 비교:**

- **이전 퍼즐 ([P8](../puzzle_08/puzzle_08.md), [P12](../puzzle_12/puzzle_12.md), [P15](../puzzle_15/puzzle_15.md)):** 단일 패스 알고리즘을 위한 단순 [`barrier()`](https://docs.modular.com/mojo/std/gpu/sync/sync/barrier/) 호출
- **이 퍼즐:** 버퍼 교체 타이밍에 대한 정밀한 제어를 위한 명시적 [mbarrier API](https://docs.modular.com/mojo/std/gpu/sync/sync/)

**Memory barrier 특화:** 기본적인 스레드 동기화와 달리, 이 퍼즐은 메모리 연산이 언제 완료되는지에 대한 세밀한 제어를 제공하는 **명시적 memory barrier**를 사용하며, 이는 복잡한 메모리 접근 패턴에 필수적입니다.

## 구성

**시스템 매개변수:**

- **이미지 크기**: `SIZE = 1024` 요소 (간소화를 위해 1D)
- **블록당 스레드 수**: `TPB = 256` 스레드, `(256, 1)` 블록 차원으로 구성
- **그리드 구성**: 전체 이미지를 타일 단위로 처리하기 위한 `(4, 1)` 블록 (총 4개 블록)
- **데이터 타입**: 모든 연산에 `DType.float32`

**반복 매개변수:**

- **Stencil 반복 횟수**: `STENCIL_ITERATIONS = 3` 개선 패스
- **버퍼 수**: `BUFFER_COUNT = 2` (double-buffering)
- **Stencil kernel**: 반지름 1의 3점 평균

**버퍼 아키텍처:**

- **buffer_A**: 주 공유 메모리 버퍼 (`[256]` 요소)
- **buffer_B**: 보조 공유 메모리 버퍼 (`[256]` 요소)
- **역할 교대**: 매 반복마다 버퍼가 읽기 소스와 쓰기 대상 사이를 교체

**처리 요구사항:**

**초기화 단계:**

- **버퍼 설정**: buffer_A를 입력 데이터로, buffer_B를 0으로 초기화
- **Barrier 초기화**: 동기화 지점을 위한 [mbarrier 객체](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_init) 설정
- **스레드 조정**: 모든 스레드가 초기화에 참여

**반복 처리:**

- **짝수 반복** (0, 2, 4...): buffer_A에서 읽고 buffer_B에 쓰기
- **홀수 반복** (1, 3, 5...): buffer_B에서 읽고 buffer_A에 쓰기
- **Stencil 연산**: 3점 평균 \\((\\text{left} + \\text{center} + \\text{right}) / 3\\)
- **경계 처리**: 버퍼 가장자리의 요소에 대해 적응적 평균 사용

**Memory barrier 조정:**

- **[mbarrier_arrive()](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_arrive)**: 각 스레드가 쓰기 단계 완료를 알림
- **[mbarrier_test_wait()](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_test_wait)**: 모든 스레드가 쓰기를 완료할 때까지 대기
- **버퍼 교체 안전성**: 다른 스레드가 아직 쓰고 있는 동안 버퍼에서 읽는 것을 방지
- **Barrier 재초기화**: 반복 간에 barrier 상태를 재설정

**출력 단계:**

- **최종 버퍼 선택**: 반복 횟수의 홀짝에 따라 활성 버퍼 선택
- **글로벌 메모리 쓰기**: 최종 결과를 출력 배열에 복사
- **완료 barrier**: 블록 종료 전 모든 쓰기 완료 보장

## 완성할 코드

```mojo
{{#include ../../../../../problems/p29/p29.mojo:double_buffered_stencil}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p29/p29.mojo" class="filename">전체 파일 보기: problems/p29/p29.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### **버퍼 초기화**

- `buffer_A`를 입력 데이터로 초기화하고, `buffer_B`는 빈 상태로 시작 가능
- 범위를 벗어난 요소에 대해 제로 패딩을 사용한 적절한 경계 검사
- 스레드 0만 mbarrier 객체를 초기화해야 함
- 서로 다른 동기화 지점에 별도의 barrier 설정

### **반복 제어**

- 컴파일 타임 루프 전개를 위해 `@parameter for iteration in range(STENCIL_ITERATIONS)` 사용
- `iteration % 2`를 사용하여 읽기/쓰기 할당을 교대하면서 버퍼 역할 결정
- 이웃 검사를 통해 유효한 범위 내에서만 stencil 연산 적용

### **Stencil 연산**

- 3점 평균 구현: `(left + center + right) / 3`
- 유효한 이웃만 평균에 포함하여 경계 조건 처리
- 엣지 케이스를 매끄럽게 처리하기 위해 적응적 카운팅 사용

### **Memory barrier 조정**

- 각 스레드가 쓰기 연산을 완료한 후 [`mbarrier_arrive()`](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_arrive) 호출
- 버퍼 교체 전 모든 스레드가 완료하도록 [`mbarrier_test_wait()`](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_test_wait) 사용
- 재사용을 위해 반복 간에 barrier 재초기화: [`mbarrier_init()`](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_init)
- 경쟁 상태를 피하기 위해 스레드 0만 barrier를 재초기화

### **출력 선택**

- `STENCIL_ITERATIONS % 2`를 기반으로 최종 활성 버퍼 선택
- 짝수 반복 횟수는 buffer_A에 데이터가 남음
- 홀수 반복 횟수는 buffer_B에 데이터가 남음
- 경계 검사를 통해 최종 결과를 글로벌 출력에 기록

</div>
</details>

## 코드 실행

풀이를 테스트하려면 터미널에서 다음 명령을 실행합니다:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p29 --double-buffer
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p29 --double-buffer
```

  </div>
  <div class="tab-content">

```bash
uv run poe p29 --double-buffer
```

  </div>
</div>

퍼즐을 성공적으로 완료하면 다음과 유사한 출력이 표시됩니다:

```
Puzzle 29: GPU Synchronization Primitives
==================================================
TPB: 256
SIZE: 1024
STENCIL_ITERATIONS: 3
BUFFER_COUNT: 2

Testing Puzzle 29B: Double-Buffered Stencil Computation
============================================================
Double-buffered stencil completed
Input sample: 1.0 1.0 1.0
GPU output sample: 1.0 1.0 1.0
✅ Double-buffered stencil test PASSED!
```

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p29/p29.mojo:double_buffered_stencil_solution}}
```

<div class="solution-explanation">

핵심 통찰은 이것이 명시적 memory barrier 조정을 사용하는 **double-buffering 아키텍처 문제**임을 인식하는 것입니다:

1. **교대하는 버퍼 역할 설계**: 매 반복마다 읽기/쓰기 책임을 교환
2. **명시적 memory barrier 구현**: 정밀한 동기화 제어를 위해 mbarrier API 사용
3. **반복 처리 조율**: 버퍼 교체 전 반복 결과가 완전히 완료되도록 보장
4. **메모리 접근 패턴 최적화**: 모든 처리를 빠른 공유 메모리에서 수행

<strong>상세 설명이 포함된 전체 풀이</strong>

Double-buffered stencil 풀이는 정교한 memory barrier 조정과 반복 처리 패턴을 보여줍니다. 이 접근 방식은 메모리 접근 타이밍에 대한 정밀한 제어가 필요한 안전한 반복적 개선 알고리즘을 가능하게 합니다.

## **Double-buffering 아키텍처 설계**

이 퍼즐의 근본적인 돌파구는 단순한 스레드 동기화가 아닌 **명시적 memory barrier 제어**입니다:

**전통적인 접근 방식:** 단순한 스레드 조정을 위해 기본 [`barrier()`](https://docs.modular.com/mojo/std/gpu/sync/sync/barrier/) 사용

- 모든 스레드가 서로 다른 데이터에 동일한 연산을 실행
- 단일 barrier 호출로 스레드 완료를 동기화
- 특정 메모리 연산 타이밍에 대한 제어 없음

**이 퍼즐의 혁신:** 명시적 memory barrier로 조정되는 서로 다른 버퍼 역할

- buffer_A와 buffer_B가 읽기 소스와 쓰기 대상 사이를 교대
- [mbarrier API](https://docs.modular.com/mojo/std/gpu/sync/sync/)가 메모리 연산 완료에 대한 정밀한 제어를 제공
- 명시적 조정으로 버퍼 전환 중 경쟁 상태를 방지

## **반복 처리 조율**

단일 패스 알고리즘과 달리, 이 퍼즐은 신중한 버퍼 관리를 통한 반복적 개선을 설정합니다:

- **반복 0**: buffer_A에서 읽기 (입력으로 초기화됨), buffer_B에 쓰기
- **반복 1**: buffer_B에서 읽기 (이전 결과), buffer_A에 쓰기
- **반복 2**: buffer_A에서 읽기 (이전 결과), buffer_B에 쓰기
- **교대 계속**: 각 반복이 이전 반복의 결과를 개선

## **Memory barrier API 사용법**

mbarrier 조정 패턴의 이해:

- **[mbarrier_init()](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_init)**: 특정 스레드 수(TPB)를 지정하여 barrier 초기화
- **[mbarrier_arrive()](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_arrive)**: 개별 스레드의 쓰기 단계 완료를 알림
- **[mbarrier_test_wait()](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_test_wait)**: 모든 스레드가 완료를 알릴 때까지 대기
- **재초기화**: 재사용을 위해 반복 간에 barrier 상태를 재설정

**핵심 타이밍 순서:**

1. **모든 스레드 쓰기**: 각 스레드가 할당된 버퍼 요소를 업데이트
2. **완료 알림**: 각 스레드가 [`mbarrier_arrive()`](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_arrive) 호출
3. **전체 대기**: 모든 스레드가 [`mbarrier_test_wait()`](https://docs.modular.com/mojo/std/gpu/sync/sync/mbarrier_test_wait) 호출
4. **진행 안전**: 이제 다음 반복을 위해 버퍼 역할을 안전하게 교체 가능

## **Stencil 연산 메커니즘**

적응적 경계 처리를 포함한 3점 stencil 연산:

**내부 요소** (인덱스 1부터 254):

```mojo
# 왼쪽, 중심, 오른쪽 이웃과의 평균
stencil_sum = buffer[i-1] + buffer[i] + buffer[i+1]
result[i] = stencil_sum / 3.0
```

**경계 요소** (인덱스 0과 255):

```mojo
# 유효한 이웃만 평균에 포함
stencil_count = 0
for neighbor in valid_neighbors:
    stencil_sum += buffer[neighbor]
    stencil_count += 1
result[i] = stencil_sum / stencil_count
```

## **버퍼 역할 교대**

핑퐁 버퍼 패턴이 데이터 무결성을 보장합니다:

**짝수 반복** (0, 2, 4...):

- **읽기 소스**: buffer_A에 현재 데이터 포함
- **쓰기 대상**: buffer_B가 업데이트된 결과를 수신
- **메모리 흐름**: buffer_A → stencil 연산 → buffer_B

**홀수 반복** (1, 3, 5...):

- **읽기 소스**: buffer_B에 현재 데이터 포함
- **쓰기 대상**: buffer_A가 업데이트된 결과를 수신
- **메모리 흐름**: buffer_B → stencil 연산 → buffer_A

## **경쟁 상태 방지**

Memory barrier가 여러 유형의 경쟁 상태를 제거합니다:

**Barrier 없이 (잘못된 경우)**:

```mojo
# 스레드 A가 buffer_B[10]에 쓰기
buffer_B[10] = stencil_result_A

# 스레드 B가 stencil 연산을 위해 buffer_B[10]을 즉시 읽기
# 경쟁 상태: 스레드 B가 스레드 A의 쓰기가 완료되기 전에 이전 값을 읽을 수 있음
stencil_input = buffer_B[10]  // 미정의 동작!
```

**Barrier 사용 (올바른 경우)**:

```mojo
# 모든 스레드가 결과를 쓰기
buffer_B[local_i] = stencil_result

# 쓰기 완료 알림
mbarrier_arrive(barrier)

# 모든 스레드의 쓰기 완료까지 대기
mbarrier_test_wait(barrier, TPB)

# 이제 읽기 안전 - 모든 쓰기 완료 보장
stencil_input = buffer_B[neighbor_index]  // 항상 올바른 값을 읽음
```

## **출력 버퍼 선택**

최종 결과 위치는 반복 횟수의 홀짝에 따라 결정됩니다:

**수학적 결정**:

- **STENCIL_ITERATIONS = 3** (홀수)
- **최종 활성 버퍼**: 반복 2가 buffer_B에 쓰기
- **출력 소스**: buffer_B에서 글로벌 메모리로 복사

**구현 패턴**:

```mojo
@parameter
if STENCIL_ITERATIONS % 2 == 0:
    # 짝수 총 반복 횟수는 buffer_A에서 종료
    output[global_i] = buffer_A[local_i]
else:
    # 홀수 총 반복 횟수는 buffer_B에서 종료
    output[global_i] = buffer_B[local_i]
```

## **성능 특성**

**메모리 계층 구조 최적화:**

- **글로벌 메모리**: 입력 로딩과 최종 출력에만 접근
- **공유 메모리**: 모든 반복 처리에 빠른 공유 메모리 사용
- **레지스터 사용량**: 공유 메모리 중심으로 최소화

**동기화 오버헤드:**

- **mbarrier 비용**: 기본 barrier()보다 높지만 필수적인 제어를 제공
- **반복 확장성**: 오버헤드가 반복 횟수에 비례하여 선형적으로 증가
- **스레드 효율성**: 모든 스레드가 처리 전반에 걸쳐 활성 상태 유지

## **실제 응용 분야**

이 double-buffering 패턴은 다음 분야의 기반이 됩니다:

**반복법:**

- 선형 시스템을 위한 Gauss-Seidel 및 Jacobi 방법
- 수치 정확도를 위한 반복적 개선
- 레벨별 처리를 수행하는 다중 그리드 방법

**이미지 처리:**

- 다중 패스 필터 (양측, 유도, 엣지 보존)
- 반복적 디노이징 알고리즘
- 열 확산과 이방성 스무딩

**시뮬레이션 알고리즘:**

- 상태 진화를 가진 셀룰러 오토마타
- 위치 업데이트를 수반하는 입자 시스템
- 반복적 압력 솔빙을 사용한 유체 역학

## **핵심 기술적 통찰**

**Memory barrier 철학:**

- **명시적 제어**: 자동 동기화 대비 메모리 연산에 대한 정밀한 타이밍 제어
- **경쟁 상태 방지**: 교대하는 읽기/쓰기 패턴을 가진 모든 알고리즘에 필수
- **성능 절충**: 보장된 정확성을 위한 더 높은 동기화 비용

**Double-buffering의 이점:**

- **데이터 무결성**: 쓰기 중 읽기 hazard 제거
- **알고리즘 명확성**: 현재와 다음 반복 상태 간의 깔끔한 분리
- **메모리 효율성**: 글로벌 메모리 중간 저장소 불필요

**반복 관리:**

- **컴파일 타임 루프 전개**: `@parameter for`가 최적화 기회를 제공
- **상태 추적**: 버퍼 역할 교대가 결정적이어야 함
- **경계 처리**: 적응적 stencil 연산이 엣지 케이스를 매끄럽게 처리

이 풀이는 정밀한 메모리 접근 제어가 필요한 반복 GPU 알고리즘을 설계하는 방법을 보여주며, 단순한 병렬 루프를 넘어 실제 수치 소프트웨어에서 사용되는 정교한 메모리 관리 패턴으로 나아갑니다.

</details>
