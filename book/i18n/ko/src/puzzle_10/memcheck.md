<!-- i18n-source-commit: 88f1de7b3de457cc54820fd71512acbcadf073d9 -->

# 👮🏼‍♂️ 메모리 위반 탐지

## 개요

테스트가 통과하는 것처럼 보여도 GPU 프로그램을 조용히 손상시킬 수 있는 메모리 위반을 탐지하는 방법을 배웁니다. NVIDIA의 `compute-sanitizer`(`pixi`를 통해 사용 가능)와 `memcheck` 도구를 사용하여, GPU 코드에서 예측 불가능한 동작을 일으킬 수 있는 숨은 메모리 버그를 발견하게 됩니다.

**핵심 통찰**: GPU 프로그램은 불법적인 메모리 접근을 수행하면서도 동시에 "올바른" 결과를 만들어낼 수 있습니다.

**선행 학습**: [Puzzle 4 LayoutTensor](../puzzle_04/introduction_layout_tensor.md)와 기본적인 GPU 메모리 개념에 대한 이해가 필요합니다.

## 조용한 메모리 버그의 발견

### 테스트는 통과했지만, 코드가 정말 올바른 걸까?

얼핏 무해해 보이고 완벽하게 동작하는 듯한 프로그램으로 시작해 봅시다 (가드가 없는 [Puzzle 04](../puzzle_04/layout_tensor.md)입니다):

```mojo
{{#include ../../../../../problems/p10/p10.mojo:add_10_2d_no_guard}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p10/p10.mojo" class="filename">전체 파일 보기: problems/p10/p10.mojo</a>

이 프로그램을 일반적으로 실행하면, 모든 것이 정상으로 보입니다:

```bash
pixi run p10 --memory-bug
```

```txt
out shape: 2 x 2
Running memory bug example (bounds checking issue)...
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
✅ Memory test PASSED! (memcheck may find bounds violations)
```

✅ **테스트 통과!** 출력이 예상 결과와 완벽하게 일치합니다. 사건 종결, 맞죠?

**아닙니다!** `compute-sanitizer`가 무엇을 보여주는지 봅시다:

```bash
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=0 pixi run compute-sanitizer --tool memcheck mojo problems/p10/p10.mojo --memory-bug
```

**참고**: `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=0`은 디바이스 컨텍스트의 버퍼 캐시를 비활성화하는 명령줄 환경 변수 설정입니다. 이 설정은 일반적인 캐싱 동작에 의해 숨겨지던 경계 위반 같은 메모리 문제를 드러낼 수 있습니다. (_역주: 버퍼 캐시가 활성화되면 해제된 메모리를 즉시 반환하지 않고 재사용을 위해 보관합니다. 이 때문에 범위를 벗어난 접근이 아직 유효한 캐시 영역에 닿아 오류가 드러나지 않을 수 있습니다. 비활성화하면 메모리가 즉시 반환되어 위반이 감지됩니다._)

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running memory bug example (bounds checking issue)...

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (2,1,0) in block (0,0,0)
=========     Access at 0xe0c000210 is out of bounds
=========     and is 513 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (0,2,0) in block (0,0,0)
=========     Access at 0xe0c000210 is out of bounds
=========     and is 513 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (1,2,0) in block (0,0,0)
=========     Access at 0xe0c000214 is out of bounds
=========     and is 517 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Invalid __global__ read of size 4 bytes
=========     at p10_add_10_2d_...+0x80
=========     by thread (2,2,0) in block (0,0,0)
=========     Access at 0xe0c000218 is out of bounds
=========     and is 521 bytes after the nearest allocation at 0xe0c000000 of size 16 bytes

========= Program hit CUDA_ERROR_LAUNCH_FAILED (error 719) due to "unspecified launch failure" on CUDA API call to cuStreamSynchronize.
========= Program hit CUDA_ERROR_LAUNCH_FAILED (error 719) due to "unspecified launch failure" on CUDA API call to cuEventCreate.
========= Program hit CUDA_ERROR_LAUNCH_FAILED (error 719) due to "unspecified launch failure" on CUDA API call to cuMemFreeAsync.

========= ERROR SUMMARY: 7 errors
```

모든 테스트를 통과했음에도 프로그램에는 **총 7개의 오류**가 있습니다:

- **4개의 메모리 위반** (Invalid **global** read)
- **3개의 런타임 오류** (메모리 위반으로 인해 발생)

## 숨겨진 버그 이해하기

### 근본 원인 분석

**문제:**

- **텐서 크기**: 2×2 (유효한 인덱스: 0, 1)
- **스레드 그리드**: 3×3 (스레드 인덱스: 0, 1, 2)
- **범위 초과 스레드**: `(2,1)`, `(0,2)`, `(1,2)`, `(2,2)`가 잘못된 메모리에 접근
- **경계 검사 누락**: 텐서 차원에 대한 `thread_idx` 검증이 없음

### 7개 오류 전체 이해하기

**4개의 메모리 위반:**

- 각 범위 초과 스레드 `(2,1)`, `(0,2)`, `(1,2)`, `(2,2)`가 "Invalid **global** read"를 발생시킴

**3개의 CUDA 런타임 오류:**

- Kernel 실행 실패로 인해 `cuStreamSynchronize` 실패
- 정리 과정에서 `cuEventCreate` 실패
- 메모리 해제 과정에서 `cuMemFreeAsync` 실패

**핵심 통찰**: 메모리 위반은 연쇄 효과를 일으킵니다 - 하나의 잘못된 메모리 접근이 여러 후속 CUDA API 실패를 야기합니다.

**그럼에도 테스트가 통과한 이유:**

- 유효한 스레드 `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`이 올바른 결과를 기록함
- 테스트가 유효한 출력 위치만 검사함
- 범위 초과 접근이 프로그램을 즉시 크래시시키지 않음

## 미정의 동작(UB) 이해하기

### 미정의 동작이란?

**미정의 동작(Undefined Behavior, UB)** 은 프로그램이 언어 명세상 정의되지 않은 연산을 수행할 때 발생합니다. 범위 초과 메모리 접근이 대표적인 예입니다.

**UB의 주요 특성:**

- 프로그램이 **말 그대로 무슨 짓이든** 할 수 있음: 크래시, 잘못된 결과, 정상 동작하는 것처럼 보이기, 메모리 손상
- **어떤 보장도 없음**: 컴파일러, 하드웨어, 드라이버, 심지어 실행할 때마다 동작이 달라질 수 있음

### 미정의 동작이 특히 위험한 이유

**정확성 문제:**

- **예측 불가능한 결과**: 테스트 중에는 동작하다가 프로덕션에서 실패할 수 있음
- **비결정적 동작**: 같은 코드가 다른 실행에서 다른 결과를 낼 수 있음
- **조용한 손상**: UB는 가시적인 오류 없이 데이터를 손상시킬 수 있음
- **컴파일러 최적화**: 컴파일러는 UB가 없다고 가정하고 예상치 못한 방식으로 최적화할 수 있음

**보안 취약점:**

- **버퍼 오버플로우**: 시스템 프로그래밍에서 보안 공격의 고전적인 원인
- **메모리 손상**: 권한 상승이나 코드 인젝션 공격으로 이어질 수 있음
- **정보 유출**: 범위를 벗어난 읽기로 민감한 데이터가 노출될 수 있음
- **제어 흐름 하이재킹**: UB를 악용해 프로그램 실행 흐름을 탈취할 수 있음

### GPU 특유의 미정의 동작 위험성

**대규모 영향:**

- **스레드 분기**: 한 스레드의 UB가 전체 Warp(32개 스레드)에 영향을 줄 수 있음
- **메모리 병합(coalescing)**: 범위 초과 접근이 인접 스레드의 데이터를 손상시킬 수 있음
- **Kernel 실패**: UB가 GPU Kernel 전체를 완전히 망가뜨릴 수 있음

**하드웨어 차이:**

- **다른 GPU 아키텍처**: UB가 다른 GPU 모델에서 다르게 나타날 수 있음
- **드라이버 차이**: 같은 UB가 드라이버 버전에 따라 다르게 동작할 수 있음
- **메모리 레이아웃 변경**: GPU 메모리 할당 패턴에 따라 UB가 다르게 나타날 수 있음

## 메모리 위반 수정하기

### 해결책

[Puzzle 04](../puzzle_04/layout_tensor.md)에서 본 것처럼, 다음과 같이 경계 검사를 해야 합니다:

```mojo
{{#include ../../../../../solutions/p04/p04_layout_tensor.mojo:add_10_2d_layout_tensor_solution}}
```

해결책은 간단합니다: **메모리에 접근하기 전에 항상 스레드 인덱스를 데이터 차원에 대해 검증**하세요.

### compute-sanitizer로 검증

```bash
# p10.mojo 복사본에서 경계 검사를 수정한 후 실행:
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=0 pixi run compute-sanitizer --tool memcheck mojo problems/p10/p10.mojo --memory-bug
```

```txt
========= COMPUTE-SANITIZER
out shape: 2 x 2
Running memory bug example (bounds checking issue)...
out: HostBuffer([10.0, 11.0, 12.0, 13.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
✅ Memory test PASSED! (memcheck may find bounds violations)
========= ERROR SUMMARY: 0 errors
```

**✅ 성공:** 메모리 위반이 탐지되지 않았습니다!

## 핵심 학습 포인트

### 수동 경계 검사가 중요한 이유

1. **명확성**: 코드에서 안전 요구사항을 명시적으로 표현
2. **제어**: 범위 초과 케이스에서 정확히 어떤 일이 일어날지 직접 결정
3. **디버깅**: 메모리 위반이 발생할 때 추론하기 쉬움

### GPU 메모리 안전 규칙

1. **항상 스레드 인덱스를 검증**하여 데이터 차원과 비교
2. **미정의 동작(UB)을 어떤 대가를 치르더라도 피하기** - 범위 초과 접근은 UB이며 모든 것을 망가뜨릴 수 있음
3. **개발과 테스트 중 compute-sanitizer 사용**
4. **메모리 검사 없이 "동작한다"고 절대 가정하지 않기**
5. **다양한 그리드/블록 구성으로 테스트**하여 일관성 없이 나타나는 미정의 동작(UB) 포착

### compute-sanitizer 모범 사례

```bash
MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=0 pixi run compute-sanitizer --tool memcheck mojo your_code.mojo
```

**참고**: sanitizer 출력에서 Mojo 런타임 경고를 볼 수 있습니다. 실제 메모리 위반을 확인하려면 `========= Invalid`와 `========= ERROR SUMMARY` 라인에 집중하세요.
