# Puzzle 1: Map

{{ youtube rLhjprX8Nck breakpoint-lg }}

## 개요

이 퍼즐에서는 GPU 병렬 처리의 기본 개념을 다룹니다. 각 스레드가 데이터 요소 하나를 맡아 동시에 처리하는 방식을 배웁니다.
벡터 `a`의 각 요소에 10을 더해 `output`에 저장하는 Kernel을 구현해 보세요.

**참고:** _각 위치마다 스레드 1개가 배정됩니다._

{{ youtube rLhjprX8Nck breakpoint-sm }}

<img src="/puzzle_01/media/01.png" alt="Map" class="light-mode-img">
<img src="/puzzle_01/media/01d.png" alt="Map" class="dark-mode-img">

## 핵심 개념

- GPU Kernel의 기본 구조
- 스레드와 데이터 간 일대일 매핑
- 메모리 접근 패턴
- GPU에서의 배열 연산

각 위치 \\(i\\)에 대해:
\\[\Large output[i] = a[i] + 10\\]

## 다루는 내용

### [🔰 Raw 메모리 방식](./raw.md)

직접 메모리를 다루며 GPU의 기본 원리를 익힙니다.

### [💡 미리보기: LayoutTensor를 활용한 현대적 방식](./layout_tensor_preview.md)

LayoutTensor가 GPU 프로그래밍을 어떻게 단순화하는지 살펴봅니다. 더 안전하고 깔끔한 코드를 작성할 수 있습니다.

💡 **팁**: 두 방식을 모두 익히면 현대적인 GPU 프로그래밍 패턴을 더 깊이 이해할 수 있습니다.
