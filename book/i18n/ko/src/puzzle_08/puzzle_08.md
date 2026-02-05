<!-- i18n-source-commit: 8726393ce8f2ba4d52d2ceec6352706da1f1806a -->

# Puzzle 8: Shared Memory

## 개요

벡터 `a`의 각 위치에 10을 더해 벡터 `output`에 저장하는 kernel을 구현해 보세요.

**참고:** _블록당 스레드 수가 `a`의 크기보다 작습니다._

<img src="/puzzle_08/media/08.png" alt="공유 메모리 시각화" class="light-mode-img">
<img src="/puzzle_08/media/08d.png" alt="공유 메모리 시각화" class="dark-mode-img">

## 구현 방식

### [🔰 Raw 메모리 방식](./raw.md)

공유 메모리와 동기화를 수동으로 관리하는 방법을 알아봅니다.

### [📐 LayoutTensor 버전](./layout_tensor.md)

LayoutTensor에 내장된 공유 메모리 관리 기능을 활용합니다.

💡 **참고**: LayoutTensor가 성능을 유지하면서도 공유 메모리 연산을 얼마나 간소화하는지 경험해 보세요.
