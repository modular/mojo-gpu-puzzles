<!-- i18n-source-commit: 5026a500b7a7ae33256e0b344629e99c7d0f50da -->

# Puzzle 12: 내적

## 개요

벡터 `a`와 벡터 `b`의 내적을 계산하여 `output`(단일 값)에 저장하는 kernel을 구현하세요. 내적은 크기가 같은 두 벡터에서 대응하는 원소끼리 곱한 뒤, 그 결과를 모두 더해 하나의 숫자(스칼라)를 구하는 연산입니다.

예를 들어, 두 벡터가 다음과 같을 때:

\\[a = [a_{1}, a_{2}, ..., a_{n}] \\]
\\[b = [b_{1}, b_{2}, ..., b_{n}] \\]

내적은 이렇게 구합니다:
\\[a \\cdot b = a_{1}b_{1} +  a_{2}b_{2} + ... + a_{n}b_{n}\\]

**참고:** _각 위치마다 스레드 1개가 있습니다. 스레드당 global read 2회, 블록당 global write 1회만 필요합니다._

<img src="/puzzle_12/media/12-w.png" alt="내적 시각화" class="light-mode-img">
<img src="/puzzle_12/media/12-b.png" alt="내적 시각화" class="dark-mode-img">

## 구현 방식

### [🔰 Raw 메모리 방식](./raw.md)

수동 메모리 관리와 동기화로 reduction을 밑바닥부터 구현하는 방법을 알아봅니다.

### [📐 LayoutTensor 버전](./layout_tensor.md)

LayoutTensor를 활용해 reduction과 공유 메모리 관리를 더 간결하게 구현합니다.

💡 **참고**: LayoutTensor로 메모리 접근 패턴이 얼마나 깔끔해지는지 확인해 보세요. 효율은 그대로입니다.
