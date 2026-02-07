<!-- i18n-source-commit: 5026a500b7a7ae33256e0b344629e99c7d0f50da -->

# Puzzle 11: Pooling

## 개요

벡터 `a`에서 각 위치의 직전 3개 값의 합을 계산하여 벡터 `output`에 저장하는 kernel을 구현하세요.

**참고:** _각 위치마다 스레드 1개가 있습니다. 스레드당 global read 1회, global write 1회만 필요합니다._

<img src="./media/11-w.png" alt="Pooling 시각화" class="light-mode-img">
<img src="./media/11-b.png" alt="Pooling 시각화" class="dark-mode-img">

## 구현 방식

### [🔰 Raw 메모리 방식](./raw.md)

슬라이딩 윈도우 연산을 수동 메모리 관리와 동기화로 직접 구현하는 방법을 알아봅니다.

### [📐 LayoutTensor 버전](./layout_tensor.md)

LayoutTensor의 기능을 활용해 효율적인 윈도우 기반 연산과 공유 메모리 관리를 구현합니다.

💡 **참고**: LayoutTensor로 슬라이딩 윈도우 연산이 얼마나 간결해지는지 확인해 보세요. 효율적인 메모리 접근 패턴도 그대로 유지됩니다.
