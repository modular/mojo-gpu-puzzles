# Puzzle 5: Broadcast

## 개요

벡터 `a`와 `b`를 broadcast로 더해 2D 행렬 `output`에 저장하는 kernel을 구현해 보세요.
(_역주: broadcast란 작은 차원의 데이터를 큰 차원으로 확장하여 연산하는 것으로, 여기서는 두 1D 벡터를 2D 행렬로 확장해서 더합니다._)

**참고**: _스레드 수가 행렬의 위치 수보다 많습니다._

<img src="/puzzle_05/media/05.png" alt="Broadcast 시각화" class="light-mode-img">
<img src="/puzzle_05/media/05d.png" alt="Broadcast 시각화" class="dark-mode-img">

## 핵심 개념

- 벡터를 행렬로 broadcast하기
- 2D 스레드 관리
- 서로 다른 차원 간 연산
- 메모리 레이아웃 패턴

## 구현 방식

### [🔰 Raw 메모리 방식](./raw.md)

수동 메모리 인덱싱으로 broadcast를 처리하는 방법을 알아봅니다.

### [📐 LayoutTensor 버전](./layout_tensor.md)

서로 다른 차원 간 연산을 LayoutTensor로 처리합니다.

💡 **참고**: 수동 인덱싱과 비교했을 때 LayoutTensor가 broadcast를 얼마나 간단하게 만들어주는지 확인해 보세요.
