# Puzzle 7: 2D Blocks

## 개요

행렬 `a`의 각 위치에 10을 더해 `output`에 저장하는 kernel을 구현해 보세요.

**참고:** _블록당 스레드 수가 `a`의 행과 열 크기보다 모두 작습니다._

<img src="/puzzle_07/media/07.png" alt="2D Blocks 시각화" class="light-mode-img">
<img src="/puzzle_07/media/07d.png" alt="2D Blocks 시각화" class="dark-mode-img">

## 핵심 개념

- 블록 기반 처리
- 그리드와 블록의 조율
- 여러 블록에 걸친 인덱싱
- 메모리 접근 패턴

> 🔑 **2D 스레드 인덱싱 방식**
>
> [Puzzle 4: 2D Map](../puzzle_04/puzzle_04.md)의 블록 기반 인덱싱을 2D로 확장합니다:
>
> ```txt
> 전역 위치 계산:
> row = block_dim.y * block_idx.y + thread_idx.y
> col = block_dim.x * block_idx.x + thread_idx.x
> ```
>
> 예를 들어, 4×4 그리드에서 2×2 블록을 사용하면:
>
> ```txt
> Block (0,0):   Block (1,0):
> [0,0  0,1]     [0,2  0,3]
> [1,0  1,1]     [1,2  1,3]
>
> Block (0,1):   Block (1,1):
> [2,0  2,1]     [2,2  2,3]
> [3,0  3,1]     [3,2  3,3]
> ```
>
> 각 위치는 해당 스레드의 전역 인덱스 (row, col)를 나타냅니다.
> 블록 차원과 인덱스가 함께 작동하여 다음을 보장합니다:
>
> - 2D 공간 전체를 빈틈없이 처리
> - 블록 간 겹침 없음
> - 효율적인 메모리 접근 패턴

## 구현 방식

### [🔰 Raw 메모리 방식](./raw.md)

수동 인덱싱으로 여러 블록에 걸친 연산을 처리하는 방법을 알아봅니다.

### [📐 LayoutTensor 버전](./layout_tensor.md)

LayoutTensor 기능을 활용해 블록 기반 처리를 깔끔하게 구현합니다.

💡 **참고**: LayoutTensor가 블록 간 조율과 메모리 접근 패턴을 얼마나 단순화하는지 확인해 보세요.
