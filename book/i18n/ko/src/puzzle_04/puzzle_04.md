# Puzzle 4: 2D Map

{{ youtube EjmBmwgdAT0 breakpoint-lg }}

## 개요

2D 정사각 행렬 `a`의 각 위치에 10을 더해 2D 정사각 행렬 `output`에 저장하는 Kernel을 구현해 보세요.

**참고**: _스레드 수가 행렬의 위치 수보다 많습니다_.

{{ youtube EjmBmwgdAT0 breakpoint-sm }}

<img src="/puzzle_04/media/04a.png" alt="2D 행렬 매핑" class="light-mode-img">
<img src="/puzzle_04/media/04ad.png" alt="2D 행렬 매핑" class="dark-mode-img">

## 핵심 개념

- 2D 스레드 인덱싱
- GPU에서의 행렬 연산
- 초과 스레드 처리
- 메모리 레이아웃 패턴

각 위치 \\((i,j)\\)에 대해:
\\[\Large output[i,j] = a[i,j] + 10\\]

> ## 스레드 인덱싱 규칙
>
> GPU 프로그래밍에서 2D 행렬을 다룰 때는 스레드 인덱스와 행렬 좌표 사이의 자연스러운 매핑을 따릅니다:
>
> - `thread_idx.y`는 행(row) 인덱스
> - `thread_idx.x`는 열(column) 인덱스
>
> <img src="/puzzle_04/media/04b.png" alt="2D 스레드 인덱싱" class="light-mode-img">
> <img src="/puzzle_04/media/04bd.png" alt="2D 스레드 인덱싱" class="dark-mode-img">
>
> 이 규칙은 다음과 잘 맞습니다:
>
> 1. 행렬 위치를 (row, column)으로 쓰는 표준 수학 표기법
> 2. 행은 위에서 아래로(y축), 열은 왼쪽에서 오른쪽으로(x축) 가는 행렬의 시각적 구조
> 3. 스레드 블록을 행렬 구조에 맞춰 2D 그리드로 구성하는 일반적인 GPU 프로그래밍 패턴
>
> ### 역사적 배경
>
> 그래픽이나 이미지 처리에서는 보통 \\((x,y)\\) 좌표를 쓰지만, 행렬 연산에서는 전통적으로 (row, column) 인덱싱을 써왔습니다. 초기 컴퓨터가 2D 데이터를 저장하고 처리하던 방식에서 비롯된 것입니다: 위에서 아래로 한 줄씩, 각 줄은 왼쪽에서 오른쪽으로 읽었죠. 이런 row-major 메모리 레이아웃은 메모리를 순차적으로 접근하는 방식과 맞아서 CPU와 GPU 모두에서 효율적임이 입증되었습니다. GPU 프로그래밍에서 병렬 처리용 스레드 블록이 도입됐을 때, `thread_idx.y`를 행에, `thread_idx.x`를 열에 매핑한 건 기존에 확립된 행렬 인덱싱 규칙과 일관성을 유지하려는 자연스러운 선택이었습니다.

## 구현 방식

### [🔰 Raw 메모리 방식](./raw.md)

수동으로 메모리를 관리하면서 2D 인덱싱이 어떻게 동작하는지 알아봅니다.

### [📚 LayoutTensor 알아보기](./introduction_layout_tensor.md)

GPU에서 다차원 배열 연산과 메모리 관리를 간편하게 해주는 강력한 추상화를 소개합니다.

### [🚀 현대적 2D 연산](./layout_tensor.md)

자연스러운 2D 인덱싱과 자동 경계 검사를 갖춘 LayoutTensor를 직접 써봅니다.

💡 **참고**: 이 퍼즐부터는 더 깔끔하고 안전한 GPU 코드를 위해 LayoutTensor를 주로 사용합니다.
