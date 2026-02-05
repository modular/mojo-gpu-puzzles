<!-- i18n-source-commit: 9ac1b899ca05c1be26f2d9ee77fe97503d00cc0f -->

# LayoutTensor 알아보기

퍼즐 풀이를 잠시 멈추고, GPU 프로그래밍을 더 즐겁게 만들어줄 강력한 추상화를 미리 살펴봅시다:
🥁 ... 바로 **[LayoutTensor](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/)**입니다.

> 💡 _LayoutTensor가 어떤 일을 할 수 있는지 맛보기로 살펴봅니다. 지금 모든 걸 이해할 필요는 없어요 - 퍼즐을 진행하면서 각 기능을 자세히 알아볼 겁니다_.

## 문제: 점점 복잡해지는 코드

지금까지 겪은 어려움을 살펴봅시다:

```mojo
# Puzzle 1: 단순 인덱싱
output[i] = a[i] + 10.0

# Puzzle 2: 여러 배열 관리
output[i] = a[i] + b[i]

# Puzzle 3: 경계 검사
if i < size:
    output[i] = a[i] + 10.0
```

차원이 늘어나면 코드는 더 복잡해집니다:

```mojo
# 전통적인 2D 인덱싱 (row-major 2D 행렬)
idx = row * WIDTH + col
if row < height and col < width:
    output[idx] = a[idx] + 10.0
```

## 해결책: LayoutTensor 미리보기

LayoutTensor는 이런 문제들을 깔끔하게 해결해줍니다. 앞으로 배울 내용을 살짝 엿보면:

1. **자연스러운 인덱싱**: 수동 오프셋 계산 대신 `tensor[i, j]` 사용
2. **유연한 메모리 레이아웃**: Row-major, column-major, tiled 구성 지원
3. **성능 최적화**: GPU에 효율적인 메모리 접근 패턴

## 앞으로 배울 내용 맛보기

LayoutTensor가 할 수 있는 일을 몇 가지 예시로 살펴봅시다. 지금 모든 세부 사항을 이해할 필요는 없습니다 - 앞으로 나올 퍼즐에서 각 기능을 꼼꼼히 다룰 거예요.

### 기본 사용 예시

```mojo
from layout import Layout, LayoutTensor

# 레이아웃 정의
comptime HEIGHT = 2
comptime WIDTH = 3
comptime layout = Layout.row_major(HEIGHT, WIDTH)

# 텐서 생성
tensor = LayoutTensor[dtype, layout](buffer.unsafe_ptr())

# 자연스럽게 요소 접근
tensor[0, 0] = 1.0  # 첫 번째 요소
tensor[1, 2] = 2.0  # 마지막 요소
```

`Layout`과 `LayoutTensor`에 대해 더 알아보려면 [Mojo 매뉴얼](https://docs.modular.com/mojo/manual/)의 가이드를 참고하세요:

- [Introduction to layouts](https://docs.modular.com/mojo/manual/layout/layouts)
- [Using LayoutTensor](https://docs.modular.com/mojo/manual/layout/tensors)

## 간단한 예제

LayoutTensor의 기본을 보여주는 간단한 예제로 모든 것을 정리해봅시다:

```mojo
{{#include ../../../../src/puzzle_04/intro.mojo}}
```

다음 명령어로 이 코드를 실행하면:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run layout_tensor_intro
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd layout_tensor_intro
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple layout_tensor_intro
```

  </div>
  <div class="tab-content">

```bash
uv run poe layout_tensor_intro
```

  </div>
</div>

```txt
Before:
0.0 0.0 0.0
0.0 0.0 0.0
After:
1.0 0.0 0.0
0.0 0.0 0.0
```

무슨 일이 일어나는지 살펴봅시다:

1. Row-major 레이아웃으로 `2 x 3` 텐서를 생성합니다
2. 처음에는 모든 요소가 0입니다
3. 자연스러운 인덱싱으로 하나의 요소를 수정합니다
4. 변경 사항이 출력에 반영됩니다

이 간단한 예제는 LayoutTensor의 핵심 장점을 보여줍니다:

- 텐서 생성과 접근을 위한 깔끔한 문법
- 자동 메모리 레이아웃 처리
- 자연스러운 다차원 인덱싱

이 예제는 간단하지만, 같은 패턴이 앞으로 나올 퍼즐의 복잡한 GPU 연산에도 그대로 적용됩니다. 이런 기본 개념이 다음으로 어떻게 확장되는지 보게 될 거예요:

- 멀티 스레드 GPU 연산
- 공유 메모리 최적화
- 복잡한 tiling 전략
- 하드웨어 가속 연산

LayoutTensor와 함께 GPU 프로그래밍 여정을 시작할 준비가 됐나요? 퍼즐로 들어가봅시다!

💡 **팁**: 진행하면서 이 예제를 기억해두세요 - 이 기본 개념을 바탕으로 점점 더 정교한 GPU 프로그램을 만들어갈 겁니다.
