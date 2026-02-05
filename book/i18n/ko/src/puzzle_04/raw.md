<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

## 개요

2D 정사각 행렬 `a`의 각 위치에 10을 더해 2D 정사각 행렬 `output`에 저장하는 Kernel을 구현해 보세요.

**참고**: _스레드 수가 행렬의 위치 수보다 많습니다_.

## 핵심 개념

이 퍼즐에서 배울 내용:

- 2D 스레드 인덱스 다루기 (`thread_idx.x`, `thread_idx.y`)
- 2D 좌표를 1D 메모리 인덱스로 변환하기
- 2차원에서 경계 검사 처리하기

핵심은 2D 스레드 좌표 \\((i,j)\\)를 크기 \\(n \times n\\)인 row-major 행렬의 원소로 매핑하는 방법을 이해하는 것입니다. 동시에 스레드 인덱스가 범위를 벗어나지 않는지도 확인해야 합니다.

- **2D 인덱싱**: 각 스레드가 고유한 \\((i,j)\\) 위치를 가짐
- **메모리 레이아웃**: Row-major 순서로 2D를 1D 메모리에 매핑
- **가드 조건**: 두 차원 모두 경계 검사 필요
- **스레드 범위**: 스레드 \\((3 \times 3)\\)가 행렬 원소 \\((2 \times 2)\\)보다 많음

## 작성할 코드

```mojo
{{#include ../../../../../problems/p04/p04.mojo:add_10_2d}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p04/p04.mojo" class="filename">전체 코드 보기: problems/p04/p04.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 2D 인덱스 가져오기: `row = thread_idx.y`, `col = thread_idx.x`
2. 가드 추가: `if row < size and col < size`
3. 가드 내부에서 row-major 방식으로 10 더하기!

</div>
</details>

## 코드 실행

솔루션을 테스트하려면 터미널에서 다음 명령어를 실행하세요:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p04
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p04
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p04
```

  </div>
  <div class="tab-content">

```bash
uv run poe p04
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력이 다음과 같이 나타납니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p04/p04.mojo:add_10_2d_solution}}
```

<div class="solution-explanation">

이 솔루션은:

1. 2D 인덱스 가져오기: `row = thread_idx.y`, `col = thread_idx.x`
2. 가드 추가: `if row < size and col < size`
3. 가드 내부: `output[row * size + col] = a[row * size + col] + 10.0`

</div>
</details>
