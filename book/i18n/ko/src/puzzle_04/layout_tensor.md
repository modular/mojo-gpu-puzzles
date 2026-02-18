<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

# LayoutTensor 버전

## 개요

2D _LayoutTensor_ `a`의 각 위치에 10을 더해 2D _LayoutTensor_ `output`에 저장하는 Kernel을 구현해 보세요.

**참고**: _스레드 수가 행렬의 위치 수보다 많습니다_.

## 핵심 개념

이 퍼즐에서 배울 내용:

- 2D 배열 접근에 `LayoutTensor` 사용하기
- `tensor[i, j]`로 직접 2D 인덱싱하기
- `LayoutTensor`에서 경계 검사 처리하기

핵심은 `LayoutTensor`가 자연스러운 2D 인덱싱 인터페이스를 제공하여 내부 메모리 레이아웃을 추상화한다는 점입니다. 그러면서도 경계 검사는 여전히 필요합니다.

- **2D 접근**: `LayoutTensor`로 자연스러운 \\((i,j)\\) 인덱싱
- **메모리 추상화**: 수동 row-major 계산 불필요
- **가드 조건**: 두 차원 모두 경계 검사 필요
- **스레드 범위**: 스레드 \\((3 \times 3)\\)가 텐서 원소 \\((2 \times 2)\\)보다 많음

## 완성할 코드

```mojo
{{#include ../../../../../problems/p04/p04_layout_tensor.mojo:add_10_2d_layout_tensor}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p04/p04_layout_tensor.mojo" class="filename">전체 코드 보기: problems/p04/p04_layout_tensor.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 2D 인덱스 가져오기: `row = thread_idx.y`, `col = thread_idx.x`
2. 가드 추가: `if row < size and col < size`
3. 가드 내부에서 `a[row, col]`에 10 더하기

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
pixi run p04_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p04_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p04_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p04_layout_tensor
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
{{#include ../../../../../solutions/p04/p04_layout_tensor.mojo:add_10_2d_layout_tensor_solution}}
```

<div class="solution-explanation">

이 솔루션은:

- `row = thread_idx.y`, `col = thread_idx.x`로 2D 스레드 인덱스를 가져옴
- `if row < size and col < size`로 범위를 벗어난 접근 방지
- `LayoutTensor`의 2D 인덱싱 사용: `output[row, col] = a[row, col] + 10.0`

</div>
</details>
