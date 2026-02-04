# LayoutTensor 버전

## 개요

1D LayoutTensor `a`와 `b`를 broadcast로 더해 2D LayoutTensor `output`에 저장하는 kernel을 구현해 보세요.

**참고**: _스레드 수가 행렬의 위치 수보다 많습니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- broadcast 연산에 `LayoutTensor` 사용하기
- 서로 다른 텐서 shape 다루기
- `LayoutTensor`로 2D 인덱싱 처리하기

핵심은 `LayoutTensor`가 서로 다른 텐서 shape \\((1, n)\\)과 \\((n, 1)\\)을 \\((n,n)\\)으로 자연스럽게 broadcast할 수 있다는 점입니다. 그러면서도 경계 검사는 여전히 필요합니다.

- **텐서 shape**: 입력 벡터의 shape은 \\((1, n)\\)과 \\((n, 1)\\)
- **Broadcast**: 두 차원을 결합해 \\((n,n)\\) 출력 생성
- **가드 조건**: 출력 크기에 대한 경계 검사는 여전히 필요
- **스레드 범위**: 텐서 원소 \\((2 \times 2)\\)보다 스레드 \\((3 \times 3)\\)가 많음

## 작성할 코드

```mojo
{{#include ../../../../../problems/p05/p05_layout_tensor.mojo:broadcast_add_layout_tensor}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p05/p05_layout_tensor.mojo" class="filename">전체 코드 보기: problems/p05/p05_layout_tensor.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 2D 인덱스 가져오기: `row = thread_idx.y`, `col = thread_idx.x`
2. 가드 추가: `if row < size and col < size`
3. 가드 내부: LayoutTensor로 `a`와 `b` 값을 어떻게 broadcast할지 생각해 보세요

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
pixi run p05_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p05_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p05_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p05_layout_tensor
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력이 다음과 같이 나타납니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 2.0, 11.0, 12.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p05/p05_layout_tensor.mojo:broadcast_add_layout_tensor_solution}}
```

<div class="solution-explanation">

LayoutTensor broadcast와 GPU 스레드 매핑의 핵심 개념을 보여주는 솔루션입니다:

1. **스레드에서 행렬로 매핑**

   - `thread_idx.y`로 행, `thread_idx.x`로 열에 접근
   - 자연스러운 2D 인덱싱이 출력 행렬 구조와 일치
   - 초과 스레드(3×3 그리드)는 경계 검사로 처리

2. **Broadcast 작동 방식**
   - 입력 `a`의 shape은 `(1,n)`: `a[0,col]`이 행을 가로질러 broadcast
   - 입력 `b`의 shape은 `(n,1)`: `b[row,0]`이 열을 가로질러 broadcast
   - 출력의 shape은 `(n,n)`: 각 원소는 해당 broadcast 값들의 합

   ```txt
   [ a0 a1 ]  +  [ b0 ]  =  [ a0+b0  a1+b0 ]
                 [ b1 ]     [ a0+b1  a1+b1 ]
   ```

3. **경계 검사**
   - guard 조건 `row < size and col < size`로 범위 초과 접근 방지
   - 행렬 범위와 초과 스레드를 효율적으로 처리
   - broadcast 덕분에 `a`와 `b`에 대한 별도 검사 불필요

이 패턴은 이후 퍼즐에서 다룰 더 복잡한 텐서 연산의 기초가 됩니다.
</div>
</details>
