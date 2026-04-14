<!-- i18n-source-commit: 430a6f13dcfdbf8b417585a7b28f41a575467d2b -->

# Puzzle 5: 브로드캐스트

## 개요

1D TileTensor `a`와 `b`를 브로드캐스트로 더해 2D TileTensor `output`에 저장하는 커널을 구현해 보세요.

**참고**: _스레드 수가 행렬의 위치 수보다 많습니다._

<img src="/puzzle_05/media/05.png" alt="Broadcast 시각화" class="light-mode-img">
<img src="/puzzle_05/media/05d.png" alt="Broadcast 시각화" class="dark-mode-img">

## 핵심 개념

이 퍼즐에서 배울 내용:

- 브로드캐스트 연산에 `TileTensor` 사용하기
- 서로 다른 텐서 크기 다루기
- `TileTensor`로 2D 인덱싱 처리하기

핵심은 `TileTensor`가 서로 다른 텐서 크기 \\((1, n)\\)와 \\((n, 1)\\)을 \\((n,n)\\)으로 자연스럽게 브로드캐스트할 수 있다는 점입니다. 그러면서도 경계 검사는 여전히 필요합니다.

- **텐서 크기**: 입력 벡터의 크기는 \\((1, n)\\)과 \\((n, 1)\\)
- **브로드캐스트**: 두 차원을 결합해 \\((n,n)\\) 출력 생성
- **가드 조건**: 출력 크기에 대한 경계 검사는 여전히 필요
- **스레드 범위**: 텐서 원소 \\((2 \times 2)\\)보다 스레드 \\((3 \times 3)\\)가 많음

## 완성할 코드

```mojo
{{#include ../../../../../problems/p05/p05.mojo:broadcast_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p05/p05.mojo" class="filename">전체 코드 보기: problems/p05/p05.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 2D 인덱스 가져오기: `row = thread_idx.y`, `col = thread_idx.x`
2. 가드 추가: `if row < size and col < size`
3. 가드 내부: TileTensor로 `a`와 `b` 값을 어떻게 브로드캐스트할지 생각해 보세요

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
pixi run p05
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p05
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p05
```

  </div>
  <div class="tab-content">

```bash
uv run poe p05
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
{{#include ../../../../../solutions/p05/p05.mojo:broadcast_add_solution}}
```

<div class="solution-explanation">

TileTensor 브로드캐스트와 GPU 스레드 매핑의 핵심 개념을 보여주는 솔루션입니다:

1. **스레드에서 행렬로 매핑**

   - `thread_idx.y`로 행, `thread_idx.x`로 열에 접근
   - 자연스러운 2D 인덱싱이 출력 행렬 구조와 일치
   - 초과 스레드(3×3 그리드)는 경계 검사로 처리

2. **브로드캐스트 작동 방식**
   - 입력 `a`의 크기는 `(1,n)`: `a[0,col]`이 행을 가로질러 브로드캐스트
   - 입력 `b`의 크기는 `(n,1)`: `b[row,0]`이 열을 가로질러 브로드캐스트
   - 출력의 크기는 `(n,n)`: 각 원소는 해당 브로드캐스트 값들의 합

   ```txt
   [ a0 a1 ]  +  [ b0 ]  =  [ a0+b0  a1+b0 ]
                 [ b1 ]     [ a0+b1  a1+b1 ]
   ```

3. **경계 검사**
   - 가드 조건 `row < size and col < size`로 범위 초과 접근 방지
   - 행렬 범위와 초과 스레드를 효율적으로 처리
   - 브로드캐스트 덕분에 `a`와 `b`에 대한 별도 검사 불필요

이 패턴은 이후 퍼즐에서 다룰 더 복잡한 텐서 연산의 기초가 됩니다.
</div>
</details>
