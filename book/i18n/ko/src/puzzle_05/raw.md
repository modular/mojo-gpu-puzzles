<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

## 개요

벡터 `a`와 `b`를 broadcast로 더해 2D 행렬 `output`에 저장하는 kernel을 구현해 보세요.

**참고**: _스레드 수가 행렬의 위치 수보다 많습니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- 1D 벡터를 각각 다른 차원 방향으로 broadcast하기
- 2D 스레드 인덱스로 broadcast 연산 수행하기
- broadcast 패턴에서 경계 조건 처리하기

핵심은 두 1D 벡터의 원소들을 broadcast로 2D 출력 행렬에 매핑하는 방법을 이해하고, 스레드 경계를 올바르게 처리하는 것입니다.

- **Broadcast**: `a`의 각 원소가 `b`의 각 원소와 결합
- **스레드 매핑**: \\(2 \times 2\\) 출력에 \\((3 \times 3)\\) 스레드 그리드 사용
- **벡터 접근**: `a`와 `b`는 서로 다른 접근 패턴 사용
- **경계 검사**: 행렬 범위를 벗어나는 스레드를 guard로 처리

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
3. 가드 내부: `a`와 `b` 값을 어떻게 broadcast할지 생각해 보세요

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

LayoutTensor 추상화 없이 GPU broadcast의 기본 개념을 보여주는 솔루션입니다:

1. **스레드에서 행렬로 매핑**
   - `thread_idx.y`로 행, `thread_idx.x`로 열에 접근
   - 2D 스레드 그리드를 출력 행렬 원소에 직접 매핑
   - 3×3 그리드의 초과 스레드를 2×2 출력에 맞게 처리

2. **Broadcast 작동 방식**
   - 벡터 `a`는 수평 방향으로 broadcast: 각 행에서 동일한 `a[col]` 사용
   - 벡터 `b`는 수직 방향으로 broadcast: 각 열에서 동일한 `b[row]` 사용
   - 두 벡터를 더해 출력 생성

   ```txt
   [ a0 a1 ]  +  [ b0 ]  =  [ a0+b0  a1+b0 ]
                 [ b1 ]     [ a0+b1  a1+b1 ]
   ```

3. **경계 검사**
   - 단일 guard 조건 `row < size and col < size`로 두 차원 모두 처리
   - 입력 벡터와 출력 행렬의 범위 초과 접근 방지
   - 3×3 스레드 그리드가 2×2 데이터보다 크므로 반드시 필요

LayoutTensor 버전과 비교해서 동일한 기본 개념을 유지하면서 추상화가 broadcast 연산을 얼마나 단순하게 만드는지 확인해 보세요.
</div>
</details>
