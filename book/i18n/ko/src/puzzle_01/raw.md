## 핵심 개념

이 퍼즐에서 배우는 내용:

- 기본 GPU Kernel 구조
- `thread_idx.x`를 사용한 스레드 인덱싱
- 간단한 병렬 연산

- **병렬성**: 각 스레드가 독립적으로 실행됩니다
- **스레드 인덱싱**: `i = thread_idx.x` 위치의 요소에 접근합니다
- **메모리 접근**: `a[i]`에서 읽고 `output[i]`에 씁니다
- **데이터 독립성**: 각 출력은 해당 입력에만 의존합니다

## 작성할 코드

```mojo
{{#include ../../../../../problems/p01/p01.mojo:add_10}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p01/p01.mojo" class="filename">전체 코드 보기: problems/p01/p01.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. `thread_idx.x`를 `i`에 저장합니다
2. `a[i]`에 10을 더합니다
3. 결과를 `output[i]`에 저장합니다

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
pixi run p01
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p01
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p01
```

  </div>
  <div class="tab-content">

```bash
uv run poe p01
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
{{#include ../../../../../solutions/p01/p01.mojo:add_10_solution}}
```

<div class="solution-explanation">

이 솔루션은:

- `i = thread_idx.x`로 스레드 인덱스를 가져옵니다
- 입력값에 10을 더합니다: `output[i] = a[i] + 10.0`

</div>
</details>
