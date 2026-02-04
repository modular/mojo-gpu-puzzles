# Puzzle 3: Guards

{{ youtube YFKutZbRYSM breakpoint-lg }}

## 개요

벡터 `a`의 각 위치에 10을 더해 `output`에 저장하는 Kernel을 구현해 보세요.

**참고**: _스레드 수가 데이터 개수보다 많아서, 일부 스레드는 처리할 데이터가 없습니다. 이런 스레드가 범위를 벗어난 메모리에 접근하지 않도록 방지해야 합니다._

{{ youtube YFKutZbRYSM breakpoint-sm }}

<img src="/puzzle_03/media/03.png" alt="Guard" class="light-mode-img">
<img src="/puzzle_03/media/03d.png" alt="Guard" class="dark-mode-img">

## 핵심 개념

이 퍼즐에서 다루는 내용:

- 스레드 수와 데이터 크기 불일치 처리
- 범위를 벗어난 메모리 접근 방지
- GPU Kernel에서 조건부 실행 사용
- 안전한 메모리 접근 패턴

### 수학적 표현

각 스레드 \\(i\\)에 대해:
\\[\Large \text{if}\\ i < \text{size}: output[i] = a[i] + 10\\]

### 메모리 안전 패턴

```txt
Thread 0 (i=0):  if 0 < size:  output[0] = a[0] + 10  ✓ Valid
Thread 1 (i=1):  if 1 < size:  output[1] = a[1] + 10  ✓ Valid
Thread 2 (i=2):  if 2 < size:  output[2] = a[2] + 10  ✓ Valid
Thread 3 (i=3):  if 3 < size:  output[3] = a[3] + 10  ✓ Valid
Thread 4 (i=4):  if 4 < size:  ❌ Skip (out of bounds)
Thread 5 (i=5):  if 5 < size:  ❌ Skip (out of bounds)
```

💡 **참고**: 다음 상황에서 경계(boundary) 검사는 점점 복잡해집니다:

- 다차원 배열
- 다양한 배열 형태
- 복잡한 접근 패턴

## 작성할 코드

```mojo
{{#include ../../../../../problems/p03/p03.mojo:add_10_guard}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p03/p03.mojo" class="filename">전체 코드 보기: problems/p03/p03.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. `thread_idx.x`를 `i`에 저장합니다
2. 가드 추가: `if i < size`
3. 가드 내부: `output[i] = a[i] + 10.0`

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
pixi run p03
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p03
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p03
```

  </div>
  <div class="tab-content">

```bash
uv run poe p03
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
{{#include ../../../../../solutions/p03/p03.mojo:add_10_guard_solution}}
```

<div class="solution-explanation">

이 솔루션은:

- `i = thread_idx.x`로 스레드 인덱스를 가져옵니다
- `if i < size`로 범위를 벗어난 접근을 방지합니다
- 가드 내부: 입력값에 10을 더합니다

> 경계 검사 없이도 테스트가 통과되는 이유가 궁금할 수 있습니다!
> 테스트 통과가 코드의 안전성이나 미정의 동작(Undefined Behavior) 부재를 보장하지는 않는다는 점을 항상 기억하세요.
> [Puzzle 10](../puzzle_10/puzzle_10.md)에서 이런 경우를 살펴보고, 안전성 버그를 잡는 도구를 사용해 봅니다.

</div>
</details>

### 앞으로 다룰 내용

간단한 경계 검사는 여기서 잘 작동하지만, 다음 상황을 생각해 보세요:

- 2D/3D 배열의 경계는 어떻게 처리할까?
- 다양한 형태를 효율적으로 처리하려면?
- 패딩이나 가장자리 처리가 필요하다면?

복잡도가 증가하는 예시:

```mojo
# 현재: 1D 경계 검사
if i < size: ...

# 곧 다룰 내용: 2D 경계 검사
if i < height and j < width: ...

# 이후: 패딩이 있는 3D
if i < height and j < width and k < depth and
   i >= padding and j >= padding: ...
```

이런 경계 처리 패턴은 [Puzzle 4에서 LayoutTensor를 배우면](../puzzle_04/) 훨씬 깔끔해집니다. LayoutTensor는 형태 관리 기능을 기본으로 제공합니다.
