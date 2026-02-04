# Puzzle 2: Zip

{{ youtube SlpgR685oGA breakpoint-lg }}

## 개요

벡터 `a`와 벡터 `b`의 각 위치를 더해 `output`에 저장하는 Kernel을 구현해 보세요.

**참고:** _각 위치마다 스레드 1개가 배정됩니다._

{{ youtube SlpgR685oGA breakpoint-sm }}

<img src="/puzzle_02/media/02.png" alt="Zip" class="light-mode-img">
<img src="/puzzle_02/media/02d.png" alt="Zip" class="dark-mode-img">

## 핵심 개념

이 퍼즐에서 배우는 내용:

- 여러 입력 배열의 병렬 처리
- 여러 입력에 대한 요소별 연산
- 배열 간 스레드-데이터 매핑
- 여러 배열의 메모리 접근 패턴

각 스레드 \\(i\\)에 대해: \\[\Large output[i] = a[i] + b[i]\\]

### 메모리 접근 패턴

```txt
Thread 0:  a[0] + b[0] → output[0]
Thread 1:  a[1] + b[1] → output[1]
Thread 2:  a[2] + b[2] → output[2]
...
```

💡 **참고**: 이제 Kernel에서 세 개의 배열(`a`, `b`, `output`)을 다루고 있습니다. 연산이 복잡해질수록 여러 배열에 대한 접근을 관리하기가 점점 어려워집니다.

## 작성할 코드

```mojo
{{#include ../../../../../problems/p02/p02.mojo:add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p02/p02.mojo" class="filename">전체 코드 보기: problems/p02/p02.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. `thread_idx.x`를 `i`에 저장합니다
2. `a[i]`와 `b[i]`를 더합니다
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
pixi run p02
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p02
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p02
```

  </div>
  <div class="tab-content">

```bash
uv run poe p02
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력이 다음과 같이 나타납니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 2.0, 4.0, 6.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p02/p02.mojo:add_solution}}
```

<div class="solution-explanation">

이 솔루션은:

- `i = thread_idx.x`로 스레드 인덱스를 가져옵니다
- 두 배열의 값을 더합니다: `output[i] = a[i] + b[i]`

</div>
</details>

### 앞으로 다룰 내용

직접 인덱싱은 간단한 요소별 연산에서 잘 작동하지만, 다음 상황을 생각해 보세요:

- 배열의 레이아웃이 서로 다르다면?
- 한 배열을 다른 배열에 확산(broadcast)해야 한다면?
- 여러 배열에서 병합(coalesced) 접근을 어떻게 보장할 수 있을까?

이러한 질문들은 [Puzzle 4에서 LayoutTensor를 소개](../puzzle_04/)할 때 다룹니다.
