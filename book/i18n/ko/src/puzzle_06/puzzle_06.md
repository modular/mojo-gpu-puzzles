<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

# Puzzle 6: Blocks

## 개요

벡터 `a`의 각 위치에 10을 더해 `output`에 저장하는 Kernel을 구현해 보세요.

**참고:** _블록당 스레드 수가 `a`의 크기보다 작습니다._

<img src="/puzzle_06/media/06.png" alt="블록 시각화" class="light-mode-img">
<img src="/puzzle_06/media/06d.png" alt="블록 시각화" class="dark-mode-img">

## 핵심 개념

이 퍼즐에서 다루는 내용:

- 스레드 블록 크기보다 큰 데이터 처리
- 여러 블록의 스레드 조율
- 전역 스레드 위치 계산

여기서 핵심은 여러 스레드 블록이 협력하여 단일 블록 용량보다 큰 데이터를 처리하면서도, 요소와 스레드 간 올바른 매핑을 유지하는 원리를 이해하는 것입니다.

## 완성할 코드

```mojo
{{#include ../../../../../problems/p06/p06.mojo:add_10_blocks}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p06/p06.mojo" class="filename">전체 코드 보기: problems/p06/p06.mojo</a>

> 참고: 이 퍼즐의 `LayoutTensor` 버전은 거의 동일하므로 독자에게 맡깁니다.

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 전역 인덱스 계산: `i = block_dim.x * block_idx.x + thread_idx.x`
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
pixi run p06
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p06
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p06
```

  </div>
  <div class="tab-content">

```bash
uv run poe p06
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력이 다음과 같이 나타납니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p06/p06.mojo:add_10_blocks_solution}}
```

<div class="solution-explanation">

이 솔루션은 블록 기반 GPU 처리의 핵심 개념을 다룹니다:

1. **전역 스레드 인덱싱**
   - 블록 인덱스와 스레드 인덱스를 결합: `block_dim.x * block_idx.x + thread_idx.x`
   - 각 스레드를 고유한 전역 위치에 매핑
   - 블록당 3개 스레드 예시:

     ```txt
     Block 0: [0 1 2]
     Block 1: [3 4 5]
     Block 2: [6 7 8]
     ```

2. **블록 조율**
   - 각 블록은 연속된 데이터 청크를 처리
   - 블록 크기(3) < 데이터 크기(9)이므로 여러 블록 필요
   - 블록 간 자동 작업 분배:

     ```txt
     Data:    [0 1 2 3 4 5 6 7 8]
     Block 0: [0 1 2]
     Block 1:       [3 4 5]
     Block 2:             [6 7 8]
     ```

3. **경계 검사**
   - 가드 조건 `i < size`로 경계 케이스 처리
   - 데이터 크기가 블록 크기로 나누어 떨어지지 않을 때 범위를 벗어난 접근 방지
   - 데이터 끝부분의 불완전한 블록 처리에 필수

4. **메모리 접근 패턴**
   - Coalesced 메모리 접근: 블록 내 스레드들이 연속된 메모리에 접근
   - 각 스레드가 하나의 요소 처리: `output[i] = a[i] + 10.0`
   - 블록 수준 병렬성으로 메모리 대역폭을 효율적으로 활용

이 패턴은 단일 스레드 블록 크기를 초과하는 대규모 데이터셋 처리의 기초가 됩니다.
</div>
</details>
