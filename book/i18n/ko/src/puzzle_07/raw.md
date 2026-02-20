<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

## 개요

행렬 `a`의 각 위치에 10을 더해 `output`에 저장하는 kernel을 구현해 보세요.

**참고:** _블록당 스레드 수가 `a`의 행과 열 크기보다 모두 작습니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- 2D 블록과 스레드 배치 다루기
- 블록 크기보다 큰 행렬 데이터 처리하기
- 2D 인덱스와 선형 메모리 접근 간 변환하기

핵심은 하나의 블록보다 큰 2D 행렬을 처리할 때 여러 블록의 스레드들이 어떻게 함께 작동하는지 이해하는 것입니다.

## 구성

- **행렬 크기**: \\(5 \times 5\\) 원소
- **2D 블록**: 각 블록이 \\(3 \times 3\\) 영역 처리
- **그리드 레이아웃**: \\(2 \times 2\\) 그리드에 블록 배치
- **총 스레드 수**: \\(25\\)개 원소에 대해 \\(36\\)개
- **메모리 패턴**: 2D 데이터를 row-major로 저장
- **커버리지**: 모든 행렬 원소가 빠짐없이 처리되도록 보장

## 완성할 코드

```mojo
{{#include ../../../../../problems/p07/p07.mojo:add_10_blocks_2d}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p07/p07.mojo" class="filename">전체 코드 보기: problems/p07/p07.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 전역 인덱스 계산: `row = block_dim.y * block_idx.y + thread_idx.y`, `col = block_dim.x * block_idx.x + thread_idx.x`
2. 가드 추가: `if row < size and col < size`
3. 가드 내부: row-major 방식으로 10을 더하는 방법을 생각해 보세요!

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
pixi run p07
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p07
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p07
```

  </div>
  <div class="tab-content">

```bash
uv run poe p07
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력이 다음과 같이 나타납니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, ... , 0.0])
expected: HostBuffer([10.0, 11.0, 12.0, ... , 34.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p07/p07.mojo:add_10_blocks_2d_solution}}
```

<div class="solution-explanation">

raw 메모리로 2D 블록 기반 처리를 구현할 때의 핵심 개념을 보여주는 솔루션입니다:

1. **2D 스레드 인덱싱**
   - 전역 row: `block_dim.y * block_idx.y + thread_idx.y`
   - 전역 col: `block_dim.x * block_idx.x + thread_idx.x`
   - 스레드 그리드를 행렬 원소에 매핑:

     ```txt
     3×3 블록으로 구성된 5×5 행렬:

     Block (0,0)         Block (1,0)
     [(0,0) (0,1) (0,2)] [(0,3) (0,4)    *  ]
     [(1,0) (1,1) (1,2)] [(1,3) (1,4)    *  ]
     [(2,0) (2,1) (2,2)] [(2,3) (2,4)    *  ]

     Block (0,1)         Block (1,1)
     [(3,0) (3,1) (3,2)] [(3,3) (3,4)    *  ]
     [(4,0) (4,1) (4,2)] [(4,3) (4,4)    *  ]
     [  *     *     *  ] [  *     *      *  ]
     ```

     (* = 스레드는 존재하지만 행렬 경계 밖)

2. **메모리 레이아웃**
   - Row-major 선형 메모리: `index = row * size + col`
   - 5×5 행렬 예시:

     ```txt
     2D 인덱스:       선형 메모리:
     (2,1) -> 11   [00 01 02 03 04]
                   [05 06 07 08 09]
                   [10 11 12 13 14]
                   [15 16 17 18 19]
                   [20 21 22 23 24]
     ```

3. **경계 검사**
   - 가드 `row < size and col < size`가 처리하는 경우:
     - 부분 블록에서 남는 스레드
     - 행렬 경계의 엣지 케이스
     - 3×3 스레드 블록의 2×2 그리드 = 25개 원소에 36개 스레드

4. **블록 조율**
   - 각 3×3 블록이 5×5 행렬의 일부분을 담당
   - 2×2 블록 그리드로 전체를 빠짐없이 커버
   - 경계 검사로 겹치는 스레드 처리
   - 블록들이 함께 효율적으로 병렬 처리

이 패턴은 블록 크기보다 큰 2D 데이터를 다룰 때 효율적인 메모리 접근과 스레드 조율을 어떻게 유지하는지 보여줍니다.
</div>
</details>
