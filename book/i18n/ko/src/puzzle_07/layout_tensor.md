<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

# LayoutTensor 버전

## 개요

2D LayoutTensor `a`의 각 위치에 10을 더해 2D LayoutTensor `output`에 저장하는 kernel을 구현해 보세요.

**참고:** _블록당 스레드 수가 `a`의 행과 열 크기보다 모두 작습니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- 여러 블록과 함께 `LayoutTensor` 사용하기
- 2D 블록 구성으로 큰 행렬 처리하기
- 블록 인덱싱과 `LayoutTensor` 접근 결합하기

핵심은 `LayoutTensor`가 2D 인덱싱을 단순화해 주지만, 큰 행렬에서는 여전히 블록 간 조율이 필요하다는 점입니다.

## 구성

- **행렬 크기**: \\(5 \times 5\\) 원소
- **레이아웃 처리**: `LayoutTensor`가 row-major 구성 관리
- **블록 조율**: 여러 블록으로 전체 행렬 커버
- **2D 인덱싱**: 경계 검사와 함께 자연스러운 \\((i,j)\\) 접근
- **총 스레드 수**: \\(25\\)개 원소에 대해 \\(36\\)개
- **스레드 매핑**: 각 스레드가 행렬 원소 하나씩 처리

## 작성할 코드

```mojo
{{#include ../../../../../problems/p07/p07_layout_tensor.mojo:add_10_blocks_2d_layout_tensor}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p07/p07_layout_tensor.mojo" class="filename">전체 코드 보기: problems/p07/p07_layout_tensor.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 전역 인덱스 계산: `row = block_dim.y * block_idx.y + thread_idx.y`, `col = block_dim.x * block_idx.x + thread_idx.x`
2. 가드 추가: `if row < size and col < size`
3. 가드 내부: 2D LayoutTensor에 10을 더하는 방법을 생각해 보세요

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
pixi run p07_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p07_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p07_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p07_layout_tensor
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
{{#include ../../../../../solutions/p07/p07_layout_tensor.mojo:add_10_blocks_2d_layout_tensor_solution}}
```

<div class="solution-explanation">

LayoutTensor가 2D 블록 기반 처리를 얼마나 간소화하는지 보여주는 솔루션입니다:

1. **2D 스레드 인덱싱**
   - 전역 row: `block_dim.y * block_idx.y + thread_idx.y`
   - 전역 col: `block_dim.x * block_idx.x + thread_idx.x`
   - 스레드 그리드를 텐서 원소에 매핑:

     ```txt
     3×3 블록으로 구성된 5×5 텐서:

     Block (0,0)         Block (1,0)
     [(0,0) (0,1) (0,2)] [(0,3) (0,4)    *  ]
     [(1,0) (1,1) (1,2)] [(1,3) (1,4)    *  ]
     [(2,0) (2,1) (2,2)] [(2,3) (2,4)    *  ]

     Block (0,1)         Block (1,1)
     [(3,0) (3,1) (3,2)] [(3,3) (3,4)    *  ]
     [(4,0) (4,1) (4,2)] [(4,3) (4,4)    *  ]
     [  *     *     *  ] [  *     *      *  ]
     ```

     (* = 스레드는 존재하지만 텐서 경계 밖)

2. **LayoutTensor의 장점**
   - 자연스러운 2D 인덱싱: 수동 offset 계산 대신 `tensor[row, col]` 사용
   - 자동 메모리 레이아웃 최적화
   - 접근 패턴 예시:

     ```txt
     Raw 메모리:          LayoutTensor:
     row * size + col    tensor[row, col]
     (2,1) -> 11        (2,1) -> 같은 원소
     ```

3. **경계 검사**
   - 가드 `row < size and col < size`가 처리하는 상황:
     - 부분 블록에서 범위를 벗어나는 스레드
     - 텐서 경계의 엣지 케이스
     - 메모리 레이아웃은 LayoutTensor가 자동으로 처리
     - 25개 원소를 36개 스레드로 처리 (3×3 블록의 2×2 그리드)

4. **블록 조율**
   - 각 3×3 블록이 5×5 텐서의 일부분을 담당
   - LayoutTensor가 처리하는 부분:
     - 메모리 레이아웃 최적화
     - 효율적인 접근 패턴
     - 블록 경계 간 조율
     - 캐시 친화적 데이터 접근

이 패턴은 LayoutTensor가 최적의 메모리 접근 패턴과 스레드 조율을 유지하면서도 2D 블록 처리를 얼마나 간소화하는지 보여줍니다.
</div>
</details>
