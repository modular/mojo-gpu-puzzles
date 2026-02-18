<!-- i18n-source-commit: 224fad345fe6e71377c89cdc596f8e28d58a1fa4 -->

# 글로벌 메모리를 사용한 기본 버전

## 개요

정방 행렬 \\(A\\)와 \\(B\\)를 곱하여 결과를 \\(\text{output}\\)에 저장하는 kernel을 구현하세요.
각 스레드가 출력 행렬의 원소 하나를 계산하는 가장 기본적인 구현입니다.

## 핵심 개념

이 퍼즐에서 다루는 내용:

- 행렬 연산을 위한 2D 스레드 구성
- 글로벌 메모리 접근 패턴
- Row-major 레이아웃에서의 행렬 인덱싱
- 스레드와 출력 원소 간 매핑

핵심은 2D 스레드 인덱스를 행렬 원소에 매핑하고, 내적을 병렬로 계산하는 방법을 이해하는 것입니다.

## 구성

- 행렬 크기: \\(\\text{SIZE} \\times \\text{SIZE} = 2 \\times 2\\)
- 블록당 스레드 수: \\(\\text{TPB} \\times \\text{TPB} = 3 \\times 3\\)
- 그리드 차원: \\(1 \\times 1\\)

레이아웃 구성:

- 입력 A: `Layout.row_major(SIZE, SIZE)`
- 입력 B: `Layout.row_major(SIZE, SIZE)`
- 출력: `Layout.row_major(SIZE, SIZE)`

## 완성할 코드

```mojo
{{#include ../../../../../problems/p16/p16.mojo:naive_matmul}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p16/p16.mojo" class="filename">전체 파일 보기: problems/p16/p16.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 스레드 인덱스로 `row`와 `col` 계산
2. 인덱스가 `size` 범위 안에 있는지 확인
3. 로컬 변수에 곱의 합 누적
4. 최종 합을 올바른 출력 위치에 기록

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
pixi run p16 --naive
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p16 --naive
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p16 --naive
```

  </div>
  <div class="tab-content">

```bash
uv run poe p16 --naive
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력은 다음과 같습니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([4.0, 6.0, 12.0, 22.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p16/p16.mojo:naive_matmul_solution}}
```

<div class="solution-explanation">

LayoutTensor를 활용한 기본 행렬 곱셈은 다음과 같은 접근 방식을 따릅니다:

### 행렬 레이아웃 (2×2 예시)

```txt
Matrix A:          Matrix B:                   Output C:
[a[0,0] a[0,1]]    [b[0,0] b[0,1]]             [c[0,0] c[0,1]]
[a[1,0] a[1,1]]    [b[1,0] b[1,1]]             [c[1,0] c[1,1]]
```

### 구현 상세

1. **스레드 매핑**:

   ```mojo
   row = block_dim.y * block_idx.y + thread_idx.y
   col = block_dim.x * block_idx.x + thread_idx.x
   ```

2. **메모리 접근 패턴**:
   - 직접 2D 인덱싱: `a[row, k]`
   - 전치 접근: `b[k, col]`
   - 출력 기록: `output[row, col]`

3. **연산 흐름**:

   ```mojo
   # var로 가변 누적 변수를 선언하고 tensor의 원소 타입을 사용
   var acc: output.element_type = 0

   # @parameter로 컴파일 타임 루프 전개
   @parameter
   for k in range(size):
       acc += a[row, k] * b[k, col]
   ```

### 주요 언어 기능

1. **변수 선언**:
   - `var acc: output.element_type = 0`에서 `var`로 가변 변수를 선언하고, `output.element_type`으로 출력 tensor와 동일한 타입을 지정합니다
   - 누적 연산 전에 0으로 초기화

2. **루프 최적화**:
   - [`@parameter`](https://docs.modular.com/mojo/manual/decorators/parameter/#parametric-for-statement) 데코레이터로 컴파일 타임에 루프 전개
   - 크기가 작고 미리 알려진 행렬에서 성능 향상
   - 더 나은 명령어 스케줄링 가능

### 성능 특성

1. **메모리 접근**:
   - 각 스레드가 `2 x SIZE`회 글로벌 메모리를 읽음
   - 스레드당 글로벌 메모리 쓰기 1회
   - 스레드 간 데이터 재사용 없음

2. **연산 효율**:
   - 단순한 구현이지만 성능은 최적이 아님
   - 글로벌 메모리를 중복으로 많이 읽음
   - 빠른 공유 메모리를 활용하지 않음

3. **한계**:
   - 글로벌 메모리 대역폭을 많이 소모
   - 낮은 데이터 지역성
   - 큰 행렬로 갈수록 확장성 부족

이 기본 구현은 GPU 행렬 곱셈을 이해하기 위한 기준점으로, 메모리 접근 패턴을 최적화해야 하는 이유를 보여줍니다.
</div>
</details>
