<!-- i18n-source-commit: 224fad345fe6e71377c89cdc596f8e28d58a1fa4 -->

# Tiled 버전

## 개요

LayoutTensor를 사용한 tiled 행렬 곱셈으로 정방 행렬 \\(A\\)와 \\(B\\)를 곱하는 kernel을 구현하세요. 큰 행렬을 작은 조각(타일)으로 나누어 처리하는 방식입니다.

## 핵심 개념

- LayoutTensor를 사용한 행렬 tiling으로 효율적인 연산
- 적절한 레이아웃을 사용한 멀티 블록 조율
- TensorBuilder를 통한 효율적인 공유 메모리 활용
- LayoutTensor 인덱싱을 사용한 타일 경계 처리

## 구성

- 행렬 크기: \\(\\text{SIZE\_TILED} = 9\\)
- 블록당 스레드 수: \\(\\text{TPB} \times \\text{TPB} = 3 \times 3\\)
- 그리드 차원: \\(3 \times 3\\) 블록
- 공유 메모리: 블록당 \\(\\text{TPB} \times \\text{TPB}\\) LayoutTensor 2개

레이아웃 구성:

- 입력 A: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- 입력 B: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- 출력: `Layout.row_major(SIZE_TILED, SIZE_TILED)`
- 공유 메모리: TensorBuilder를 사용한 `TPB × TPB` LayoutTensor 2개

## Tiling 전략

### 블록 구성

```txt
Grid Layout (3×3):           Thread Layout per Block (3×3):
[B00][B01][B02]               [T00 T01 T02]
[B10][B11][B12]               [T10 T11 T12]
[B20][B21][B22]               [T20 T21 T22]

각 블록은 LayoutTensor 인덱싱을 사용하여 하나의 타일을 처리
```

### 타일 처리 단계

1. 스레드 위치에 대한 전역 인덱스와 로컬 인덱스 계산
2. A와 B 타일을 위한 공유 메모리 할당
3. 각 타일에 대해:
   - 행렬 A와 B에서 타일 로드
   - 부분 곱 계산
   - 레지스터에 결과 누적
4. 최종 누적 결과 기록

### 메모리 접근 패턴

```txt
Matrix A (8×8)                 Matrix B (8×8)               Matrix C (8×8)
+---+---+---+                  +---+---+---+                +---+---+---+
|T00|T01|T02| ...              |T00|T01|T02| ...            |T00|T01|T02| ...
+---+---+---+                  +---+---+---+                +---+---+---+
|T10|T11|T12|                  |T10|T11|T12|                |T10|T11|T12|
+---+---+---+                  +---+---+---+                +---+---+---+
|T20|T21|T22|                  |T20|T21|T22|                |T20|T21|T22|
+---+---+---+                  +---+---+---+                +---+---+---+
  ...                            ...                          ...

Tile Processing (for computing C[T11]):
1. Load tiles from A and B:
   +---+      +---+
   |A11| ×    |B11|     For each phase k:
   +---+      +---+     C[T11] += A[row, k] × B[k, col]

2. Tile movement:
   Phase 1     Phase 2     Phase 3
   A: [T10]    A: [T11]    A: [T12]
   B: [T01]    B: [T11]    B: [T21]

3. Each thread (i,j) in tile computes:
   C[i,j] = Σ (A[i,k] × B[k,j]) for k in tile width

Synchronization required:
* After loading tiles to shared memory
* After computing each phase
```

## 작성할 코드

```mojo
{{#include ../../../../../problems/p16/p16.mojo:matmul_tiled}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p16/p16.mojo" class="filename">전체 파일 보기: problems/p16/p16.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 표준 인덱싱 규칙을 사용하세요: `local_row = thread_idx.y`, `local_col = thread_idx.x`
2. 전역 위치 계산:

   ```
   global_row = block_idx.y * TPB + local_row
   ```

   그리고

   ```
   global_col = block_idx.x * TPB + local_col
   ```

   **전역 인덱싱 공식 이해하기:**
   - 각 블록은 행렬의 `TPB × TPB` 타일을 처리합니다
   - `block_idx.y`는 현재 몇 번째 블록 행인지를 나타냅니다 (0, 1, 2...)
   - `block_idx.y * TPB`는 해당 블록 타일의 시작 행입니다
   - `local_row` (0~TPB-1)은 블록 내 스레드의 offset입니다
   - 둘을 더하면 전체 행렬에서의 실제 행 위치가 됩니다

       **TPB=3 예시:**

    ```txt
    Block Layout:        Global Matrix (9×9):
    [B00][B01][B02]      [0 1 2 | 3 4 5 | 6 7 8]
    [B10][B11][B12]  →   [9 A B | C D E | F G H]
    [B20][B21][B22]      [I J K | L M N | O P Q]
                         ——————————————————————
                         [R S T | U V W | X Y Z]
                         [a b c | d e f | g h i]
                         [j k l | m n o | p q r]
                         ——————————————————————
                         [s t u | v w x | y z α]
                         [β γ δ | ε ζ η | θ ι κ]
                         [λ μ ν | ξ ο π | ρ σ τ]

    Thread(1,2) in Block(1,0):
    - block_idx.y = 1, local_row = 1
    - global_row = 1 * 3 + 1 = 4
    - 이 스레드는 행렬의 4번째 행을 담당
    ```

3. 공유 메모리 할당 (`.fill(0)`으로 사전 초기화됨)
4. 9×9 완벽한 tiling이므로 경계 검사가 불필요!
5. 적절한 동기화와 함께 타일 간 결과를 누적

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
pixi run p16 --tiled
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p16 --tiled
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p16 --tiled
```

  </div>
  <div class="tab-content">

```bash
uv run poe p16 --tiled
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력은 다음과 같습니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([3672.0, 3744.0, 3816.0, 3888.0, 3960.0, 4032.0, 4104.0, 4176.0, 4248.0, 9504.0, 9738.0, 9972.0, 10206.0, 10440.0, 10674.0, 10908.0, 11142.0, 11376.0, 15336.0, 15732.0, 16128.0, 16524.0, 16920.0, 17316.0, 17712.0, 18108.0, 18504.0, 21168.0, 21726.0, 22284.0, 22842.0, 23400.0, 23958.0, 24516.0, 25074.0, 25632.0, 27000.0, 27720.0, 28440.0, 29160.0, 29880.0, 30600.0, 31320.0, 32040.0, 32760.0, 32832.0, 33714.0, 34596.0, 35478.0, 36360.0, 37242.0, 38124.0, 39006.0, 39888.0, 38664.0, 39708.0, 40752.0, 41796.0, 42840.0, 43884.0, 44928.0, 45972.0, 47016.0, 44496.0, 45702.0, 46908.0, 48114.0, 49320.0, 50526.0, 51732.0, 52938.0, 54144.0, 50328.0, 51696.0, 53064.0, 54432.0, 55800.0, 57168.0, 58536.0, 59904.0, 61272.0])
```

## 솔루션: 수동 tiling

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p16/p16.mojo:matmul_tiled_solution}}
```

<div class="solution-explanation">

tiled 행렬 곱셈 구현은 작은 타일 \\((3 \times 3)\\)을 사용하여 큰 행렬 \\((9 \times 9)\\)을 효율적으로 처리하는 방법을 보여줍니다. 동작 방식은 다음과 같습니다:

1. **공유 메모리 할당**

   ```txt
   Input matrices (9×9) - (3×3) tiling에 딱 맞는 크기:
   A = [0  1  2  3  4  5  6  7  8 ]    B = [0  2  4  6  8  10 12 14 16]
       [9  10 11 12 13 14 15 16 17]        [18 20 22 24 26 28 30 32 34]
       [18 19 20 21 22 23 24 25 26]        [36 38 40 42 44 46 48 50 52]
       [27 28 29 30 31 32 33 34 35]        [54 56 58 60 62 64 66 68 70]
       [36 37 38 39 40 41 42 43 44]        [72 74 76 78 80 82 84 86 88]
       [45 46 47 48 49 50 51 52 53]        [90 92 94 96 98 100 102 104 106]
       [54 55 56 57 58 59 60 61 62]        [108 110 112 114 116 118 120 122 124]
       [63 64 65 66 67 68 69 70 71]        [126 128 130 132 134 136 138 140 142]
       [72 73 74 75 76 77 78 79 80]        [144 146 148 150 152 154 156 158 160]

   블록당 공유 메모리 (3×3):
   a_shared[TPB, TPB]  b_shared[TPB, TPB]
   ```

2. **타일 처리 루프**

   ```txt
   타일 수 = 9 // 3 = 3개 (나머지 없이 딱 나눠짐!)

   각 타일에 대해:
   1. A와 B에서 타일 로드
   2. 부분 곱 계산
   3. 레지스터에 누적
   ```

3. **메모리 로딩 패턴**
   - \\((9 \times 9)\\)이 딱 나눠지므로 경계 검사가 기술적으로는 불필요하지만, 방어적 프로그래밍과 다른 행렬 크기에도 대응할 수 있도록 포함합니다.

     ```mojo
        # A 타일 로드 - 전역 행은 그대로, 열은 타일에 의해 결정
        if tiled_row < size and (tile * TPB + local_col) < size:
            a_shared[local_row, local_col] = a[
                tiled_row, tile * TPB + local_col
            ]

        # B 타일 로드 - 행은 타일에 의해 결정, 전역 열은 그대로
        if (tile * TPB + local_row) < size and tiled_col < size:
            b_shared[local_row, local_col] = b[
                tile * TPB + local_row, tiled_col
            ]
     ```

4. **타일 내 연산**

   ```mojo
   for k in range(min(TPB, size - tile * TPB)):
       acc += a_shared[local_row, k] * b_shared[k, local_col]
   ```

   - 공유 메모리 뱅크 충돌 회피:

     ```txt
     Bank Conflict Free (Good):        Bank Conflicts (Bad):
     Thread0: a_shared[0,k] b_shared[k,0]  Thread0: a_shared[k,0] b_shared[0,k]
     Thread1: a_shared[0,k] b_shared[k,1]  Thread1: a_shared[k,0] b_shared[1,k]
     Thread2: a_shared[0,k] b_shared[k,2]  Thread2: a_shared[k,0] b_shared[2,k]
     ↓                                     ↓
     서로 다른 뱅크에 병렬 접근             b_shared가 column-major였다면
     (a_shared는 broadcast)               같은 뱅크에 직렬 접근
     ```

     **공유 메모리 뱅크 충돌 설명:**
     - **왼쪽 (Good)**: `b_shared[k,threadIdx.x]`는 서로 다른 뱅크에 접근하고, `a_shared[0,k]`는 모든 스레드에 broadcast됩니다
     - **오른쪽 (Bad)**: b_shared가 column-major였다면 스레드들이 동시에 같은 뱅크에 접근하게 됩니다
     - **핵심**: 이것은 글로벌 메모리 coalescing이 아닌 공유 메모리 접근 패턴에 관한 것입니다
     - **뱅크 구조**: 공유 메모리는 32개 뱅크로 구성되어 있으며, 여러 스레드가 동시에 같은 뱅크의 다른 주소에 접근할 때 충돌이 발생합니다

5. **동기화 지점**

   ```txt
   barrier() 호출 시점:
   1. 타일 로딩 후
   2. 타일 연산 후
   ```

주요 성능 특성:

- \\((3 \times 3)\\) 타일로 \\((9 \times 9)\\) 행렬 처리 (딱 맞는 크기!)
- 공유 메모리로 빠른 타일 접근
- 병합된 메모리 접근으로 글로벌 메모리 트랜잭션 최소화
- 뱅크 충돌을 피하도록 최적화된 공유 메모리 레이아웃과 접근 패턴

1. **결과 기록**:

   ```mojo
   if tiled_row < size and tiled_col < size:
      output[tiled_row, tiled_col] = acc
   ```

   - 다른 행렬 크기와 tiling 전략을 위한 방어적 경계 검사 포함
   - 출력 행렬에 직접 대입
   - 모든 스레드가 유효한 결과를 기록

### 주요 최적화

1. **레이아웃 최적화**:
   - 모든 tensor에 row-major 레이아웃
   - 효율적인 2D 인덱싱

2. **메모리 접근**:
   - 병합된 글로벌 메모리 로드
   - 효율적인 공유 메모리 활용

3. **연산**:
   - 레지스터 기반 누적, 즉 `var acc: output.element_type = 0`
   - `@parameter`를 통한 컴파일 타임 루프 전개

이 구현은 다음을 통해 높은 성능을 달성합니다:

- LayoutTensor를 활용한 효율적인 메모리 접근
- 최적의 tiling 전략
- 적절한 스레드 동기화
- 세심한 경계 처리

</div>
</details>

## 솔루션: 관용적 LayoutTensor tiling

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p16/p16.mojo:matmul_idiomatic_tiled_solution}}
```

<div class="solution-explanation">

관용적 tiled 행렬 곱셈은 Mojo의 LayoutTensor API와 비동기 메모리 연산을 활용하여 깔끔한 구현을 제공합니다.

**핵심 포인트: 이 구현은 두 행렬 모두 병합 로딩을 사용하여 표준 A × B 행렬 곱셈을 수행합니다.**

**이 구현이 하는 것:**

- **행렬 연산**: 표준 \\(A \times B\\) 곱셈 (\\(A \times B^T\\)가 아님)
- **로딩 패턴**: 두 행렬 모두 `Layout.row_major(1, TPB)`로 병합 접근
- **연산**: `acc += a_shared[local_row, k] * b_shared[k, local_col]`
- **데이터 레이아웃**: 로딩 시 전치 없음 - 두 행렬을 같은 방향으로 로드

**이 구현이 하지 않는 것:**

- \\(A \times B^T\\) 곱셈을 수행하지 않음
- 전치 로딩 패턴을 사용하지 않음
- 복사 과정에서 데이터를 전치하지 않음

\\((9 \times 9)\\) 행렬 크기에서는 완벽한 tiling이 이루어져 모든 경계 검사가 불필요합니다:

1. **LayoutTensor 타일 API**

   ```mojo
   out_tile = output.tile[TPB, TPB](block_idx.y, block_idx.x)
   a_tile = a.tile[TPB, TPB](block_idx.y, idx)
   b_tile = b.tile[TPB, TPB](idx, block_idx.x)
   ```

   수동 좌표 계산 없이 "(block_idx.y, block_idx.x) 위치의 타일을 가져온다"를 직접 표현합니다. 자세한 내용은 [문서](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/#tile)를 참고하세요.

2. **비동기 메모리 연산**

   ```mojo
   copy_dram_to_sram_async[
      thread_layout = load_a_layout,
      num_threads = NUM_THREADS,
      block_dim_count = BLOCK_DIM_COUNT
   ](a_shared,a_tile)
   copy_dram_to_sram_async[
      thread_layout = load_b_layout,
      num_threads = NUM_THREADS,
      block_dim_count = BLOCK_DIM_COUNT
   ](b_shared, b_tile)
   async_copy_wait_all()
   ```

   이 연산들은:
   - 레지스터를 우회하는 전용 복사 엔진을 사용하여 연산과 메모리 전송의 중첩을 가능하게 합니다 ([copy_dram_to_sram_async](https://docs.modular.com/mojo/kernels/layout/layout_tensor/copy_dram_to_sram_async/) 참고)
   - 최적의 메모리 접근 패턴을 위한 특화된 스레드 레이아웃을 사용합니다
   - 수동 메모리 초기화가 불필요합니다
   - **중요**:
      - 표준 GPU 로드는 이미 비동기적입니다. 이 함수들은 더 나은 리소스 활용과 레지스터 우회를 제공합니다
      - `copy_dram_to_sram_async`는 기본적으로 1D 스레드 블록(`block_dim.y == block_dim.z == 1`)을 가정하며, 별도 지정이 없으면 스레드 블록의 모든 스레드가 복사에 참여합니다. 다음을 지정하여 이 동작을 변경할 수 있습니다:
         - `block_dim_count`: 스레드 블록의 차원 수 (2D 스레드 블록 `THREADS_PER_BLOCK_TILED = (TPB, TPB)`의 경우 `2`)
         - `num_threads`: 스레드 블록의 스레드 수 (`TPB*TPB == 9`)

3. **최적화된 메모리 접근 레이아웃**

   ```mojo
   comptime load_a_layout = Layout.row_major(1, TPB)    # 병합 로딩
   comptime load_b_layout = Layout.row_major(1, TPB)    # 병합 로딩
   # 참고: 표준 A × B 곱셈에서 두 행렬 모두 같은 레이아웃을 사용
   ```

   **현재 구현의 메모리 접근 분석:**

   두 행렬 모두 글로벌 메모리에서 병합 로딩을 위해 `Layout.row_major(1, TPB)`를 사용합니다:
   - `load_a_layout`: 스레드들이 협력하여 행렬 A 행의 연속 원소를 로드
   - `load_b_layout`: 스레드들이 협력하여 행렬 B 행의 연속 원소를 로드
   - **핵심**: 스레드 레이아웃은 복사 시 스레드 간 협력 방식을 결정하며, 최종 데이터 레이아웃과는 별개입니다

   **실제 연산 패턴 (A × B임을 증명):**

   ```mojo
   # 현재 구현의 실제 연산
   acc += a_shared[local_row, k] * b_shared[k, local_col]

   # 이것은 C[i,j] = Σ(A[i,k] * B[k,j])에 해당
   # 즉, 표준 행렬 곱셈 A × B
   ```

   **두 행렬이 같은 병합 로딩 패턴을 사용하는 이유:**

   ```txt
   글로벌 메모리에서 타일 로딩:
   - Matrix A 타일: 스레드들이 A[block_row, k], A[block_row, k+1], A[block_row, k+2]... 로드 (연속)
   - Matrix B 타일: 스레드들이 B[k, block_col], B[k, block_col+1], B[k, block_col+2]... 로드 (연속)

   Layout.row_major(1, TPB)로 두 패턴 모두 병합
   ```

   **세 가지 별개의 메모리 고려사항:**
   1. **글로벌→공유 coalescing**: `Layout.row_major(1, TPB)`로 병합 글로벌 메모리 접근 보장
   2. **공유 메모리 연산**: `a_shared[local_row, k] * b_shared[k, local_col]`로 뱅크 충돌 회피
   3. **행렬 연산**: 연산 패턴이 A × B를 결정 (A × B^T가 아님)

4. **완벽한 tiling으로 경계 검사 불필요**

   ```mojo
   @parameter
   for idx in range(size // TPB):  # 나머지 없는 나눗셈: 9 // 3 = 3
   ```

   \\((9 \times 9)\\) 행렬과 \\((3 \times 3)\\) 타일에서는 모든 타일이 정확히 꽉 차기 때문에 경계 검사가 필요 없습니다!

5. **방어적 경계 검사를 포함한 깔끔한 타일 처리**

   ```mojo
   # 완벽한 tiling에서도 방어적 경계 검사 포함
   if tiled_row < size and tiled_col < size:
       out_tile[local_row, local_col] = acc
   ```

   \\((9 \times 9)\\)의 완벽한 tiling에서는 이 경계 검사가 기술적으로 불필요하지만, 방어적 프로그래밍과 다른 행렬 크기와의 일관성을 위해 포함합니다.

### 성능 고려사항

관용적 구현은 tiling의 성능 이점을 유지하면서 더 깔끔한 추상화를 제공합니다:

1. **메모리 지역성**: tiling을 통해 공간적, 시간적 지역성을 활용
2. **병합 접근**: 특화된 로드 레이아웃으로 병합 메모리 접근 패턴 보장
3. **연산-메모리 중첩**: 비동기 메모리 연산을 통한 중첩 가능
4. **공유 메모리 효율**: 불필요한 공유 메모리 초기화 없음
5. **레지스터 압력**: 최적의 연산 처리량을 위한 누적 레지스터 사용

이 구현은 고수준 추상화로도 성능 저하 없이 복잡한 GPU 알고리즘을 표현할 수 있음을 보여줍니다. 고수준의 표현력과 저수준의 성능 제어를 결합하는 Mojo의 철학을 잘 보여주는 예시입니다.

### 수동 tiling과의 주요 차이점

| 기능 | 수동 Tiling | 관용적 Tiling |
|---------|--------------|------------------|
| 메모리 접근 | 경계 검사가 있는 직접 인덱싱 | LayoutTensor 타일 API |
| 타일 로딩 | 원소별 명시적 복사 | 전용 복사 엔진의 벌크 전송 |
| 공유 메모리 | 수동 초기화 (방어적) | 복사 함수가 관리 |
| 코드 복잡도 | 명시적 인덱싱으로 다소 장황 | 고수준 API로 더 간결 |
| 경계 검사 | 로딩과 연산 중 다수의 검사 | 최종 기록 시 단일 방어적 검사 |
| 행렬 방향 | A와 B 모두 같은 방향 (표준 A × B) | A와 B 모두 같은 방향 (표준 A × B) |
| 성능 | 메모리 패턴의 명시적 제어 | 레지스터 우회를 포함한 최적화된 레이아웃 |

관용적 접근 방식은 단순히 더 깔끔할 뿐 아니라, 특화된 메모리 레이아웃과 비동기 연산 덕분에 성능도 더 좋을 수 있습니다.

### 참고: 전치 로딩은 언제 유용할까?

현재 구현은 전치 로딩을 사용하지 않습니다. 이 섹션은 레이아웃 시스템으로 할 수 있는 것을 보여주기 위한 교육적 내용입니다.

**현재 구현 요약:**

- 두 행렬 모두 `Layout.row_major(1, TPB)` 사용
- 표준 A × B 곱셈 수행
- 복사 중 데이터 전치 없음

**전치 로딩을 사용하는 교육적 시나리오:**

이 퍼즐은 두 행렬 모두 표준 병합 로딩을 사용하지만, 레이아웃 시스템의 유연성은 다른 시나리오에서 강력한 최적화를 가능하게 합니다:

```mojo
# 예시: A × B를 계산하기 위해 사전 전치된 행렬 B^T를 로드
# (현재 구현에서는 이렇게 하지 않음)
comptime load_b_layout = Layout.row_major(TPB, 1)   # B^T를 병합 접근으로 로드
comptime store_b_layout = Layout.row_major(1, TPB)  # 공유 메모리에 B로 저장
copy_dram_to_sram_async[src_thread_layout=load_b_layout, dst_thread_layout=store_b_layout](b_shared, b_tile)
```

**전치 로딩의 활용 사례 (이 퍼즐에서는 사용하지 않음):**

1. **이미 전치된 입력 행렬**: \\(B\\)가 글로벌 메모리에 전치 상태로 저장되어 있는 경우
2. **다른 알고리즘**: \\(A^T \times B\\), \\(A \times B^T\\), 또는 \\(A^T \times B^T\\) 계산
3. **메모리 레이아웃 변환**: row-major와 column-major 레이아웃 간 변환
4. **별도 전치 연산 없이 로드**: 필요한 방향으로 데이터를 직접 로드

**핵심 구분:**

- **현재 구현**: 두 행렬 모두 표준 \\(A \times B\\) 곱셈에 `Layout.row_major(1, TPB)` 사용
- **전치 로딩 예시**: 이미 전치된 데이터나 다른 행렬 연산을 처리할 때 다른 레이아웃 사용

이것은 Mojo의 철학을 보여줍니다: 일반적인 경우에 고수준 추상화를 유지하면서도, 필요할 때 저수준 제어를 제공합니다.

---

## 요약: 핵심 정리

**관용적 tiled 구현이 실제로 하는 것:**

1. **행렬 연산**: 표준 A × B 곱셈
2. **메모리 로딩**: 두 행렬 모두 `Layout.row_major(1, TPB)`로 병합 접근
3. **연산 패턴**: `acc += a_shared[local_row, k] * b_shared[k, local_col]`
4. **데이터 레이아웃**: 로딩 시 전치 없음

**이것이 최적인 이유:**

- **병합 글로벌 메모리 접근**: `Layout.row_major(1, TPB)`로 효율적인 로딩 보장
- **뱅크 충돌 회피**: 공유 메모리 접근 패턴이 충돌을 방지
- **표준 알고리즘**: 가장 일반적인 행렬 곱셈 패턴을 구현

</div>
</details>
