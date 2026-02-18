<!-- i18n-source-commit: 7a8e8be1635ae0615b896ce069f6c241d562343a -->

# Puzzle 33: 텐서 코어 연산

## 소개

GPU 행렬 곱셈 최적화의 최전선에 오신 것을 환영합니다! 이 퍼즐에서는 mixed precision 행렬 연산을 전례 없는 속도로 가속하기 위해 설계된 전용 하드웨어 유닛인 **텐서 코어**를 탐구합니다.

지금까지 배운 모든 것, 특히 [Puzzle 16의 관용적 tiled 행렬 곱셈](../puzzle_16/puzzle_16.md)을 기반으로, 최신 GPU가 행렬 연산을 극적으로 빠르게 만드는 전용 실리콘을 어떻게 제공하는지 살펴보겠습니다.

## 텐서 코어란?

텐서 코어(AMD 하드웨어에서는 Matrix Core라고도 함)는 단일 명령어로 mixed precision 행렬-행렬 연산을 수행할 수 있는 전용 프로세싱 유닛입니다. 이 유닛은 최신 GPU 아키텍처에서 사용할 수 있습니다:

- **NVIDIA**: Tensor Cores (Volta, Turing, Ampere, Hopper)
- **AMD**: Matrix Cores (CDNA/CDNA2/CDNA3 아키텍처)

GPU에 직접 내장된 하드웨어 가속 GEMM(_역주: General Matrix Multiply, 범용 행렬 곱셈_) 엔진이라고 생각하면 됩니다.

### 핵심 특징

- **Warp 수준 연산**: 각 명령어가 전체 Warp의 데이터를 대상으로 동작합니다 (NVIDIA에서 32개 스레드, AMD에서 32 또는 64개)
- **고정 타일 크기**: 연산이 특정 행렬 프래그먼트 크기에서 동작합니다 (예: FP32의 경우 16×8×8)
- **Mixed precision**: 최적의 성능을 위해 입력과 출력의 정밀도를 혼합할 수 있습니다
- **대규모 처리량**: 행렬 연산에서 일반 컴퓨트 코어 대비 10~100배 속도 향상을 달성할 수 있습니다

## Tiled에서 텐서 코어로

기본 행렬 곱셈에서 텐서 코어까지의 여정을 돌아보겠습니다:

1. **Puzzle 16**: 공유 메모리를 활용한 관용적 tiled 행렬 곱셈을 배웠습니다
2. **공유 메모리 최적화**: 효율적인 메모리 전송을 위해 `copy_dram_to_sram_async`를 사용했습니다
3. **스레드 협력**: barrier와 비동기 연산으로 Warp를 조정했습니다
4. **지금**: 핵심 연산을 가속하기 위해 전용 하드웨어(텐서 코어)를 사용할 것입니다

## 텐서 코어 프로그래밍 모델

텐서 코어는 기존과 다른 프로그래밍 패러다임을 제공합니다:

### 기존 컴퓨트 코어 방식

```mojo
# Each thread computes one element
acc += a_shared[local_row, k] * b_shared[k, local_col]
```

### 텐서 코어 방식

```mojo
# Entire warp cooperates on matrix fragments
a_reg = mma_op.load_a(A_mma_tile)           # Load 16×8 fragment
b_reg = mma_op.load_b(B_mma_tile)           # Load 8×8 fragment
c_reg = mma_op.load_c(C_mma_tile)           # Load 16×8 accumulator
d_reg = mma_op.mma_op(a_reg, b_reg, c_reg)  # D = A×B + C
mma_op.store_d(C_mma_tile, d_reg)           # Store result
```

## Mojo의 텐서 코어 API

Mojo는 [`TensorCore`](https://docs.modular.com/mojo/kernels/layout/tensor_core/TensorCore/) 타입을 통해 텐서 코어에 대한 깔끔한 인터페이스를 제공합니다:

```mojo
from layout.tensor_core import TensorCore

# Create a Tensor Core operator for specific tile sizes
mma_op = TensorCore[A.dtype, C.dtype, Index(MMA_M, MMA_N, MMA_K)]()

# Core operations:
# - load_a(): Load matrix A fragment from shared memory
# - load_b(): Load matrix B fragment from shared memory
# - load_c(): Load matrix C fragment (accumulator)
# - mma_op(): Perform D = A×B + C operation
# - store_d(): Store result fragment to memory
```

**고급 기능:** TensorCore API는 양자화 연산, 메모리 접근 최적화를 위한 다양한 swizzle 패턴(_역주: 공유 메모리의 뱅크 충돌을 피하기 위해 데이터 주소를 비트 연산으로 재배치하는 기법_), mixed precision 연산도 지원합니다. 지원되는 모든 형태, 데이터 타입, 메서드에 대한 전체 문서는 [공식 TensorCore API 레퍼런스](https://docs.modular.com/mojo/kernels/layout/tensor_core/TensorCore/)를 참고하세요.

### 행렬 프래그먼트 크기

TensorCore API는 GPU 하드웨어에 따라 다양한 형태와 데이터 타입을 지원합니다:

**NVIDIA GPU:**

- **float32**: 16×8×8 또는 16×8×4
- **half-precision**: 16×8×16
- **float8**: 16×8×32

**AMD GPU:**

- **float32**: 16×16×4
- **half-precision**: 16×16×16 또는 32×32×8

**이 퍼즐에서는 FP32와 16×8×8 프래그먼트를 사용합니다:**

- **MMA_M = 16**: 행렬 A의 높이 (출력 높이와 동일)
- **MMA_N = 8**: 행렬 B의 너비 (출력 너비와 동일)
- **MMA_K = 8**: 내부 차원 (A의 너비 = B의 높이)

**MMA란?** MMA는 "Mixed-precision Matrix-Multiply-Accumulate"의 약자로, 텐서 코어가 수행하는 기본 연산입니다. 각 MMA 명령어는 `D = A × B + C`를 계산하며, 여기서 A, B, C, D는 행렬 프래그먼트입니다.

**프래그먼트 시각화:**

```txt
A fragment (16×8)  ×  B fragment (8×8)  +  C fragment (16×8)  =  D fragment (16×8)

    16 rows             8 rows               16 rows              16 rows
    8 cols              8 cols               8 cols               8 cols
      |                   |                    |                    |
   [A data]         ×   [B data]         +   [C data]         =  [D result]
```

즉, 각 텐서 코어 명령어는 A의 16×8 타일과 B의 8×8 타일을 곱한 뒤 기존 16×8 누산기에 더하여 16×8 출력 타일을 계산합니다.

## 텐서 코어를 위한 Warp 구성

**Warp란?** Warp는 lockstep으로 명령어를 함께 실행하는 스레드 그룹(NVIDIA에서 32개, AMD에서 32 또는 64개)입니다. 텐서 코어는 단일 행렬 연산에 Warp 내 모든 스레드가 협력해야 합니다.

**왜 Warp 수준일까?** 각 스레드가 독립적으로 동작하는 일반 연산과 달리, 텐서 코어는 전체 Warp가 함께 행렬 프래그먼트를 로드하고, MMA 연산을 수행하고, 결과를 저장해야 합니다.

텐서 코어가 Warp 수준에서 동작하므로, 스레드를 다르게 구성해야 합니다:

```mojo
# Calculate warp coordinates within the block
warp_id = thread_idx.x // WARP_SIZE
warps_in_n = BN // WN  # Number of warps along N dimension
warps_in_m = BM // WM  # Number of warps along M dimension
warp_y = warp_id // warps_in_n  # Warp's row
warp_x = warp_id % warps_in_n   # Warp's column

# Each warp handles a WM×WN tile of the output
C_warp_tile = C_block_tile.tile[WM, WN](warp_y, warp_x)
```

**Warp 구성 예시** (BM=128, BN=64, WM=32, WN=32인 경우):

```txt
Block (128×64) contains 8 warps arranged as:

    32 cols    32 cols
     |          |
[  Warp 0  ][  Warp 1  ]  ← 32 rows each
[  Warp 2  ][  Warp 3  ]  ← 32 rows each
[  Warp 4  ][  Warp 5  ]  ← 32 rows each
[  Warp 6  ][  Warp 7  ]  ← 32 rows each

Total: 4×2 = 8 warps, each handling 32×32 output region
```

## 텐서 코어와 메모리 계층 구조

텐서 코어는 메모리 최적화에 한 단계를 더 추가합니다:

1. **글로벌 메모리** → **공유 메모리**: `copy_dram_to_sram_async` 사용 (Puzzle 16에서 배운 것)
2. **공유 메모리** → **레지스터 프래그먼트**: `mma_op.load_a/load_b` 사용
3. **연산**: 레지스터 프래그먼트에서 `mma_op.mma_op` 사용
4. **레지스터 프래그먼트** → **글로벌 메모리**: `mma_op.store_d` 사용

## 도전 과제

`tensor_core_matrix_multiplication` 함수를 완성하는 것이 목표입니다. 스켈레톤 코드는 tiled 방식을 기반으로 하되 실제 텐서 코어 하드웨어 연산을 사용합니다.

### 핵심 요구사항

1. **실제 텐서 코어 API 사용**: 시뮬레이션이 아닌 실제 `mma_op.load_a()`, `mma_op.mma_op()` 등을 사용하세요
2. **정확성 유지**: 결과가 CPU 참조 구현과 일치해야 합니다
3. **올바른 Warp 조정**: 블록당 여러 Warp를 올바르게 처리합니다 (NVIDIA와 AMD 모두에서 동작)
4. **메모리 효율성**: Puzzle 16에서 배운 비동기 복사 패턴을 동일하게 사용합니다
5. **크로스 플랫폼 호환성**: tiling 파라미터가 `WARP_SIZE`의 배수인지 확인합니다

## 설정

- 행렬 크기: \\(\\text{SIZE} = 1024\\)
- 블록 tiling: \\(\\text{BM} = 128, \\text{BN} = 64, \\text{BK} = 32\\)
- Warp tiling: \\(\\text{WM} = 32, \\text{WN} = 32\\) (`WARP_SIZE`의 배수)
- MMA 프래그먼트: \\(16 \\times 8 \\times 8\\) (FP32)
- 블록당 스레드 수: \\(8 \\times \\text{WARP\_SIZE}\\) (블록당 8개 Warp)
- 그리드 차원: 블록 타일로 전체 행렬을 커버

레이아웃 설정:

- 입력 A: `Layout.row_major(SIZE, SIZE)`
- 입력 B: `Layout.row_major(SIZE, SIZE)`
- 출력 C: `Layout.row_major(SIZE, SIZE)`
- 공유 메모리: 비동기 복사 연산을 사용하는 블록 크기 타일

## 도전 과제

이 퍼즐에서는 Puzzle 16의 관용적 tiled 행렬 곱셈을 텐서 코어 구현으로 변환합니다. 단계별로 살펴보겠습니다:

### 1단계: Tiled 기본 구현 이해하기

퍼즐은 참조용으로 완성된 관용적 tiled 구현을 제공합니다:

```mojo
{{#include ../../../../../problems/p33/p33.mojo:matmul_idiomatic_tiled_solution}}
```

**이 기본 구현이 하는 일:**

- **정확성**: 이 구현은 완벽하게 동작하며 모든 테스트를 통과합니다
- **스레드 협력**: 효율적인 메모리 전송을 위해 `copy_dram_to_sram_async`를 사용합니다
- **공유 메모리**: barrier와 비동기 연산으로 스레드를 조정합니다
- **Tiled 연산**: 각 스레드가 공유 메모리 타일을 사용하여 하나의 출력 요소를 계산합니다

### 2단계: 텐서 코어 미션

위 방식을 전용 하드웨어 가속을 활용하도록 변환합니다:

- **기존:** 스레드 수준 연산 → **변환 후:** Warp 수준 행렬 프래그먼트
- **기존:** 표준 FP32 산술 → **변환 후:** 하드웨어 가속 GEMM 연산
- **기존:** 개별 요소 결과 → **변환 후:** 16×8 행렬 프래그먼트 결과

### 3단계: 설정 이해하기

텐서 코어 버전은 하드웨어에 최적화된 다른 tiling 파라미터를 사용합니다:

- **블록 tiling**: `BM=128, BN=64, BK=32` (더 나은 점유율을 위해 더 큰 블록)
- **Warp tiling**: `WM=32, WN=32` (각 Warp가 32×32 출력 영역을 담당)
- **MMA 프래그먼트**: `16×8×8` (하드웨어가 정의한 행렬 프래그먼트 크기)
- **블록당 Warp**: 8개 (BM×BN 블록 내에서 4×2로 배치)

**왜 이 특정 크기인가?**

- **BM=128, BN=64**: 텐서 코어를 더 잘 활용하기 위해 tiled 버전(32×32)보다 큽니다
- **WM=WN=32**: WARP_SIZE의 배수이며 2×4=8개의 MMA 프래그먼트를 포함합니다 (32÷16=2, 32÷8=4)
- **MMA 16×8×8**: 하드웨어에 의해 고정됨 - 텐서 코어가 물리적으로 계산하는 크기입니다
- **8 Warp**: BM÷WM × BN÷WN = 128÷32 × 64÷32 = 4×2 = 블록당 8개 Warp

**Warp가 MMA 프래그먼트에 매핑되는 방식:**

```txt
Each 32×32 warp tile contains multiple 16×8 MMA fragments:

    16 cols   16 cols
     |         |
[ MMA 0,0 ][ MMA 0,1 ]  ← 8 rows each (32÷8=4 fragments down)
[ MMA 1,0 ][ MMA 1,1 ]  ← 8 rows each
[ MMA 2,0 ][ MMA 2,1 ]  ← 8 rows each
[ MMA 3,0 ][ MMA 3,1 ]  ← 8 rows each

2 fragments across (32÷16=2) × 4 fragments down (32÷8=4) = 8 MMA operations per warp per K-tile
```

### 4단계: 완성할 코드

```mojo
{{#include ../../../../../problems/p33/p33.mojo:tensor_core_matrix_multiplication}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p33/p33.mojo" class="filename">전체 파일 보기: problems/p33/p33.mojo</a>

**할 일**: 세 겹의 중첩 루프 안에 있는 빈 부분(`# FILL IN (roughly 8 lines)`으로 표시됨)을 완성하세요.

**이해해야 할 것:**

- 스켈레톤이 모든 메모리 관리, Warp 구성, 동기화를 처리합니다
- 핵심 텐서 코어 연산만 구현하면 됩니다
- 루프는 MMA 프래그먼트를 순회합니다: `mma_k`, `mma_m`, `mma_n`
- 각 반복에서 하나의 16×8×8 행렬 프래그먼트를 처리합니다

**세 겹 중첩 루프 이해하기:**

```mojo
@parameter
for mma_k in range(BK // MMA_K):     # 32÷8 = 4 iterations (K dimension)
    @parameter
    for mma_m in range(WM // MMA_M): # 32÷16 = 2 iterations (M dimension)
        @parameter
        for mma_n in range(WN // MMA_N): # 32÷8 = 4 iterations (N dimension)
            # YOUR CODE HERE: Process one 16×8×8 MMA fragment
```

**각 루프가 하는 일:**

- `mma_k`: 현재 K-타일의 K-슬라이스를 순회합니다 (각 8개 요소의 4개 슬라이스)
- `mma_m`: Warp 출력의 M-슬라이스를 순회합니다 (각 16행의 2개 슬라이스)
- `mma_n`: Warp 출력의 N-슬라이스를 순회합니다 (각 8열의 4개 슬라이스)
- **합계**: 4×2×4 = K-타일당 Warp당 32개 MMA 연산

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

텐서 코어 워크플로우를 생각해 보세요. 필요한 단계는 다음과 같습니다:

1. **올바른 행렬 프래그먼트 추출하기**:
   - Warp 타일(`A_warp_tile`, `B_warp_tile`, `C_warp_accum`)에서 MMA 크기의 특정 프래그먼트를 추출합니다
   - 루프 인덱스(`mma_m`, `mma_k`, `mma_n`)를 사용하여 올바른 타일 좌표를 구합니다
   - 기억하세요: A는 [MMA_M, MMA_K], B는 [MMA_K, MMA_N], C는 [MMA_M, MMA_N]이 필요합니다

2. **프래그먼트를 텐서 코어 레지스터에 로드하기**:
   - `mma_op` 객체에는 각 행렬 타입을 로드하는 메서드가 있습니다
   - 각 로드 메서드는 타일을 받아서 레지스터 프래그먼트를 반환합니다
   - 생각해 보세요: `load_a()`, `load_b()`, `load_c()` - 각각 무엇을 받을까요?

3. **하드웨어 연산을 수행하고 결과 저장하기**:
   - MMA 연산을 수행하여 결과를 계산합니다
   - 결과를 누산기 타일에 저장합니다
   - 연산 패턴: result = A × B + C

**핵심 인사이트**: 128개의 개별 곱셈-덧셈 연산을 하나의 하드웨어 명령어로 대체하는 것입니다!

**디버깅 팁**: 차원 오류가 발생하면 타일 인덱싱을 다시 확인하세요 - `mma_m`, `mma_k`, `mma_n`의 순서가 올바른 프래그먼트를 가져오는 데 중요합니다.

</div>
</details>

## 코드 실행

풀이를 테스트하려면 터미널에서 다음 명령어를 실행하세요:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p33 --test
```

  </div>
  <div class="tab-content">

```bash
uv run poe p33 --test
```

  </div>
</div>

완성하면 다음과 같은 정확도 테스트 결과가 출력됩니다:

```txt
=== Running All Accuracy Tests ===
--- Test 1: Tensor Core vs CPU Reference ---
✅ TENSOR CORE ACCURACY TEST PASSED!
--- Test 2: Idiomatic Tiled vs CPU Reference ---
✅ IDIOMATIC TILED ACCURACY TEST PASSED!
ALL TESTS PASSED!
```

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p33/p33.mojo:tensor_core_matrix_multiplication_solution}}
```

<div class="solution-explanation">

이 풀이는 텐서 코어 프로그래밍 모델을 보여줍니다:

1. **Warp 구성**
   - `warp_id = thread_idx.x // WARP_SIZE`로 블록 내 Warp 좌표를 계산합니다
   - Warp를 출력 타일에 매핑합니다: 각 Warp가 `WM×WN` 영역을 담당합니다
   - 예상보다 적은 수의 Warp가 있는 블록을 처리하기 위해 `warp_is_active` 가드를 사용합니다

2. **메모리 계층 구조 최적화**
   - **글로벌 → 공유**: 효율적인 블록 수준 전송을 위해 `copy_dram_to_sram_async`를 사용합니다
   - **공유 → 레지스터**: Warp 수준 프래그먼트 로딩을 위해 `mma_op.load_a/load_b`를 사용합니다
   - **레지스터 연산**: 하드웨어 가속 행렬 연산을 위해 `mma_op.mma_op`를 사용합니다
   - **레지스터 → 글로벌**: 효율적인 결과 저장을 위해 `mma_op.store_d`를 사용합니다

3. **텐서 코어 연산**
   - `load_a(A_mma_tile)`: 16×8 행렬 A 프래그먼트를 레지스터에 로드합니다
   - `load_b(B_mma_tile)`: 8×8 행렬 B 프래그먼트를 레지스터에 로드합니다
   - `load_c(C_mma_tile)`: 16×8 누산기 프래그먼트를 로드합니다
   - `mma_op(a_reg, b_reg, c_reg)`: 전용 하드웨어를 사용하여 D = A×B + C를 계산합니다
   - `store_d(C_mma_tile, d_reg)`: 16×8 결과 프래그먼트를 저장합니다

4. **크로스 플랫폼 호환성**
   - 모든 tiling 파라미터가 `WARP_SIZE`의 배수입니다 (NVIDIA에서 32, AMD에서 64)
   - Mojo는 `TensorCore` 인터페이스를 통해 하드웨어 차이를 추상화합니다
   - 동일한 코드가 NVIDIA 텐서 코어와 AMD Matrix Core 모두에서 동작합니다

핵심 인사이트는 텐서 코어가 스레드 수준의 개별 요소가 아닌 Warp 수준의 전체 행렬 프래그먼트 단위로 동작한다는 것입니다. 이를 통해 대규모 병렬 처리와 전용 하드웨어 가속이 가능해집니다.

</div>
</details>

## 성능 분석: 이것으로 끝일까?

이제 텐서 코어가 관용적 tiled 방식 대비 약속된 성능 우위를 실제로 제공하는지 확인해 보겠습니다.

### 프로파일링용 빌드

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run mojo build problems/p33/p33.mojo -o problems/p33/p33_profiler
```

  </div>
  <div class="tab-content">

```bash
pixi run mojo build problems/p33/p33.mojo -o problems/p33/p33_profiler
```

  </div>
</div>

### NVIDIA Nsight Compute로 프로파일링 (NVIDIA 전용)

먼저 `ncu`에 접근하기 위해 CUDA 환경에 진입합니다:

```bash
# Enter CUDA environment
pixi shell -e nvidia

# Profile tensor core version
ncu --set full --metrics sm__cycles_elapsed.avg,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed_pipe_tensor_op_hmma.sum ./problems/p33p33_profiler --tensor-core

# Profile tiled version for comparison
ncu --set full --metrics sm__cycles_elapsed.avg,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./problems/p33p33_profiler --tiled
```

### 비교할 핵심 메트릭

**성능 메트릭:**

- **Duration**: 전체 kernel 실행 시간 (낮을수록 좋음)
- **SM Active %**: SM 활용률 (높을수록 좋음)
- **DRAM Throughput**: 메모리 대역폭 활용률 (메모리 바운드 여부를 보여줌)
- **Tensor Op Instructions**: 실제 텐서 코어 연산 횟수 (텐서 코어 버전에만 해당)

**일반적인 결과:**

**텐서 코어 버전 (더 느림):**

- **Duration**: ~13.9 ms (훨씬 느림!)
- **SM Active**: 83.7% (좋은 활용률)
- **DRAM Throughput**: 72.5% (메모리 바운드!)
- **Occupancy**: 26.3% (나쁨 - 레지스터에 의해 제한됨)
- **Tensor Op Instructions**: 1,048,576 (텐서 코어가 동작 중임을 확인)

**Tiled 버전 (더 빠름):**

- **Duration**: ~1.62 ms (8.6배 빠름!)
- **SM Active**: 98.0% (탁월한 활용률)
- **DRAM Throughput**: 1.7% (예상대로 연산 바운드)
- **Occupancy**: 66.7% (훨씬 나음)
- **L2 Hit Rate**: 96.9% vs 29.7% (훨씬 나은 캐시 지역성)

**왜 텐서 코어가 더 느릴까?**

- **메모리 병목**: 72% DRAM 사용량은 연산 바운드가 아닌 메모리 바운드임을 보여줍니다
- **낮은 점유율**: 26% vs 67% - 높은 레지스터 사용량(스레드당 68 vs 38)이 동시 Warp 수를 제한합니다
- **캐시 미스**: 29% L2 적중률 vs 97%는 낮은 메모리 지역성을 보여줍니다
- **공유 메모리 충돌**: 최적화되지 않은 접근 패턴으로 인한 뱅크 충돌
- **실행 설정**: 이 문제 크기에 대해 최적이 아닌 블록/Warp 구성

## 성능의 현실

프로파일링 결과에서 볼 수 있듯이, "전용 하드웨어"가 자동으로 빨라지는 것은 아닙니다! 텐서 코어 버전은 단순한 tiled 방식보다 상당히 느립니다(~8.6배). 이는 GPU 최적화에서 흔히 볼 수 있는 현실입니다 - 하드웨어의 원시 성능이 곧 더 나은 성능을 보장하지는 않습니다.

**핵심 인사이트:**

- **메모리 병목**: 72% DRAM 사용량은 텐서 코어가 연산 바운드가 아닌 메모리 바운드임을 보여줍니다
- **낮은 점유율**: 높은 레지스터 사용량으로 인해 26% vs 67%로 동시 Warp 수가 제한됩니다
- **캐시 미스**: 29% vs 97% L2 적중률은 낮은 메모리 지역성을 보여줍니다
- **리소스 낭비**: 공유 메모리 뱅크 충돌과 최적이 아닌 실행 설정

**교훈**: 성능 병목을 이해하고 체계적으로 최적화하는 것이 "최신의 가장 뛰어난" API를 사용하는 것보다 중요합니다. 하드웨어 기능은 세심한 튜닝이 필요한 도구이지, 마법의 은탄환이 아닙니다.

## 다음 단계

보람 있는 GPU 최적화 도전을 할 준비가 되셨나요? [🎯 성능 보너스 챌린지](../bonuses/part5.md)로 이동하여 메모리 바운드인 텐서 코어 구현을 단순한 tiled 버전을 실제로 이기는 구현으로 변환하는 방법을 배워보세요!
