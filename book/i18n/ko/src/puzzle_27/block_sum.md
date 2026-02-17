<!-- i18n-source-commit: 43fce1182f8029e7edc50157aed0e6ebb8129d42 -->

# block.sum()의 핵심 - 블록 레벨 내적

[Puzzle 12](../puzzle_12/puzzle_12.md)에서 살펴본 내적을 블록 레벨 [sum](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum) 연산으로 구현합니다. 복잡한 공유 메모리 패턴을 간단한 함수 호출로 대체합니다. 블록 내 각 스레드가 하나의 요소를 처리하고 `block.sum()`으로 결과를 자동으로 합산하여, 블록 프로그래밍이 전체 스레드 블록에 걸친 GPU 동기화를 어떻게 변환하는지 보여줍니다.

**핵심 통찰:** _[block.sum()](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum) 연산은 블록 전체 실행을 활용하여 공유 메모리 + barrier + 트리 reduction을 블록 내 모든 스레드에 걸쳐 Warp 패턴을 사용하는 정교하게 최적화된 구현으로 대체합니다. LLVM 분석은 [기술 분석](#기술-분석-blocksum은-실제로-무엇으로-컴파일될까)을 참고하세요._

## 핵심 개념

이 퍼즐에서 배울 내용:

- `block.sum()`을 활용한 **블록 레벨 reduction**
- **블록 전체 동기화**와 스레드 조율
- 단일 블록 내 **크로스 Warp 통신**
- 복잡한 패턴에서 간단한 패턴으로의 **성능 변환**
- **스레드 0 결과 관리**와 조건부 쓰기

수학적 연산은 내적(dot product)입니다:
\\[\Large \text{output}[0] = \sum_{i=0}^{N-1} a[i] \times b[i]\\]

하지만 구현 과정에서 Mojo의 모든 블록 레벨 GPU 프로그래밍에 적용되는 기본 패턴을 배웁니다.

## 구성

- 벡터 크기: `SIZE = 128` 요소
- 데이터 타입: `DType.float32`
- 블록 구성: `(128, 1)` 블록당 스레드 수 (`TPB = 128`)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)
- 블록당 Warp 수: `128 / WARP_SIZE` (NVIDIA에서 4개, AMD에서 2개 또는 4개)

## 기존 방식의 복잡성 (Puzzle 12에서)

[Puzzle 12](../puzzle_12/layout_tensor.md)의 복잡한 방식을 떠올려 봅시다. 공유 메모리, barrier, 트리 reduction이 필요했습니다:

```mojo
{{#include ../../../../../solutions/p27/p27.mojo:traditional_dot_product_solution}}
```

**이 방식이 복잡한 이유:**

- **공유 메모리 할당**: 블록 내에서 수동으로 메모리를 관리
- **명시적 barrier**: 블록 내 모든 스레드를 동기화하기 위한 `barrier()` 호출
- **트리 reduction**: stride 기반 인덱싱을 사용하는 복잡한 루프 (64→32→16→8→4→2→1)
- **크로스 Warp 조율**: 여러 Warp 간 동기화가 필요
- **조건부 쓰기**: 스레드 0만 최종 결과를 기록

이 방식은 전체 블록(GPU에 따라 2개 또는 4개 Warp에 걸친 128 스레드)에서 동작하지만, 코드가 장황하고 오류가 발생하기 쉬우며 블록 레벨 GPU 동기화에 대한 깊은 이해가 필요합니다.

## Warp 레벨 개선 (Puzzle 24에서)

블록 레벨 연산으로 넘어가기 전에, [Puzzle 24](../puzzle_24/warp_sum.md)에서 `warp.sum()`을 사용하여 단일 Warp 내 reduction을 어떻게 단순화했는지 떠올려 봅시다:

```mojo
{{#include ../../../../../solutions/p24/p24.mojo:simple_warp_kernel_solution}}
```

**`warp.sum()`이 달성한 것:**

- **단일 Warp 범위**: 32 스레드(NVIDIA) 또는 32/64 스레드(AMD) 내에서 동작
- **하드웨어 shuffle**: 효율적인 `shfl.sync.bfly.b32` 명령 사용
- **공유 메모리 불필요**: 명시적 메모리 관리 없음
- **한 줄 reduction**: `total = warp_sum[warp_size=WARP_SIZE](val=partial_product)`

**그러나 한계가 있습니다:** `warp.sum()`은 단일 Warp 내에서만 동작합니다. 여러 Warp가 필요한 문제(예: 128 스레드 블록)에서는 여전히 Warp 간 조율을 위해 복잡한 공유 메모리 + barrier 방식이 필요합니다.

**기존 방식 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p27 --traditional-dot-product
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p27 --traditional-dot-product
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p27 --traditional-dot-product
```

  </div>
  <div class="tab-content">

```bash
uv run poe p27 --traditional-dot-product
```

  </div>
</div>

## 작성할 코드

### `block.sum()` 방식

복잡한 기존 방식을 `block.sum()`을 사용하는 간단한 블록 커널로 변환합니다:

```mojo
{{#include ../../../../../problems/p27/p27.mojo:block_sum_dot_product}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p27/p27.mojo" class="filename">전체 파일 보기: problems/p27/p27.mojo</a>

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p27 --block-sum-dot-product
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p27 --block-sum-dot-product
```

  </div>
  <div class="tab-content">

```bash
uv run poe p27 --block-sum-dot-product
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
SIZE: 128
TPB: 128
Expected result: 1381760.0
Block.sum result: 1381760.0
Block.sum() gives identical results!
Compare the code: 15+ lines of barriers → 1 line of block.sum()!
Just like warp.sum() but for the entire block
```

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **세 단계 패턴 이해하기**

모든 블록 reduction은 동일한 개념적 패턴을 따릅니다:

1. 각 스레드가 자신의 로컬 기여분을 계산
2. 모든 스레드가 블록 전체 reduction에 참여
3. 지정된 하나의 스레드가 최종 결과를 처리

### 2. **내적 수학 기억하기**

각 스레드는 벡터 `a`와 `b`에서 하나의 요소 쌍을 처리해야 합니다. 이들을 스레드 간에 합산할 수 있는 "부분 결과"로 합치는 연산은 무엇일까요?

### 3. **LayoutTensor 인덱싱 패턴**

`LayoutTensor` 요소에 접근할 때, 인덱싱이 SIMD 값을 반환한다는 점을 기억하세요. 산술 연산을 위해 스칼라 값을 추출해야 합니다.

### 4. **[block.sum()](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum) API 개념**

함수 시그니처를 살펴보세요 - 다음이 필요합니다:

- 블록 크기를 지정하는 템플릿 파라미터
- 결과 분배 방식을 위한 템플릿 파라미터 (`broadcast`)
- reduce할 값을 담은 런타임 파라미터

### 5. **스레드 조율 원칙**

- 어떤 스레드가 처리할 유효한 데이터를 가지고 있을까요? (힌트: 경계 검사)
- 어떤 스레드가 최종 결과를 기록해야 할까요? (힌트: 일관된 선택)
- 그 특정 스레드를 어떻게 식별할까요? (힌트: 스레드 인덱싱)

</div>
</details>

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p27/p27.mojo:block_sum_dot_product_solution}}
```

<div class="solution-explanation">

`block.sum()` 커널은 복잡한 블록 동기화에서 정교하게 최적화된 구현으로의 근본적인 변환을 보여줍니다:

**기존 방식에서 사라진 것들:**

- **15줄 이상 → 8줄**: 획기적인 코드 축소
- **공유 메모리 할당**: 메모리 관리 불필요
- **7회 이상의 barrier() 호출**: 명시적 동기화 제로
- **복잡한 트리 reduction**: 단일 함수 호출로 대체
- **Stride 기반 인덱싱**: 완전히 제거
- **크로스 Warp 조율**: 최적화된 구현이 자동으로 처리

**블록 전체 실행 모델:**

```
블록 스레드 (128 스레드, 4개 Warp):
Warp 0 (스레드 0-31):
  스레드 0: partial_product = a[0] * b[0] = 0.0
  스레드 1: partial_product = a[1] * b[1] = 2.0
  ...
  스레드 31: partial_product = a[31] * b[31] = 1922.0

Warp 1 (스레드 32-63):
  스레드 32: partial_product = a[32] * b[32] = 2048.0
  ...

Warp 2 (스레드 64-95):
  스레드 64: partial_product = a[64] * b[64] = 8192.0
  ...

Warp 3 (스레드 96-127):
  스레드 96: partial_product = a[96] * b[96] = 18432.0
  스레드 127: partial_product = a[127] * b[127] = 32258.0

block.sum() 하드웨어 연산:
모든 스레드 → 0.0 + 2.0 + 1922.0 + 2048.0 + ... + 32258.0 = 1381760.0
스레드 0이 수신 → total = 1381760.0 (broadcast=False일 때)
```

**Barrier 없이 동작하는 이유:**

1. **블록 전체 실행**: 모든 스레드가 Warp 내에서 lockstep으로 각 명령을 실행
2. **내장 동기화**: `block.sum()` 구현이 동기화를 내부적으로 처리
3. **크로스 Warp 통신**: 블록 내 Warp 간 최적화된 통신
4. **조율된 결과 전달**: 스레드 0만 최종 결과를 수신

**warp.sum() (Puzzle 24)과의 비교:**

- **Warp 범위**: `warp.sum()`은 32/64 스레드(단일 Warp) 내에서 동작
- **블록 범위**: `block.sum()`은 전체 블록(여러 Warp)에 걸쳐 동작
- **동일한 단순함**: 둘 다 복잡한 수동 reduction을 한 줄 호출로 대체
- **자동 조율**: `block.sum()`은 `warp.sum()`이 처리할 수 없는 크로스 Warp barrier를 자동으로 처리

</div>
</details>

## 기술 분석: block.sum()은 실제로 무엇으로 컴파일될까?

`block.sum()`이 실제로 무엇을 생성하는지 이해하기 위해, 디버그 정보와 함께 퍼즐을 컴파일했습니다:

```bash
pixi run mojo build --emit llvm --debug-level=line-tables solutions/p27/p27.mojo -o solutions/p27/p27.ll
```

이렇게 생성된 **LLVM 파일** `solutions/p27/p27.ll`에는, 호환 NVIDIA GPU에서 실제 GPU 명령을 보여주는 **PTX 어셈블리**가 내장되어 있습니다:

### **발견 1: 단일 명령이 아니다**

`block.sum()`은 약 **20개 이상의 PTX 명령**으로 컴파일되며, 2단계 reduction으로 구성됩니다:

**1단계: Warp 레벨 reduction (butterfly shuffle)**

```ptx
shfl.sync.bfly.b32 %r23, %r46, 16, 31, -1;   // offset 16으로 shuffle
add.f32            %r24, %r46, %r23;         // shuffle된 값을 합산
shfl.sync.bfly.b32 %r25, %r24, 8, 31, -1;    // offset 8로 shuffle
add.f32            %r26, %r24, %r25;         // shuffle된 값을 합산
// ... offset 4, 2, 1에 대해 계속
```

**2단계: 크로스 Warp 조율**

```ptx
shr.u32            %r32, %r1, 5;             // Warp ID 계산
mov.b32            %r34, _global_alloc_$__gpu_shared_mem; // 공유 메모리
bar.sync           0;                        // barrier 동기화
// ... 크로스 Warp reduction을 위한 또 다른 butterfly shuffle 시퀀스
```

### **발견 2: 하드웨어 최적화 구현**

- **Butterfly shuffle**: 트리 reduction보다 효율적
- **자동 barrier 배치**: 크로스 Warp 동기화를 자동으로 처리
- **최적화된 메모리 접근**: 공유 메모리를 전략적으로 사용
- **아키텍처 인식**: 동일한 API가 NVIDIA(32 스레드 Warp)와 AMD(32 또는 64 스레드 Warp)에서 동작

### **발견 3: 알고리즘 복잡도 분석**

**분석 접근 방식:**

1. 바이너리 ELF 섹션(`.nv_debug_ptx_txt`)에서 PTX 어셈블리를 확인
2. 개별 명령 수를 세기보다 알고리즘적 차이를 식별

**관찰된 주요 알고리즘 차이:**

- **기존 방식**: 공유 메모리를 사용한 트리 reduction + 다수의 `bar.sync` 호출
- **block.sum()**: Butterfly shuffle 패턴 + 최적화된 크로스 Warp 조율

성능 이점은 명령 수나 마법 같은 하드웨어가 아니라 **정교하게 최적화된 알고리즘 선택**(butterfly > tree)에서 비롯됩니다. 구현에 대한 자세한 내용은 Mojo gpu 모듈의 [block.mojo]를 참고하세요.

## 성능 인사이트

**`block.sum()` vs 기존 방식:**

- **코드 단순함**: reduction 부분이 15줄 이상 → 1줄로
- **메모리 사용**: 공유 메모리 할당 불필요
- **동기화**: 명시적 barrier 불필요
- **확장성**: 하드웨어 한도 내에서 모든 블록 크기에 동작

**`block.sum()` vs `warp.sum()`:**

- **범위**: 블록 전체(128 스레드) vs Warp 전체(32 스레드)
- **용도**: 전체 블록에 걸친 reduction이 필요할 때
- **편의성**: 동일한 프로그래밍 모델, 다른 규모

**`block.sum()`을 사용해야 할 때:**

- **단일 블록 문제**: 모든 데이터가 하나의 블록에 들어갈 때
- **블록 레벨 알고리즘**: reduction이 필요한 공유 메모리 연산
- **확장성보다 편의성**: 멀티 블록 방식보다 단순

## 이전 퍼즐과의 관계

**Puzzle 12 (기존 방식)에서:**

```
복잡함: 공유 메모리 + barrier + 트리 reduction
↓
단순함: block.sum() 하드웨어 기본 요소
```

**Puzzle 24 (`warp.sum()`)에서:**

```
Warp 레벨: warp.sum() - 32 스레드 (단일 Warp)
↓
블록 레벨: block.sum() - 128 스레드 (여러 Warp)
```

**3단계 진행:**

1. **수동 reduction** (Puzzle 12): 복잡한 공유 메모리 + barrier + 트리 reduction
2. **Warp 기본 요소** (Puzzle 24): `warp.sum()` - 단순하지만 단일 Warp로 제한
3. **블록 기본 요소** (Puzzle 27): `block.sum()` - Warp의 단순함을 여러 Warp로 확장

**핵심 통찰:** `block.sum()`은 `warp.sum()`의 단순함을 제공하면서 전체 블록으로 확장됩니다. 수동으로 구현해야 했던 복잡한 크로스 Warp 조율을 자동으로 처리합니다.

## 다음 단계

`block.sum()` 연산을 배웠으니, 다음으로 진행할 수 있습니다:

- **[block.prefix_sum()과 병렬 히스토그램 구간 분류](./block_prefix_sum.md)**: 블록 스레드에 걸친 누적 연산
- **[block.broadcast()와 벡터 정규화](./block_broadcast.md)**: 블록 내 모든 스레드에 값을 공유

💡 **핵심 요점**: 블록 연산은 Warp 프로그래밍 개념을 전체 스레드 블록으로 확장하여, 여러 Warp에 걸쳐 동시에 동작하면서 복잡한 동기화 패턴을 대체하는 최적화된 기본 요소를 제공합니다. `warp.sum()`이 Warp 레벨 reduction을 단순화한 것처럼, `block.sum()`은 성능을 희생하지 않고 블록 레벨 reduction을 단순화합니다.
