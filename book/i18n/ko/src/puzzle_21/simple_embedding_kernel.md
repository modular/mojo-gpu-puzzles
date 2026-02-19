<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# Embedding 커널: Coalesced vs Non-Coalesced

이 퍼즐에서는 동일한 결과를 생성하지만 서로 다른 메모리 접근 패턴을 사용하는 두 가지 GPU embedding 커널을 구현합니다. GPU 성능에서 메모리 병합이 얼마나 중요한지 직접 체험할 수 있습니다.

## 1D Coalesced 커널 (최적화된 접근법)

이 커널은 각 스레드가 정확히 하나의 출력 요소를 처리하는 단순한 1D 그리드를 사용합니다. 핵심은 연속된 스레드가 연속된 메모리 위치에 접근하여 최적의 메모리 병합을 달성한다는 점입니다.

**스레드 구성:**

- **그리드 구성**: `[total_elements // 256]` 블록, 블록당 `256` 스레드
- **스레드 매핑**: 각 스레드가 하나의 `(batch, seq, embed)` 위치 처리
- **메모리 패턴**: 연속된 스레드가 연속된 embedding 차원 접근

**구현할 내용:**

1. 블록 인덱스와 스레드 인덱스로부터 전역 스레드 인덱스 계산
2. 1차원 인덱스를 3D 좌표 `(batch_idx, seq_idx, embed_idx)`로 변환
3. indices 텐서에서 토큰 인덱스 조회
4. 해당하는 embedding 벡터 요소를 출력에 복사

### 완성할 코드

두 embedding 커널의 빈 부분을 완성해야 합니다:

```mojo
{{#include ../../../../../problems/p21/op/embedding.mojo:embedding_kernel_coalesced}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p21/op/embedding.mojo" class="filename">전체 파일 보기: problems/p21/op/embedding.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

- `global_idx = block_idx.x * block_dim.x + thread_idx.x`로 시작하세요
- 나눗셈과 나머지 연산으로 3D 좌표를 구합니다: `batch_idx = global_idx // (seq_len * embed_dim)`
- `remaining = global_idx % (seq_len * embed_dim)`을 사용하면 이후 계산이 간단해집니다
- 항상 경계 검사를 하세요: `if global_idx >= total_elements: return`
- 유효하지 않은 토큰 인덱스는 출력을 0으로 설정하세요
- Embedding 조회: `output[batch_idx, seq_idx, embed_idx] = weights[token_idx, embed_idx]`

</div>
</details>

## 2D Non-coalesced 커널 (비교용 접근법)

이 커널은 X 차원이 `(batch × seq)` 위치를, Y 차원이 embedding 차원을 담당하는 2D 그리드를 사용합니다. 이 방식은 메모리 접근이 병합되지 않을 수 있습니다.

**스레드 구성:**

- **그리드 구성**: `[batch x seq // 16, embed_dim // 16]` 블록, `16 x 16` 스레드
- **스레드 매핑**: `thread_idx.x`는 batch/sequence에, `thread_idx.y`는 embedding 차원에 매핑
- **메모리 패턴**: Warp 내 스레드들이 흩어진 메모리 위치에 접근할 수 있음

**구현할 내용:**

1. 2D 그리드에서 X, Y 좌표 계산
2. X 좌표를 batch 인덱스와 sequence 인덱스로 분리
3. Y 좌표를 embedding 차원으로 직접 사용
4. 경계 검사와 함께 동일한 embedding 조회 수행

### 완성할 코드

두 embedding 커널의 빈 부분을 완성해야 합니다:

```mojo
{{#include ../../../../../problems/p21/op/embedding.mojo:embedding_kernel_2d}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p21/op/embedding.mojo" class="filename">전체 파일 보기: problems/p21/op/embedding.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

- X, Y 스레드 좌표를 모두 사용합니다: `batch_seq_idx = block_idx.x * block_dim.x + thread_idx.x`
- 그리고: `embed_idx = block_idx.y * block_dim.y + thread_idx.y`
- `batch_seq_idx`를 batch와 sequence 인덱스로 분리합니다: `batch_idx = batch_seq_idx // seq_len`
- 두 차원 모두 경계 검사를 잊지 마세요: `if batch_seq_idx >= total_positions or embed_idx >= embed_dim`
- 토큰 조회는 1D와 동일하지만, 스레드당 하나의 embedding 차원만 처리합니다
- 이 커널은 전체 벡터가 아닌 스레드당 하나의 embedding 차원을 처리합니다

</div>
</details>

## 커스텀 op 등록

커널들은 PyTorch와 쉽게 통합할 수 있도록 커스텀 연산으로 래핑됩니다. 등록 패턴은 [MAX Graph 커스텀 op 이해하기](../puzzle_17/puzzle_17.md#max-graph-커스텀-op-이해하기)에서 설명한 MAX 커스텀 op과 동일합니다:

### 1D Coalesced 연산

이 연산은 최적화된 1D embedding 커널을 `"embedding"`으로 등록합니다:

```mojo
{{#include ../../../../../solutions/p21/op/embedding.mojo:embedding_custom_op_solution}}
```

**등록의 핵심 요소:**

- **단순한 그리드 구성**: `ceildiv(total_elements, THREADS_PER_BLOCK)` 블록으로 직관적인 1D 그리드 사용
- **메모리 최적화**: 단일 `enqueue_memset` 호출로 출력 버퍼를 효율적으로 초기화
- **컴파일 타임 파라미터**: 모든 텐서 차원을 컴파일 타임 파라미터로 전달하여 최적 성능 달성
- **디바이스 추상화**: GPU 실행과 CPU 폴백을 매끄럽게 처리

### 2D Non-coalesced 연산

이 연산은 비교용 2D embedding 커널을 `"embedding_2d"`로 등록합니다:

```mojo
{{#include ../../../../../solutions/p21/op/embedding.mojo:embedding_2d_custom_op_solution}}
```

**1D 연산과의 주요 차이점:**

- **복잡한 그리드 구성**: `blocks_x`와 `blocks_y`를 별도로 계산하는 2D 그리드 사용
- **고정 블록 차원**: 2D 스레드 구성을 위해 `BLOCK_X = 16`, `BLOCK_Y = 16`으로 고정
- **동일한 메모리 관리**: 메모리 초기화와 CPU 폴백 로직은 동일
- **다른 커널 호출 방식**: 2D 그리드 차원 `(blocks_x, blocks_y)`과 블록 차원 `(BLOCK_X, BLOCK_Y)` 전달

### 공통 래퍼 기능

두 커스텀 연산은 다음과 같은 필수 인프라를 제공합니다:

1. **메모리 관리**:
   - `enqueue_memset`으로 출력 텐서 0 초기화
   - 적절한 버퍼 생성과 메모리 레이아웃 처리
   - 자동 정리 및 리소스 관리

2. **디바이스 추상화**:
   - 최적화된 커널로 GPU 실행
   - 호환성과 디버깅을 위한 CPU 폴백
   - 실행 대상에 관계없이 일관된 인터페이스

3. **파라미터 전달**:
   - 커널 최적화를 위한 컴파일 타임 텐서 차원
   - 레이아웃 텐서 변환을 통한 런타임 텐서 데이터
   - 타입 안전한 파라미터 검증

4. **그리드 구성**:
   - 최적의 그리드 차원 자동 계산
   - 각 커널의 접근 패턴에 최적화된 서로 다른 전략
   - 적절한 블록 차원 관리

### PyTorch 통합

등록된 연산은 [CustomOpLibrary](https://docs.modular.com/max/api/python/torch/)를 통해 Python에서 호출할 수 있습니다:

```python
# Load the custom operations
ops = CustomOpLibrary(mojo_kernels)

# Call the 1D coalesced version
result_1d = ops.embedding[{"batch_size": B, "seq_len": L, "vocab_size": V, "embed_dim": E}](
    indices, weights
)

# Call the 2D non-coalesced version
result_2d = ops.embedding_2d[{"batch_size": B, "seq_len": L, "vocab_size": V, "embed_dim": E}](
    indices, weights
)
```

이 접근법의 장점은 동일한 커널 구현을 다양한 Python 프레임워크에서 사용하면서도 최적의 성능 특성을 유지할 수 있다는 것입니다.

## 코드 실행

다음 명령으로 퍼즐을 실행할 수 있습니다:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p21
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p21
```

  </div>
  <div class="tab-content">

```bash
uv run poe p21
```

  </div>
</div>

성공하면 다음과 비슷한 출력을 볼 수 있습니다:

```
Puzzle 21: Mojo Embedding Kernel Comparison
======================================================================
Configuration: B=8, L=512, V=10000, E=512
------------------------------------------------------------

Testing Correctness...
   1D Coalesced - Max difference: 1.19e-07
   2D Non-coalesced - Max difference: 1.19e-07
   ✅ Both implementations CORRECT

Benchmarking Mojo Kernels...

Performance Results:
   1D Coalesced:     2.145 ms
   2D Non-coalesced: 3.867 ms
   1D is 1.80x faster than 2D

Key Learning Points:
• Compare different GPU kernel implementations
• 1D vs 2D grid patterns have different memory access
• Coalesced memory access should be faster
• Grid configuration affects GPU utilization
```

## 풀이

<details class="solution-details">
<summary></summary>

두 커널의 좌표 변환과 메모리 연산을 구현하면 됩니다:

## 1D Coalesced 커널

```mojo
{{#include ../../../../../solutions/p21/op/embedding.mojo:embedding_kernel_coalesced_solution}}
```

## 2D Non-Coalesced 커널

```mojo
{{#include ../../../../../solutions/p21/op/embedding.mojo:embedding_kernel_2d_solution}}
```

<div class="solution-explanation">

두 풀이 모두 동일한 embedding 조회 로직을 구현하지만 스레드 구성이 다릅니다:

### 주요 차이점

1. **스레드 매핑**:
   - **1D 커널**: 출력 요소당 하나의 스레드, 단순한 1차원 인덱싱
   - **2D 커널**: (batch×seq, embed_dim) 좌표에 대한 2D 그리드 매핑

2. **메모리 접근 패턴**:
   - **1D 커널**: 연속된 스레드가 연속된 embedding 차원에 접근 → 병합됨
   - **2D 커널**: 스레드 접근 패턴이 블록 구성에 따라 달라짐 → 병합되지 않을 수 있음

3. **인덱싱 복잡도**:
   - **1D 커널**: 단일 나눗셈/나머지 체인으로 3D 좌표 계산
   - **2D 커널**: X/Y 좌표를 별도로 계산

### 성능에 미치는 영향

1D 커널이 일반적으로 더 높은 성능을 보이는 이유:

- **메모리 병합**: 연속된 스레드가 연속된 메모리 주소에 접근
- **단순한 인덱싱**: 좌표 계산의 연산 오버헤드가 낮음
- **더 나은 캐시 활용**: 예측 가능한 메모리 접근 패턴

2D 커널의 성능이 떨어질 수 있는 이유:

- **흩어진 메모리 접근**: Warp 내 스레드들이 서로 다른 embedding 벡터에 접근할 수 있음
- **복잡한 그리드 구성**: 16×16 블록이 메모리 레이아웃과 최적으로 맞지 않을 수 있음
- **Warp 분기**: 서로 다른 스레드가 서로 다른 실행 경로를 따를 수 있음

</div>

</details>

## 핵심 개념

| 개념 | 1D Coalesced | 2D Non-coalesced |
|---------|---------------|-------------------|
| **스레드 구성** | 1D 1차원 인덱싱 | 2D 그리드 (batch×seq, embed) |
| **메모리 접근** | 연속된 주소 | 흩어질 수 있음 |
| **그리드 구성** | 단순: `[total_elements // 256]` | 복잡: `[batch×seq // 16, embed // 16]` |
| **성능** | 메모리 대역폭에 최적화 | 최적화되지 않은 메모리 패턴 |
| **사용 목적** | 프로덕션 커널 | 교육용 비교 |

핵심 교훈: **메모리 병합**은 embedding과 같은 메모리 바운드 연산에서 2~3배의 성능 차이를 가져올 수 있습니다.
