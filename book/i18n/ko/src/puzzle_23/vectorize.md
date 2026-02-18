<!-- i18n-source-commit: 9c7176b81f278a6e8efa26c92005c139967c0c27 -->

# Vectorize - SIMD 제어

## 개요

이 퍼즐에서는 수동 벡터화와 [vectorize](https://docs.modular.com/mojo/stdlib/algorithm/functional/vectorize/)를 사용하여 GPU 커널 내에서 SIMD 연산을 정밀하게 제어하는 **고급 벡터화 기법**을 탐구합니다. 벡터화된 연산에 대해 두 가지 다른 접근법을 구현합니다:

1. **수동 벡터화**: 명시적 인덱스 계산을 통한 직접적인 SIMD 제어
2. **Mojo의 vectorize 함수**: 자동 경계 검사를 포함한 고수준 벡터화

두 접근법 모두 tiling 개념을 기반으로 하지만, 제어, 안전성, 성능 최적화 간의 트레이드오프가 다릅니다.

**핵심 통찰:** _벡터화 전략은 성능 요구 사항과 복잡도 수준에 따라 달리 선택해야 합니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- 명시적 인덱스 관리를 통한 **수동 SIMD 연산**
- 안전하고 자동적인 벡터화를 위한 **Mojo의 vectorize 함수**
- 최적의 SIMD 정렬을 위한 **chunk 기반 메모리 구성**
- 경계 조건을 위한 **경계 검사 전략**
- 수동 제어와 안전성 간의 **성능 트레이드오프**

이전과 동일한 수학적 연산:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

하지만 최대 성능을 위한 정교한 벡터화 전략을 사용합니다.

## 설정

- 벡터 크기: `SIZE = 1024`
- 타일 크기: `TILE_SIZE = 32`
- 데이터 타입: `DType.float32`
- SIMD 폭: GPU 의존적
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)

## 1. 수동 벡터화 방식

### 완성할 코드

```mojo
{{#include ../../../../../problems/p23/p23.mojo:manual_vectorized_tiled_elementwise_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p23/p23.mojo" class="filename">전체 파일 보기: problems/p23/p23.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **Chunk 구성 이해하기**

```mojo
comptime chunk_size = tile_size * simd_width  # 32 * 4 = chunk당 128개 요소
```

각 타일은 이제 단순한 순차 요소가 아닌 여러 SIMD 그룹을 포함합니다.

### 2. **전역 인덱스 계산**

```mojo
global_start = tile_id * chunk_size + i * simd_width
```

Chunk 내 각 SIMD 벡터의 정확한 전역 위치를 계산합니다.

### 3. **텐서 직접 접근**

```mojo
a_vec = a.load[simd_width](global_start, 0)     # 전역 텐서에서 로드
output.store[simd_width](global_start, 0, ret)  # 전역 텐서에 저장
```

참고: 타일 뷰가 아닌 원본 텐서에 접근합니다.

### 4. **주요 특성**

- 더 많은 제어, 더 많은 복잡성, 전역 텐서 접근
- 하드웨어에 대한 완벽한 SIMD 정렬
- 수동 경계 검사 필요

</div>
</details>

### 수동 벡터화 실행

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p23 --manual-vectorized
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p23 --manual-vectorized
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p23 --manual-vectorized
```

  </div>
  <div class="tab-content">

```bash
uv run poe p23 --manual-vectorized
```

  </div>
</div>

퍼즐이 아직 풀리지 않은 경우 다음과 같이 출력됩니다:

```txt
SIZE: 1024
simd_width: 4
tile size: 32
tile_id: 0
tile_id: 1
tile_id: 2
tile_id: 3
tile_id: 4
tile_id: 5
tile_id: 6
tile_id: 7
out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

### 수동 벡터화 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p23/p23.mojo:manual_vectorized_tiled_elementwise_add_solution}}
```

<div class="solution-explanation">

### 수동 벡터화 심층 분석

**수동 벡터화**는 명시적 인덱스 계산을 통해 SIMD 연산에 대한 직접적인 제어를 제공합니다:

- **Chunk 기반 구성**: `chunk_size = tile_size * simd_width`
- **전역 인덱싱**: 메모리 위치의 직접 계산
- **수동 경계 관리**: 경계 조건을 직접 처리

**아키텍처와 메모리 레이아웃:**

```mojo
comptime chunk_size = tile_size * simd_width  # 32 * 4 = 128
```

**Chunk 구성 시각화 (TILE_SIZE=32, SIMD_WIDTH=4):**

```
원본 배열: [0, 1, 2, 3, ..., 1023]

Chunk 0 (thread 0): [0:128]    ← 128개 요소 = 4개씩 32개 SIMD 그룹
Chunk 1 (thread 1): [128:256]  ← 다음 128개 요소
Chunk 2 (thread 2): [256:384]  ← 다음 128개 요소
...
Chunk 7 (thread 7): [896:1024] ← 마지막 128개 요소
```

**하나의 chunk 내 처리:**

```mojo
@parameter
for i in range(tile_size):  # i = 0, 1, 2, ..., 31
    global_start = tile_id * chunk_size + i * simd_width
    # tile_id=0일 때: global_start = 0, 4, 8, 12, ..., 124
    # tile_id=1일 때: global_start = 128, 132, 136, 140, ..., 252
```

**성능 특성:**

- **스레드 수**: 8개 스레드 (1024 ÷ 128 = 8)
- **스레드당 작업량**: 128개 요소 (각 4개 요소의 SIMD 연산 32회)
- **메모리 패턴**: 완벽한 SIMD 정렬을 갖춘 대형 chunk
- **오버헤드**: 최소 - 하드웨어에 직접 매핑
- **안전성**: 수동 경계 검사 필요

**주요 장점:**

- **예측 가능한 인덱싱**: 메모리 접근 패턴에 대한 정확한 제어
- **최적의 정렬**: SIMD 연산이 하드웨어에 완벽히 정렬
- **최대 처리량**: 안전성 검사로 인한 오버헤드 없음
- **하드웨어 최적화**: GPU SIMD 유닛에 직접 매핑

**주요 과제:**

- **인덱스 복잡성**: 전역 위치의 수동 계산
- **경계 처리 책임**: 경계 조건을 직접 처리해야 함
- **디버깅 난이도**: 정확성 검증이 더 복잡

</div>
</details>

## 2. Mojo vectorize 방식

### 완성할 코드

```mojo
{{#include ../../../../../problems/p23/p23.mojo:vectorize_within_tiles_elementwise_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p23/p23.mojo" class="filename">전체 파일 보기: problems/p23/p23.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **타일 경계 계산**

```mojo
tile_start = tile_id * tile_size
tile_end = min(tile_start + tile_size, size)
actual_tile_size = tile_end - tile_start
```

마지막 타일이 `tile_size`보다 작을 수 있는 경우를 처리합니다.

### 2. **벡터화 함수 패턴**

```mojo
fn vectorized_add[
  width: Int
](i: Int) unified {read tile_start, read a, read b, mut output}:
    global_idx = tile_start + i
    if global_idx + width <= size:  # 경계 검사
        # SIMD 연산 코드
```

`width` 매개변수는 vectorize 함수에 의해 자동으로 결정됩니다.

### 3. **vectorize 호출**

```mojo
vectorize[simd_width](actual_tile_size, vectorized_add)
```

제공된 SIMD 폭으로 벡터화 루프를 자동 처리합니다.

### 4. **주요 특성**

- 자동 나머지 처리, 내장 안전성, 타일 기반 접근
- 명시적 SIMD 폭 매개변수 사용
- 내장 경계 검사와 자동 나머지 요소 처리

</div>
</details>

### Mojo vectorize 실행

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">uv</button>
    <button class="tab-button">pixi</button>
  </div>
  <div class="tab-content">

```bash
uv run poe p23 --vectorized
```

  </div>
  <div class="tab-content">

```bash
pixi run p23 --vectorized
```

  </div>
</div>

퍼즐이 아직 풀리지 않은 경우 다음과 같이 출력됩니다:

```txt
SIZE: 1024
simd_width: 4
tile size: 32
tile_id: 0 tile_start: 0 tile_end: 32 actual_tile_size: 32
tile_id: 1 tile_start: 32 tile_end: 64 actual_tile_size: 32
tile_id: 2 tile_start: 64 tile_end: 96 actual_tile_size: 32
tile_id: 3 tile_start: 96 tile_end: 128 actual_tile_size: 32
...
tile_id: 29 tile_start: 928 tile_end: 960 actual_tile_size: 32
tile_id: 30 tile_start: 960 tile_end: 992 actual_tile_size: 32
tile_id: 31 tile_start: 992 tile_end: 1024 actual_tile_size: 32
out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

### Mojo vectorize 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p23/p23.mojo:vectorize_within_tiles_elementwise_add_solution}}
```

<div class="solution-explanation">

### Mojo vectorize 심층 분석

**Mojo의 vectorize 함수**는 내장 안전성과 함께 자동 벡터화를 제공합니다:

- **명시적 SIMD 폭 매개변수**: 사용할 simd_width를 직접 지정
- **내장 경계 검사**: 버퍼 오버플로우를 자동으로 방지
- **자동 나머지 처리**: 남은 요소를 자동으로 처리
- **중첩 함수 패턴**: 벡터화 로직의 깔끔한 분리

**타일 기반 구성:**

```mojo
tile_start = tile_id * tile_size    # 0, 32, 64, 96, ...
tile_end = min(tile_start + tile_size, size)
actual_tile_size = tile_end - tile_start
```

**자동 벡터화 메커니즘:**

```mojo
fn vectorized_add[
  width: Int
](i: Int) unified {read tile_start, read a, read b, mut output}:
    global_idx = tile_start + i
    if global_idx + width <= size:
        # 자동 SIMD 최적화
```

**Vectorize의 동작 방식:**

- **자동 chunk 분할**: `actual_tile_size`를 지정한 `simd_width`의 chunk로 분할
- **나머지 처리**: 남은 요소를 더 작은 폭으로 자동 처리
- **경계 안전성**: 버퍼 오버플로우를 자동으로 방지
- **루프 관리**: 벡터화 루프를 자동으로 처리

**실행 시각화 (TILE_SIZE=32, SIMD_WIDTH=4):**

```
Tile 0 처리:
  vectorize 호출 0: 요소 [0:4]를 SIMD_WIDTH=4로 처리
  vectorize 호출 1: 요소 [4:8]를 SIMD_WIDTH=4로 처리
  ...
  vectorize 호출 7: 요소 [28:32]를 SIMD_WIDTH=4로 처리
  합계: 8회 자동 SIMD 연산
```

**성능 특성:**

- **스레드 수**: 32개 스레드 (1024 ÷ 32 = 32)
- **스레드당 작업량**: 32개 요소 (자동 SIMD chunk 분할)
- **메모리 패턴**: 자동 벡터화를 갖춘 작은 타일
- **오버헤드**: 약간 - 자동 최적화 및 경계 검사
- **안전성**: 내장 경계 검사와 경계 조건 처리

</div>
</details>

## 성능 비교와 모범 사례

### 각 접근법의 선택 기준

**수동 벡터화를 선택할 때:**

- **최대 성능**이 중요한 경우
- **예측 가능하고 정렬된 데이터** 패턴이 있는 경우
- 메모리 접근에 대한 **전문가 수준의 제어**가 필요한 경우
- 수동으로 **경계 안전성을 보장**할 수 있는 경우
- **하드웨어별 최적화**가 필요한 경우

**Mojo vectorize를 선택할 때:**

- **개발 속도**와 안전성이 우선인 경우
- **불규칙하거나 동적인 데이터 크기**를 다루는 경우
- 수동 경계 조건 관리 대신 **자동 나머지 처리**를 원하는 경우
- **경계 검사** 복잡도가 오류를 유발할 수 있는 경우
- 수동 루프 관리보다 **깔끔한 벡터화 패턴**을 선호하는 경우

### 고급 최적화 인사이트

**메모리 대역폭 활용:**

```
Manual:    8 스레드 × 32 SIMD 연산 = 총 256회 SIMD 연산
Vectorize: 32 스레드 × 8 SIMD 연산 = 총 256회 SIMD 연산
```

둘 다 비슷한 총 처리량을 달성하지만, 병렬성 전략이 다릅니다.

**캐시 동작:**

- **수동**: 대형 chunk가 L1 캐시를 초과할 수 있지만, 완벽한 순차 접근
- **Vectorize**: 작은 타일이 캐시에 더 잘 맞고, 자동 나머지 처리

**하드웨어 매핑:**

- **수동**: Warp 활용과 SIMD 유닛 매핑에 대한 직접 제어
- **Vectorize**: 자동 루프 및 나머지 관리를 통한 간소화된 벡터화

### 모범 사례 요약

**수동 벡터화 모범 사례:**

- 인덱스 계산을 항상 신중하게 검증
- 가능하면 `chunk_size`에 컴파일 타임 상수 사용
- 캐시 최적화를 위해 메모리 접근 패턴 프로파일링
- 최적의 SIMD 성능을 위한 정렬 요구 사항 고려

**Mojo vectorize 모범 사례:**

- 데이터와 하드웨어에 적합한 SIMD 폭 선택
- 미세 최적화보다 알고리즘의 명확성에 집중
- 깔끔한 벡터화 로직을 위해 중첩 파라미터 함수 사용
- 경계 조건에는 자동 경계 검사와 나머지 처리 신뢰

두 접근법 모두 GPU 성능 최적화 도구 모음에서 유효한 전략입니다. 수동 벡터화는 최대한의 제어를, Mojo의 vectorize는 안전성과 자동 나머지 처리를 제공합니다.

## 다음 단계

세 가지 기본 패턴을 모두 이해했다면:

- **[🧠 GPU 스레딩 vs SIMD 개념](./gpu-thread-vs-simd.md)**: 실행 계층 구조 이해
- **[📊 Mojo 벤치마킹](./benchmarking.md)**: 성능 분석과 최적화

💡 **핵심 요약**: 벡터화 전략은 성능 요구 사항에 따라 달리 선택해야 합니다. 수동 벡터화는 최대한의 제어를, Mojo의 vectorize 함수는 안전성과 자동 나머지 처리를 제공합니다. 구체적인 성능 요구 사항과 개발 제약 조건에 따라 선택하세요.
