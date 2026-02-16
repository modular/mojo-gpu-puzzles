<!-- i18n-source-commit: 5426b744b3cbf1861feb709814917d33f03bb103 -->

# Tile - 메모리 효율적인 Tiled 처리

## 개요

**Elementwise** 패턴을 기반으로, 이 퍼즐에서는 **tiled 처리**를 소개합니다. 이는 GPU에서 메모리 접근 패턴과 캐시 활용을 최적화하는 핵심 기법입니다. 각 스레드가 전체 배열에 걸쳐 개별 SIMD 벡터를 처리하는 대신, tiling은 데이터를 캐시 메모리에 더 잘 맞는 작고 관리 가능한 chunk로 구성합니다.

**[Puzzle 16의 Tiled 행렬 곱셈](../puzzle_16/tiled.md)** 에서 이미 tiling을 경험한 바 있습니다. 거기서는 타일을 사용해 대규모 행렬을 효율적으로 처리했습니다. 여기서는 동일한 tiling 원칙을 벡터 연산에 적용하여, 이 기법이 2D 행렬에서 1D 배열까지 어떻게 확장되는지 보여줍니다.

Mojo의 tiled 방식을 사용하여 동일한 벡터 덧셈 연산을 구현합니다. 각 GPU 스레드가 데이터의 타일 전체를 순차적으로 처리하며, 메모리 지역성이 특정 워크로드에서 어떻게 성능을 향상시킬 수 있는지 보여줍니다.

**핵심 통찰:** _Tiling은 병렬 폭을 메모리 지역성과 교환합니다 - 더 적은 수의 스레드가 더 나은 캐시 활용으로 더 많은 작업을 수행합니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- 캐시 최적화를 위한 **타일 기반 메모리 구성**
- 타일 내의 **순차적 SIMD 처리**
- **메모리 지역성 원칙**과 캐시 친화적 접근 패턴
- **스레드-타일 매핑** vs 스레드-요소 매핑
- 병렬성과 메모리 효율 간의 **성능 트레이드오프**

Elementwise와 동일한 수학적 연산:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

하지만 메모리 계층 구조에 최적화된 완전히 다른 실행 전략을 사용합니다.

## 설정

- 벡터 크기: `SIZE = 1024`
- 타일 크기: `TILE_SIZE = 32`
- 데이터 타입: `DType.float32`
- SIMD 폭: GPU 의존적 (타일 내 연산용)
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)

## 작성할 코드

```mojo
{{#include ../../../../../problems/p23/p23.mojo:tiled_elementwise_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p23/p23.mojo" class="filename">전체 파일 보기: problems/p23/p23.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **타일 구성 이해하기**

Tiled 방식은 데이터를 고정 크기의 chunk로 나눕니다:

```mojo
num_tiles = (size + tile_size - 1) // tile_size  # 올림 나눗셈
```

`TILE_SIZE=32`인 1024개 요소 벡터의 경우: `1024 ÷ 32 = 32`개 타일이 정확히 생깁니다.

### 2. **타일 추출 패턴**

[LayoutTensor `.tile` 문서](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/#tile)를 참고하세요.

```mojo
tile_id = indices[0]  # 각 스레드가 처리할 타일 하나를 받음
out_tile = output.tile[tile_size](tile_id)
a_tile = a.tile[tile_size](tile_id)
b_tile = b.tile[tile_size](tile_id)
```

`tile[size](id)` 메서드는 `id × size` 위치부터 시작하는 `size`개의 연속 요소에 대한 뷰를 생성합니다.

### 3. **타일 내 순차 처리**

Elementwise와 달리, 타일을 순차적으로 처리합니다:

```mojo
@parameter
for i in range(tile_size):
    # 현재 타일 내의 요소 i를 처리
```

이 `@parameter` 루프는 최적의 성능을 위해 컴파일 타임에 전개됩니다.

### 4. **타일 요소 내 SIMD 연산**

```mojo
a_vec = a_tile.load[simd_width](i, 0)  # 타일 내 위치 i에서 로드
b_vec = b_tile.load[simd_width](i, 0)  # 타일 내 위치 i에서 로드
result = a_vec + b_vec                 # SIMD 덧셈 (GPU 의존적 폭)
out_tile.store[simd_width](i, 0, result)  # 타일 내 위치 i에 저장
```

### 5. **스레드 구성의 차이점**

```mojo
elementwise[process_tiles, 1, target="gpu"](num_tiles, ctx)
```

`SIMD_WIDTH` 대신 `1`을 사용합니다 - 각 스레드가 하나의 타일 전체를 순차적으로 처리합니다.

### 6. **메모리 접근 패턴 인사이트**

각 스레드는 연속적인 메모리 블록(타일)에 접근한 다음, 다음 타일로 이동합니다. 이렇게 하면 각 스레드의 실행 내에서 우수한 **공간 지역성**이 만들어집니다.

### 7. **디버깅 핵심 포인트**

tiling을 사용하면 스레드 실행 수는 줄어들지만 각 스레드가 더 많은 작업을 수행합니다:

- Elementwise: ~256개 스레드 (SIMD_WIDTH=4 기준), 각각 4개 요소 처리
- Tiled: ~32개 스레드, 각각 32개 요소를 순차적으로 처리

</div>
</details>

## 코드 실행

풀이를 테스트하려면 터미널에서 다음 명령을 실행하세요:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p23 --tiled
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p23 --tiled
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p23 --tiled
```

  </div>
  <div class="tab-content">

```bash
uv run poe p23 --tiled
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
...
tile_id: 29
tile_id: 30
tile_id: 31
out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p23/p23.mojo:tiled_elementwise_add_solution}}
```

<div class="solution-explanation">

tiled 처리 패턴은 GPU 프로그래밍을 위한 고급 메모리 최적화 기법을 보여줍니다:

### 1. **Tiling 철학과 메모리 계층 구조**

Tiling은 병렬 처리에 대한 사고 방식의 근본적인 전환을 나타냅니다:

**Elementwise 방식:**

- **넓은 병렬성**: 많은 스레드가 각각 최소한의 작업 수행
- **글로벌 메모리 부하**: 스레드들이 전체 배열에 분산
- **캐시 미스**: 스레드 경계를 넘나드는 낮은 공간 지역성

**Tiled 방식:**

- **깊은 병렬성**: 더 적은 스레드가 각각 상당한 작업 수행
- **지역화된 메모리 접근**: 각 스레드가 연속적인 데이터에서 작업
- **캐시 최적화**: 우수한 공간 및 시간 지역성

### 2. **타일 구성과 인덱싱**

```mojo
tile_id = indices[0]
out_tile = output.tile[tile_size](tile_id)
a_tile = a.tile[tile_size](tile_id)
b_tile = b.tile[tile_size](tile_id)
```

**타일 매핑 시각화 (TILE_SIZE=32):**

```
원본 배열: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ..., 1023]

Tile 0 (thread 0): [0, 1, 2, ..., 31]      ← 요소 0-31
Tile 1 (thread 1): [32, 33, 34, ..., 63]   ← 요소 32-63
Tile 2 (thread 2): [64, 65, 66, ..., 95]   ← 요소 64-95
...
Tile 31 (thread 31): [992, 993, ..., 1023] ← 요소 992-1023
```

**핵심 인사이트:**

- `tile[size](id)`는 원본 텐서에 대한 **뷰**를 생성합니다
- 뷰는 제로 카피로 동작합니다 - 데이터를 복사하지 않고 포인터 연산만 수행
- 타일 경계는 항상 `tile_size` 단위로 정렬됩니다

### 3. **순차 처리 심층 분석**

```mojo
@parameter
for i in range(tile_size):
    a_vec = a_tile.load[simd_width](i, 0)
    b_vec = b_tile.load[simd_width](i, 0)
    ret = a_vec + b_vec
    out_tile.store[simd_width](i, 0, ret)
```

**왜 순차 처리인가?**

- **캐시 최적화**: 연속적인 메모리 접근이 캐시 히트율을 극대화
- **컴파일러 최적화**: `@parameter` 루프가 컴파일 타임에 완전히 전개됨
- **메모리 대역폭**: 순차 접근이 메모리 컨트롤러 설계에 부합
- **조정 비용 감소**: SIMD 그룹 간 동기화가 불필요

**하나의 타일 내 실행 패턴 (TILE_SIZE=32, SIMD_WIDTH=4):**

```
스레드가 타일을 순차 처리:
Step 0: 요소 [0:4]를 SIMD로 처리
Step 1: 요소 [4:8]를 SIMD로 처리
Step 2: 요소 [8:12]를 SIMD로 처리
...
Step 7: 요소 [28:32]를 SIMD로 처리
합계: 스레드당 8회 SIMD 연산 (32 ÷ 4 = 8)
```

### 4. **메모리 접근 패턴 분석**

**캐시 동작 비교:**

**Elementwise 패턴:**

```
Thread 0: 글로벌 위치 [0, 4, 8, 12, ...] 접근    ← Stride = SIMD_WIDTH
Thread 1: 글로벌 위치 [4, 8, 12, 16, ...] 접근   ← Stride = SIMD_WIDTH
...
결과: 메모리 접근이 전체 배열에 분산
```

**Tiled 패턴:**

```
Thread 0: 위치 [0:32]를 순차 접근               ← 연속적인 32개 요소 블록
Thread 1: 위치 [32:64]를 순차 접근             ← 다음 연속적인 32개 요소 블록
...
결과: 각 스레드 내에서 완벽한 공간 지역성
```

**캐시 효율 시사점:**

- **L1 캐시**: 작은 타일이 L1 캐시에 더 잘 맞아 캐시 미스 감소
- **메모리 대역폭**: 순차 접근이 유효 대역폭을 극대화
- **TLB 효율**: TLB 미스 감소 (_역주: TLB(Translation Lookaside Buffer)는 가상 주소를 물리 주소로 변환하는 캐시로, 미스가 줄면 메모리 접근이 빨라집니다_)
- **프리페칭**: 하드웨어 프리페처가 순차 패턴에서 최적으로 동작

### 5. **스레드 구성 전략**

```mojo
elementwise[process_tiles, 1, target="gpu"](num_tiles, ctx)
```

**왜 `SIMD_WIDTH` 대신 `1`인가?**

- **스레드 수**: `num_tiles × SIMD_WIDTH`가 아닌 정확히 `num_tiles`개의 스레드만 실행
- **작업 분배**: 각 스레드가 하나의 완전한 타일을 처리
- **로드 밸런싱**: 스레드당 더 많은 작업, 전체적으로 더 적은 스레드
- **메모리 지역성**: 각 스레드의 작업이 공간적으로 지역화

**성능 트레이드오프:**

- **더 적은 논리적 스레드**: 낮은 점유율에서 모든 GPU 코어를 활용하지 못할 수 있음
- **스레드당 더 많은 작업**: 더 나은 캐시 활용과 조정 오버헤드 감소
- **순차 접근**: 각 스레드 내에서 최적의 메모리 대역폭 활용
- **오버헤드 감소**: 스레드 실행 및 조정 오버헤드 감소

**중요 참고**: "더 적은 스레드"는 논리적 프로그래밍 모델을 의미합니다. GPU 스케줄러는 여러 Warp를 실행하고 메모리 지연 시 효율적으로 전환하여 높은 하드웨어 활용률을 달성할 수 있습니다.

### 6. **성능 특성**

**Tiling이 도움이 되는 경우:**

- **메모리 바운드 연산**: 메모리 대역폭이 병목인 경우
- **캐시 민감 워크로드**: 데이터 재사용의 이점이 있는 연산
- **복잡한 연산**: 요소당 연산량이 많은 경우
- **제한된 병렬성**: GPU 코어보다 스레드가 적은 경우

**Tiling이 불리한 경우:**

- **고도로 병렬적인 워크로드**: 최대 스레드 활용이 필요한 경우
- **단순한 연산**: 메모리 접근이 연산보다 지배적인 경우
- **불규칙적 접근 패턴**: tiling이 지역성을 개선하지 못하는 경우

**단순 덧셈 예시 (TILE_SIZE=32):**

- **스레드 수**: 256개 대신 32개 (8배 적음)
- **스레드당 작업량**: 4개 대신 32개 요소 (8배 많음)
- **메모리 패턴**: 순차 vs stride 접근
- **캐시 활용**: 훨씬 나은 공간 지역성

### 7. **고급 tiling 고려 사항**

**타일 크기 선택:**

- **너무 작으면**: 캐시 활용이 떨어지고, 오버헤드가 증가
- **너무 크면**: 캐시에 맞지 않을 수 있고, 병렬성이 감소
- **최적 지점**: L1 캐시 최적화를 위해 보통 16-64개 요소
- **현재 선택**: 32개 요소로 캐시 활용과 병렬성의 균형 달성

**하드웨어 고려 사항:**

- **캐시 크기**: 가능하면 타일이 L1 캐시에 맞아야 함
- **메모리 대역폭**: 메모리 컨트롤러 폭을 고려
- **코어 수**: 모든 코어를 활용하기에 충분한 타일 확보
- **SIMD 폭**: 타일 크기는 SIMD 폭의 배수여야 함

**비교 요약:**

```
Elementwise: 높은 병렬성, 분산된 메모리 접근
Tiled:       적당한 병렬성, 지역화된 메모리 접근
```

Elementwise와 tiled 패턴 간의 선택은 특정 워크로드 특성, 데이터 접근 패턴, 대상 하드웨어 능력에 따라 달라집니다.

</div>
</details>

## 다음 단계

Elementwise와 tiled 패턴을 모두 이해했다면:

- **[Vectorize - SIMD 제어](./vectorize.md)**: SIMD 연산에 대한 세밀한 제어
- **[🧠 GPU 스레딩 vs SIMD 개념](./gpu-thread-vs-simd.md)**: 실행 계층 구조 이해
- **[📊 Mojo 벤치마킹](./benchmarking.md)**: 성능 분석과 최적화

💡 **핵심 요약**: Tiling은 메모리 접근 패턴이 원시 연산 처리량보다 더 중요할 수 있음을 보여줍니다. 최고의 GPU 코드는 병렬성과 메모리 계층 구조 최적화의 균형을 맞춥니다.
