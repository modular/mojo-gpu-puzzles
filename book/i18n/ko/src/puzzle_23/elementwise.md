<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# Elementwise - 기본 GPU 함수형 연산

이 퍼즐은 Mojo의 함수형 `elementwise` 패턴을 사용하여 벡터 덧셈을 구현합니다. 각 스레드가 자동으로 여러 SIMD 요소를 처리하며, 현대 GPU 프로그래밍이 어떻게 저수준 세부 사항을 추상화하면서도 높은 성능을 유지하는지 보여줍니다.

**핵심 통찰:** _[elementwise](https://docs.modular.com/mojo/std/algorithm/functional/elementwise/) 함수는 스레드 관리, SIMD 벡터화, 메모리 병합을 자동으로 처리합니다._

## 핵심 개념

이 퍼즐에서 다루는 내용:

- `elementwise`를 활용한 **함수형 GPU 프로그래밍**
- GPU 스레드 내의 **자동 SIMD 벡터화**
- 안전한 메모리 접근을 위한 **LayoutTensor 연산**
- **GPU 스레드 계층 구조** vs SIMD 연산
- 중첩 함수에서의 **캡처 의미론**

수학적 연산은 단순한 요소별 덧셈입니다:
\\[\Large \text{output}[i] = a[i] + b[i]\\]

이 구현은 Mojo에서의 모든 GPU 함수형 프로그래밍에 적용할 수 있는 기본 패턴을 다룹니다.

## 설정

- 벡터 크기: `SIZE = 1024`
- 데이터 타입: `DType.float32`
- SIMD 폭: 타겟 의존적 (GPU 아키텍처와 데이터 타입에 따라 결정)
- 레이아웃: `Layout.row_major(SIZE)` (1D row-major)

## 완성할 코드

```mojo
{{#include ../../../../../problems/p23/p23.mojo:elementwise_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p23/p23.mojo" class="filename">전체 파일 보기: problems/p23/p23.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **함수 구조 이해하기**

`elementwise` 함수는 다음과 같은 정확한 시그니처를 가진 중첩 함수를 기대합니다:

```mojo
@parameter
@always_inline
fn your_function[simd_width: Int, rank: Int](indices: IndexList[rank]) capturing -> None:
    # 구현 코드
```

**각 부분이 중요한 이유:**

- `@parameter`: 최적의 GPU 코드 생성을 위한 컴파일 타임 특수화를 활성화합니다
- `@always_inline`: GPU 커널에서 함수 호출 오버헤드를 제거하기 위해 인라이닝을 강제합니다
- `capturing`: 외부 스코프의 변수(입출력 텐서)에 접근할 수 있게 합니다
- `IndexList[rank]`: 다차원 인덱싱을 제공합니다 (벡터는 rank=1, 행렬은 rank=2)

### 2. **인덱스 추출과 SIMD 처리**

```mojo
idx = indices[0]  # 1D 연산을 위한 선형 인덱스 추출
```

이 `idx`는 단일 요소가 아닌 SIMD 벡터의 **시작 위치**를 나타냅니다. `SIMD_WIDTH=4` (GPU 의존적)인 경우:

- Thread 0은 `idx=0`부터 시작하여 요소 `[0, 1, 2, 3]`을 처리
- Thread 1은 `idx=4`부터 시작하여 요소 `[4, 5, 6, 7]`을 처리
- Thread 2는 `idx=8`부터 시작하여 요소 `[8, 9, 10, 11]`을 처리
- 이런 식으로 계속...

### 3. **SIMD 로드 패턴**

```mojo
a_simd = a.aligned_load[simd_width](Index(idx))  # 연속 float 4개 로드 (GPU 의존적)
b_simd = b.aligned_load[simd_width](Index(idx))  # 연속 float 4개 로드 (GPU 의존적)
```

두 번째 매개변수 `0`은 차원 offset입니다 (1D 벡터에서는 항상 0). 이 연산은 한 번에 **벡터화된 chunk**의 데이터를 로드합니다. 로드되는 정확한 요소 수는 GPU의 SIMD 능력에 따라 달라집니다.

### 4. **벡터 연산**

```mojo
result = a_simd + b_simd  # 4개 요소의 SIMD 덧셈을 동시에 수행 (GPU 의존적)
```

전체 SIMD 벡터에 걸쳐 요소별 덧셈을 병렬로 수행합니다 - 4개의 개별 스칼라 덧셈보다 훨씬 빠릅니다.

### 5. **SIMD 저장**

```mojo
output.store[simd_width](idx, 0, result)  # 4개 결과를 한 번에 저장 (GPU 의존적)
```

전체 SIMD 벡터를 한 번의 연산으로 메모리에 다시 기록합니다.

### 6. **elementwise 함수 호출**

```mojo
elementwise[your_function, SIMD_WIDTH, target="gpu"](total_size, ctx)
```

- `total_size`는 모든 요소를 처리하기 위해 `a.size()`로 설정해야 합니다
- GPU는 실행할 스레드 수를 자동으로 결정합니다: `total_size // SIMD_WIDTH`

### 7. **디버깅 핵심 포인트**

템플릿에 있는 `print("idx:", idx)`에 주목하세요. 실행하면 다음과 같이 출력됩니다:

```
idx: 0, idx: 4, idx: 8, idx: 12, ...
```

각 스레드가 서로 다른 SIMD chunk를 처리하며, `SIMD_WIDTH` (GPU 의존적) 간격으로 자동 배치됨을 보여줍니다.

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
pixi run p23 --elementwise
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p23 --elementwise
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p23 --elementwise
```

  </div>
  <div class="tab-content">

```bash
uv run poe p23 --elementwise
```

  </div>
</div>

퍼즐이 아직 풀리지 않은 경우 다음과 같이 출력됩니다:

```txt
SIZE: 1024
simd_width: 4
...
idx: 404
idx: 408
idx: 412
idx: 416
...

out: HostBuffer([0.0, 0.0, 0.0, ..., 0.0, 0.0, 0.0])
expected: HostBuffer([1.0, 5.0, 9.0, ..., 4085.0, 4089.0, 4093.0])
```

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p23/p23.mojo:elementwise_add_solution}}
```

<div class="solution-explanation">

Mojo의 elementwise 함수형 패턴은 현대 GPU 프로그래밍을 위한 몇 가지 기본 개념을 소개합니다:

### 1. **함수형 추상화 철학**

`elementwise` 함수는 기존 GPU 프로그래밍에서의 패러다임 전환을 나타냅니다:

**전통적인 CUDA/HIP 방식:**

```mojo
# 수동 스레드 관리
idx = thread_idx.x + block_idx.x * block_dim.x
if idx < size:
    output[idx] = a[idx] + b[idx];  // 스칼라 연산
```

**Mojo 함수형 방식:**

```mojo
# 자동 관리 + SIMD 벡터화
elementwise[add_function, simd_width, target="gpu"](size, ctx)
```

**`elementwise`가 추상화하는 것들:**

- **스레드 그리드 구성**: 블록/그리드 차원을 계산할 필요 없음
- **경계 검사**: 배열 경계를 자동으로 처리
- **메모리 병합**: 최적의 메모리 접근 패턴이 내장
- **SIMD 오케스트레이션**: 벡터화를 투명하게 처리
- **GPU 타겟 선택**: 다양한 GPU 아키텍처에서 동작

### 2. **심층 분석: 중첩 함수 아키텍처**

```mojo
@parameter
@always_inline
fn add[simd_width: Int, rank: Int](indices: IndexList[rank]) capturing -> None:
```

**매개변수 분석:**

- **`@parameter`**: 이 데코레이터는 **컴파일 타임 특수화**를 제공합니다. 각 고유한 `simd_width`와 `rank`에 대해 함수가 별도로 생성되어 적극적인 최적화가 가능합니다.
- **`@always_inline`**: GPU 성능에 매우 중요합니다 - 코드를 커널에 직접 내장하여 함수 호출 오버헤드를 제거합니다.
- **`capturing`**: **렉시컬 스코핑**을 활성화합니다 - 내부 함수가 명시적 매개변수 전달 없이 외부 스코프의 변수에 접근할 수 있습니다.
- **`IndexList[rank]`**: **차원 무관 인덱싱**을 제공합니다 - 동일한 패턴이 1D 벡터, 2D 행렬, 3D 텐서 등에서 작동합니다.

### 3. **SIMD 실행 모델 심층 분석**

```mojo
idx = indices[0]                                    # 선형 인덱스: 0, 4, 8, 12... (GPU 의존적 간격)
a_simd = a.aligned_load[simd_width](Index(idx))     # 로드: [a[0:4], a[4:8], a[8:12]...] (로드당 4개 요소)
b_simd = b.aligned_load[simd_width](Index(idx))     # 로드: [b[0:4], b[4:8], b[8:12]...] (로드당 4개 요소)
ret = a_simd + b_simd                               # SIMD: 4개 덧셈을 병렬 수행 (GPU 의존적)
output.store[simd_width](Index(global_start), ret)  # 저장: 4개 결과를 동시 저장 (GPU 의존적)
```

**실행 계층 구조 시각화:**

```
GPU 아키텍처:
├── Grid (전체 문제)
│   ├── Block 1 (여러 Warp)
│   │   ├── Warp 1 (32개 스레드) --> Warp는 다음 Part VI에서 학습
│   │   │   ├── Thread 1 → SIMD[4개 요소]  ← 현재 초점 (GPU 의존적 폭)
│   │   │   ├── Thread 2 → SIMD[4개 요소]
│   │   │   └── ...
│   │   └── Warp 2 (32개 스레드)
│   └── Block 2 (여러 Warp)
```

**SIMD_WIDTH=4인 1024개 요소 벡터의 경우 (GPU 예시):**

- **필요한 총 SIMD 연산 수**: 1024 ÷ 4 = 256
- **GPU 실행**: 256개 스레드 (1024 ÷ 4)
- **각 스레드가 처리하는 양**: 정확히 4개의 연속 요소
- **메모리 대역폭**: 스칼라 연산 대비 SIMD_WIDTH배 향상

**참고**: SIMD 폭은 GPU 아키텍처에 따라 다릅니다 (예: 일부 GPU는 4, RTX 4090은 8, A100은 16).

### 4. **메모리 접근 패턴 분석**

```mojo
a.aligned_load[simd_width](Index(idx))  // 병합 메모리 접근
```

**메모리 coalescing의 이점:**

- **순차적 접근**: 스레드들이 연속적인 메모리 위치에 접근
- **캐시 최적화**: L1/L2 캐시 히트율 극대화
- **대역폭 활용**: 이론적 메모리 대역폭에 근접하는 성능 달성
- **하드웨어 효율**: GPU 메모리 컨트롤러가 이 패턴에 최적화되어 있음

**SIMD_WIDTH=4 (GPU 의존적) 예시:**

```
Thread 0: a[0:4] 로드   → 메모리 뱅크 0-3
Thread 1: a[4:8] 로드   → 메모리 뱅크 4-7
Thread 2: a[8:12] 로드  → 메모리 뱅크 8-11
...
결과: 최적의 메모리 컨트롤러 활용
```

### 5. **성능 특성 및 최적화**

**산술 강도 분석 (SIMD_WIDTH=4 기준):**

- **산술 연산**: 4개 요소당 1회 SIMD 덧셈
- **메모리 연산**: 4개 요소당 2회 SIMD 로드 + 1회 SIMD 저장
- **산술 강도**: 1 덧셈 ÷ 3 메모리 연산 = 0.33 (메모리 바운드)

**이것이 메모리 바운드인 이유:**

```
단순 연산에서는 메모리 대역폭 >>> 연산 능력
```

**최적화 시사점:**

- 산술 최적화보다 메모리 접근 패턴에 집중해야 함
- SIMD 벡터화가 주요 성능 이점을 제공
- 메모리 병합이 성능에 매우 중요
- 연산 복잡도보다 캐시 지역성이 더 중요

### 6. **확장성과 적응성**

**자동 하드웨어 적응:**

```mojo
comptime SIMD_WIDTH = simd_width_of[dtype, target = _get_gpu_target()]()
```

- **GPU별 최적화**: SIMD 폭이 하드웨어에 맞게 조정됨 (예: 일부 카드는 4, RTX 4090은 8, A100은 16)
- **데이터 타입 인식**: float32와 float16에 대해 서로 다른 SIMD 폭 적용
- **컴파일 타임 최적화**: 하드웨어 감지에 대한 런타임 오버헤드 없음

**확장성 특성:**

- **스레드 수**: 문제 크기에 따라 자동 확장
- **메모리 사용량**: 입력 크기에 비례하여 선형 확장
- **성능**: 메모리 대역폭 포화 시점까지 거의 선형적인 속도 향상

### 7. **고급 인사이트: 이 패턴이 중요한 이유**

**복잡한 연산의 기초:**
이 elementwise 패턴은 다음 연산들의 기반이 됩니다:

- **Reduction 연산**: 대규모 배열에서의 합계, 최댓값, 최솟값
- **Broadcast 연산**: 스칼라-벡터 연산
- **복잡한 변환**: 활성화 함수, 정규화
- **다차원 연산**: 행렬 연산, convolution

**전통적인 방식과의 비교:**

```mojo
// 전통적: 오류 발생 가능, 장황함, 하드웨어 종속적
__global__ void add_kernel(float* output, float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];  // 벡터화 없음
    }
}

// Mojo: 안전, 간결, 자동 벡터화
elementwise[add, SIMD_WIDTH, target="gpu"](size, ctx)
```

**함수형 접근법의 이점:**

- **안전성**: 자동 경계 검사로 버퍼 오버플로우 방지
- **이식성**: 동일한 코드가 다양한 GPU 벤더/세대에서 동작
- **성능**: 컴파일러 최적화가 수동 튜닝 코드를 종종 능가
- **유지보수성**: 깔끔한 추상화로 디버깅 복잡도 감소
- **조합성**: 다른 함수형 연산과 쉽게 결합 가능

이 패턴은 GPU 프로그래밍의 미래를 나타냅니다 - 성능을 희생하지 않는 고수준 추상화로, 최적의 효율성을 유지하면서 GPU 컴퓨팅을 더 쉽게 접근할 수 있게 합니다.

</div>
</details>

## 다음 단계

Elementwise 연산을 학습했다면 다음으로 넘어갈 준비가 되었습니다:

- **[Tile - 메모리 효율적인 Tiled 처리](./tile.md)**: 메모리 효율적인 tiled 처리 패턴
- **[Vectorize - SIMD 제어](./vectorize.md)**: 세밀한 SIMD 제어
- **[🧠 GPU 스레딩 vs SIMD 개념](./gpu-thread-vs-simd.md)**: 실행 계층 구조 이해
- **[📊 Mojo 벤치마킹](./benchmarking.md)**: 성능 분석과 최적화

💡 **핵심 요약**: `elementwise` 패턴은 Mojo가 함수형 프로그래밍의 우아함과 GPU 성능을 어떻게 결합하는지 보여줍니다. 연산에 대한 완전한 제어를 유지하면서 벡터화와 스레드 관리를 자동으로 처리합니다.
