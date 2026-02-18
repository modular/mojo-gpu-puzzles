<!-- i18n-source-commit: 43fce1182f8029e7edc50157aed0e6ebb8129d42 -->

# block.broadcast()와 벡터 정규화

[block.sum](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum)과 [block.broadcast](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/broadcast) 연산을 결합하여 벡터 평균 정규화를 구현하고, 블록 레벨 통신 워크플로우의 전체 모습을 보여줍니다. 각 스레드가 평균 계산에 기여한 다음, broadcast된 평균을 받아 자신의 요소를 정규화하여, 블록 연산이 실제 병렬 알고리즘을 해결하기 위해 어떻게 함께 동작하는지 보여줍니다.

**핵심 통찰:** _[block.broadcast()](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/broadcast) 연산은 하나→전체 통신을 가능하게 하여, 기본 블록 통신 패턴을 완성합니다: reduction(전체→하나), scan(전체→각각), broadcast(하나→전체)._

## 핵심 개념

이 퍼즐에서 배울 내용:

- `block.broadcast()`를 활용한 **블록 레벨 broadcast**
- **하나→전체 통신** 패턴
- **소스 스레드 지정**과 파라미터 제어
- 여러 연산을 결합하는 **완전한 블록 연산 워크플로우**
- 조율된 블록 기본 요소를 사용한 **실제 알고리즘 구현**

이 알고리즘은 벡터 평균 정규화를 보여줍니다:
\\[\Large \text{output}[i] = \frac{\text{input}[i]}{\frac{1}{N}\sum_{j=0}^{N-1} \text{input}[j]}\\]

각 스레드가 평균 계산에 기여한 다음, broadcast된 평균을 받아 자신의 요소를 정규화합니다.

## 구성

- 벡터 크기: `SIZE = 128` 요소
- 데이터 타입: `DType.float32`
- 블록 구성: `(128, 1)` 블록당 스레드 수 (`TPB = 128`)
- 그리드 구성: `(1, 1)` 그리드당 블록 수
- 레이아웃: `Layout.row_major(SIZE)` (입력과 출력 모두 1D row-major)
- 테스트 데이터: 1-8 반복 값, 평균 = 4.5
- 예상 출력: 평균이 1.0인 정규화된 벡터

## 도전 과제: 블록 전체 계산과 분배의 조율

기존의 평균 정규화 방식은 복잡한 조율이 필요합니다:

```python
# 순차적 방식 - 병렬성을 활용하지 못함
total = sum(input_array)
mean = total / len(input_array)
output_array = [x / mean for x in input_array]
```

**단순한 GPU 병렬화의 문제점:**

- **다중 kernel 실행**: 평균 계산과 정규화에 각각 별도의 패스가 필요
- **글로벌 메모리 왕복**: 평균을 글로벌 메모리에 저장했다가 나중에 다시 읽기
- **동기화 복잡성**: 계산 단계 간에 barrier가 필요
- **스레드 분기**: 서로 다른 스레드가 서로 다른 작업을 수행

**기존 GPU 풀이의 복잡성:**

```mojo
# 1단계: 합계를 구하기 위한 reduction (복잡한 공유 메모리 + barrier)
shared_sum[local_i] = my_value
barrier()
# 여러 barrier() 호출이 필요한 수동 트리 reduction...

# 2단계: 스레드 0이 평균을 계산
if local_i == 0:
    mean = shared_sum[0] / size
    shared_mean[0] = mean

barrier()

# 3단계: 모든 스레드가 평균을 읽고 정규화
mean = shared_mean[0]  # 모두가 같은 값을 읽음
output[global_i] = my_value / mean
```

## 고급 방식: `block.sum()` + `block.broadcast()` 조율

다단계 조율을 간결한 블록 연산 워크플로우로 변환합니다:

## 작성할 코드

### 완전한 블록 연산 워크플로우

블록 연산 도구 모음 전체를 사용하여 고급 벡터 평균 정규화를 구현합니다:

```mojo
{{#include ../../../../../problems/p27/p27.mojo:block_normalize}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p27/p27.mojo" class="filename">전체 파일 보기: problems/p27/p27.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

### 1. **완전한 워크플로우 구조 (모든 이전 연산을 기반으로 구축)**

알고리즘은 완벽한 블록 연산 패턴을 따릅니다:

1. 각 스레드가 자신의 요소를 로드 (모든 이전 퍼즐에서 익숙한 패턴)
2. `block.sum()`으로 합계를 계산 (이 퍼즐의 앞부분에서 배운 내용)
3. 스레드 0이 합계로부터 평균을 계산
4. `block.broadcast()`로 평균을 모든 스레드에 공유 (새로운 내용!)
5. 각 스레드가 broadcast된 평균으로 정규화

### 2. **데이터 로딩과 합계 계산 (익숙한 패턴)**

기존 LayoutTensor 패턴으로 요소를 로드합니다:

```mojo
var my_value: Scalar[dtype] = 0.0
if global_i < size:
    my_value = input_data[global_i][0]  # SIMD 추출
```

그런 다음 앞서 배운 내적과 동일하게 `block.sum()`을 사용합니다:

```mojo
total_sum = block.sum[block_size=tpb, broadcast=False](...)
```

### 3. **평균 계산 (스레드 0만)**

스레드 0만 평균을 계산해야 합니다:

```mojo
var mean_value: Scalar[dtype] = 1.0  # 안전한 기본값
if local_i == 0:
    # total_sum과 size로 평균 계산
```

**왜 스레드 0인가?** `block.sum()` 패턴에서 스레드 0이 결과를 받는 것과 일관성을 유지합니다.

### 4. **[block.broadcast()](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/broadcast) API 개념**

함수 시그니처를 살펴보세요 - 다음이 필요합니다:

- 템플릿 파라미터: `dtype`, `width`, `block_size`
- 런타임 파라미터: `val` (broadcast할 SIMD 값), `src_thread` (기본값=0)

호출 패턴은 기존 템플릿 스타일을 따릅니다:

```mojo
result = block.broadcast[
    dtype = DType.float32,
    width = 1,
    block_size = tpb
](val=SIMD[DType.float32, 1](value_to_broadcast), src_thread=UInt(0))
```

### 5. **Broadcast 패턴 이해하기**

**핵심 통찰**: `block.broadcast()`는 하나의 스레드에서 값을 가져와 모든 스레드에 전달합니다:

- **스레드 0**이 계산된 평균값을 가지고 있음
- **모든 스레드**가 같은 평균값이 필요
- **`block.broadcast()`**가 스레드 0의 값을 모두에게 복사

이것은 `block.sum()`(전체→하나)의 반대이며, `block.prefix_sum()`(전체→각각 위치)과도 다릅니다.

### 6. **최종 정규화 단계**

모든 스레드가 broadcast된 평균을 받으면, 자신의 요소를 정규화합니다:

```mojo
if global_i < size:
    normalized_value = my_value / broadcasted_mean[0]  # SIMD 추출
    output_data[global_i] = normalized_value
```

**SIMD 추출**: `block.broadcast()`가 SIMD를 반환하므로 `[0]`으로 스칼라를 추출해야 합니다.

### 7. **이전 퍼즐에서의 패턴 인식**

- **스레드 인덱싱**: 항상 동일한 `global_i`, `local_i` 패턴
- **경계 검사**: 동일한 `if global_i < size` 검증
- **SIMD 처리**: 동일한 `[0]` 추출 패턴
- **블록 연산**: `block.sum()`과 동일한 템플릿 파라미터 스타일

각 블록 연산이 일관된 패턴을 따르는 것이 핵심입니다!

</div>
</details>

**block.broadcast() 방식 테스트:**
<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p27 --normalize
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p27 --normalize
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p27 --normalize
```

  </div>
  <div class="tab-content">

```bash
uv run poe p27 --normalize
```

  </div>
</div>

풀었을 때의 예상 출력:

```txt
SIZE: 128
TPB: 128

Input sample: 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 ...
Sum value: 576.0
Mean value: 4.5

Mean Normalization Results:
Normalized sample: 0.22222222 0.44444445 0.6666667 0.8888889 1.1111112 1.3333334 1.5555556 1.7777778 ...

Output sum: 128.0
Output mean: 1.0
✅ Success: Output mean is 1.0 (should be close to 1.0)
```

## 풀이

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p27/p27.mojo:block_normalize_solution}}
```

<div class="solution-explanation">

`block.broadcast()` 커널은 세 가지 기본 통신 패턴을 모두 결합하여 수학적으로 검증 가능한 결과를 생성하는 실제 알고리즘으로 완전한 블록 연산 워크플로우를 보여줍니다:

## **구체적인 실행을 통한 완전한 알고리즘 분석:**

### **1단계: 병렬 데이터 로딩 (모든 이전 퍼즐에서 확립된 패턴)**

```
스레드 인덱싱 (모든 퍼즐에서 일관됨):
  global_i = block_dim.x * block_idx.x + thread_idx.x  // 입력 배열 위치에 매핑
  local_i = thread_idx.x                              // 블록 내 위치 (0-127)

LayoutTensor 패턴을 사용한 병렬 요소 로딩:
  스레드 0:   my_value = input_data[0][0] = 1.0    // 첫 번째 순환 값
  스레드 1:   my_value = input_data[1][0] = 2.0    // 두 번째 순환 값
  스레드 7:   my_value = input_data[7][0] = 8.0    // 마지막 순환 값
  스레드 8:   my_value = input_data[8][0] = 1.0    // 순환 반복: 1,2,3,4,5,6,7,8,1,2...
  스레드 15:  my_value = input_data[15][0] = 8.0   // 15 % 8 = 7, 8번째 값
  스레드 127: my_value = input_data[127][0] = 8.0  // 127 % 8 = 7, 8번째 값

128개 스레드가 동시에 로드 - 완벽한 병렬 효율!
```

### **2단계: 블록 전체 합계 reduction (앞서 배운 block.sum() 지식 활용)**

```
128개 스레드에 걸친 block.sum() 조율:
  기여분 분석:
    - 값 1,2,3,4,5,6,7,8이 각각 16번 반복 (128/8 = 16)
    - 스레드 기여분: 16×1 + 16×2 + 16×3 + 16×4 + 16×5 + 16×6 + 16×7 + 16×8
    - 수학적 합계: 16 × (1+2+3+4+5+6+7+8) = 16 × 36 = 576.0

block.sum() 하드웨어 실행:
  모든 스레드 → [reduction 트리] → 스레드 0
  total_sum = SIMD[DType.float32, 1](576.0)  // 스레드 0만 이 값을 수신

스레드 1-127: total_sum에 접근 불가 (block.sum에서 broadcast=False)
```

### **3단계: 독점적 평균 계산 (단일 스레드 처리)**

```
스레드 0이 핵심 계산을 수행:
  입력: total_sum[0] = 576.0, size = 128
  계산: mean_value = 576.0 / 128.0 = 4.5

  검증: 기대 평균 = (1+2+3+4+5+6+7+8)/8 = 36/8 = 4.5 ✓

다른 모든 스레드 (1-127):
  mean_value = 1.0 (기본 안전 값)
  이 값들은 무관 - broadcast로 덮어씌워질 예정

핵심 통찰: 이 시점에서 올바른 평균값을 가진 것은 스레드 0뿐입니다!
```

### **4단계: 블록 전체 broadcast 분배 (하나 → 전체 통신)**

```
block.broadcast() API 실행:
  소스: src_thread = UInt(0) → 스레드 0의 mean_value = 4.5
  대상: 블록 내 모든 128 스레드

broadcast 전:
  스레드 0:   mean_value = 4.5  ← 진실의 원천
  스레드 1:   mean_value = 1.0  ← 덮어씌워질 예정
  스레드 2:   mean_value = 1.0  ← 덮어씌워질 예정
  ...
  스레드 127: mean_value = 1.0  ← 덮어씌워질 예정

block.broadcast() 실행 후:
  스레드 0:   broadcasted_mean[0] = 4.5  ← 자신의 값을 다시 수신
  스레드 1:   broadcasted_mean[0] = 4.5  ← 이제 올바른 값을 가짐!
  스레드 2:   broadcasted_mean[0] = 4.5  ← 이제 올바른 값을 가짐!
  ...
  스레드 127: broadcasted_mean[0] = 4.5  ← 이제 올바른 값을 가짐!

결과: 완벽한 동기화 - 모든 스레드가 동일한 평균값을 가짐!
```

### **5단계: 병렬 평균 정규화 (조율된 처리)**

```
각 스레드가 broadcast된 평균을 사용하여 독립적으로 정규화:
  스레드 0:   normalized = 1.0 / 4.5 = 0.22222222...
  스레드 1:   normalized = 2.0 / 4.5 = 0.44444444...
  스레드 2:   normalized = 3.0 / 4.5 = 0.66666666...
  스레드 7:   normalized = 8.0 / 4.5 = 1.77777777...
  스레드 8:   normalized = 1.0 / 4.5 = 0.22222222...  (패턴 반복)
  ...

수학적 검증:
  출력 합계 = (0.222... + 0.444... + ... + 1.777...) × 16 = 4.5 × 16 × 2 = 128.0
  출력 평균 = 128.0 / 128 = 1.0  완벽한 정규화!

각 값을 원래 평균으로 나누면 평균이 1.0인 출력을 생성
```

### **6단계: 정확성 검증**

```
입력 분석:
  - 합계: 576.0, 평균: 4.5
  - 최댓값: 8.0, 최솟값: 1.0
  - 범위: [1.0, 8.0]

출력 분석:
  - 합계: 128.0, 평균: 1.0 ✓
  - 최댓값: 1.777..., 최솟값: 0.222...
  - 범위: [0.222, 1.777] (모든 값이 1/4.5 비율로 스케일링)

비례 관계 보존:
  - 원래 8:1 비율이 1.777:0.222 = 8:1로 유지 ✓
  - 모든 상대적 크기가 완벽하게 유지
```

## **이 완전한 워크플로우가 수학적·계산적으로 우수한 이유:**

### **기술적 정확성과 검증:**

```
수학적 정확성 증명:
  입력: x₁, x₂, ..., xₙ (n = 128)
  평균: μ = (∑xᵢ)/n = 576/128 = 4.5

  정규화: yᵢ = xᵢ/μ
  출력 평균: (∑yᵢ)/n = (∑xᵢ/μ)/n = (1/μ)(∑xᵢ)/n = (1/μ)μ = 1 ✓

알고리즘이 증명 가능하게 올바른 수학적 결과를 생성합니다.
```

### **[Puzzle 12](../puzzle_12/layout_tensor.md) (기초 패턴)과의 연결:**

- **스레드 조율의 진화**: 동일한 `global_i`, `local_i` 패턴이지만 블록 기본 요소 사용
- **메모리 접근 패턴**: 동일한 LayoutTensor SIMD 추출 `[0]`이지만 최적화된 워크플로우
- **복잡성 제거**: 20줄 이상의 수동 barrier를 2개의 블록 연산으로 대체
- **교육적 진행**: 수동 → 자동, 복잡 → 단순, 오류 발생 가능 → 신뢰성

### **[`block.sum()`](./block_sum.md) (완벽한 통합)과의 연결:**

- **API 일관성**: 동일한 템플릿 구조 `[block_size=tpb, broadcast=False]`
- **결과 흐름 설계**: 스레드 0이 합계를 수신하고, 자연스럽게 파생 파라미터를 계산
- **매끄러운 조합**: `block.sum()`의 출력이 계산 + broadcast의 입력이 됨
- **성능 최적화**: 단일 kernel 워크플로우 vs 다중 패스 방식

### **[`block.prefix_sum()`](./block_prefix_sum.md) (상보적 통신)과의 연결:**

- **분배 패턴**: `prefix_sum`은 고유한 위치를, `broadcast`는 공유 값을 제공
- **사용 시나리오**: `prefix_sum`은 병렬 파티셔닝용, `broadcast`는 매개변수 공유용
- **템플릿 일관성**: 모든 연산에서 동일한 `dtype`, `block_size` 파라미터 패턴
- **SIMD 처리 통일성**: 모든 블록 연산이 `[0]` 추출이 필요한 SIMD를 반환

### **고급 알고리즘 인사이트:**

```
통신 패턴 비교:
  기존 방식:
    1. 수동 reduction:      O(log n), 명시적 barrier 필요
    2. 공유 메모리 쓰기:    O(1), 동기화 필요
    3. 공유 메모리 읽기:    O(1), 뱅크 충돌 가능성
    총합: 다수의 동기화 지점, 오류 발생 가능

  블록 연산 방식:
    1. block.sum():          O(log n), 하드웨어 최적화, 자동 barrier
    2. 계산:                O(1), 단일 스레드
    3. block.broadcast():    O(log n), 하드웨어 최적화, 자동 분배
    총합: 두 개의 기본 요소, 자동 동기화, 증명된 정확성
```

### **실제 응용 알고리즘 패턴:**

```
일반적인 병렬 알고리즘 구조:
  1단계: 병렬 데이터 처리        → 모든 스레드가 기여
  2단계: 전역 파라미터 계산      → 하나의 스레드가 계산
  3단계: 파라미터 분배           → 모든 스레드가 수신
  4단계: 조율된 병렬 출력        → 모든 스레드가 처리

이 정확한 패턴이 등장하는 분야:
  - 배치 정규화 (딥러닝)
  - 히스토그램 균등화 (이미지 처리)
  - 반복적 수치 해법 (과학 연산)
  - 조명 계산 (컴퓨터 그래픽)

평균 정규화는 이 근본적인 패턴의 완벽한 교육 사례입니다.
```

## **블록 연산 3부작 완성:**

### **1. `block.sum()` - 전체→하나 (Reduction)**

- **입력**: 모든 스레드가 값을 제공
- **출력**: 스레드 0이 집계된 결과를 수신
- **용도**: 합계, 최댓값 계산 등

### **2. `block.prefix_sum()` - 전체→각각 (Scan)**

- **입력**: 모든 스레드가 값을 제공
- **출력**: 각 스레드가 누적 위치를 수신
- **용도**: 쓰기 위치 계산, 병렬 파티셔닝

### **3. `block.broadcast()` - 하나→전체 (Broadcast)**

- **입력**: 하나의 스레드가 값을 제공 (일반적으로 스레드 0)
- **출력**: 모든 스레드가 같은 값을 수신
- **용도**: 계산된 매개변수 공유, 설정값 분배

</div>
</details>

**완전한 블록 연산 진행:**

1. **수동 조율** ([Puzzle 12](../puzzle_12/layout_tensor.md)): 병렬 기초 이해
2. **Warp 기본 요소** ([Puzzle 24](../puzzle_24/warp_sum.md)): 하드웨어 가속 패턴 학습
3. **블록 reduction** ([`block.sum()`](./block_sum.md)): 전체→하나 통신 학습
4. **블록 scan** ([`block.prefix_sum()`](./block_prefix_sum.md)): 전체→각각 통신 학습
5. **블록 broadcast** (`block.broadcast()`): 하나→전체 통신 학습

**전체 그림:** 블록 연산은 고급 병렬 알고리즘을 위한 기본 통신 빌딩 블록을 제공하며, 복잡한 수동 조율을 깔끔하고 조합 가능한 기본 요소로 대체합니다.

## 성능 인사이트와 기술 분석

### **정량적 성능 비교:**

**`block.broadcast()` vs 기존 공유 메모리 방식 (참고용):**

**기존 수동 방식:**

```
1단계: 수동 reduction
  • 공유 메모리 할당: ~5 사이클
  • Barrier 동기화: ~10 사이클
  • 트리 reduction 루프: ~15 사이클
  • 오류 발생 가능한 수동 인덱싱

2단계: 평균 계산: ~2 사이클

3단계: 공유 메모리 broadcast
  • 공유 메모리에 수동 쓰기: ~2 사이클
  • Barrier 동기화: ~10 사이클
  • 모든 스레드 읽기: ~3 사이클

총합: ~47 사이클
  + 동기화 오버헤드
  + 경쟁 상태 가능성
  + 수동 오류 디버깅
```

**블록 연산 방식:**

```
1단계: block.sum()
  • 하드웨어 최적화: ~3 사이클
  • 자동 barrier: 명시적 비용 0
  • 최적화된 reduction: ~8 사이클
  • 검증된 올바른 구현

2단계: 평균 계산: ~2 사이클

3단계: block.broadcast()
  • 하드웨어 최적화: ~4 사이클
  • 자동 분배: 명시적 비용 0
  • 검증된 올바른 구현

총합: ~17 사이클
  + 자동 최적화
  + 보장된 정확성
  + 조합 가능한 설계
```

### **메모리 계층 구조 이점:**

**캐시 효율:**

- **block.sum()**: 최적화된 메모리 접근 패턴으로 캐시 미스 감소
- **block.broadcast()**: 효율적인 분배로 메모리 대역폭 사용 최소화
- **결합 워크플로우**: 단일 kernel이 글로벌 메모리 왕복을 100% 감소

**메모리 대역폭 활용:**

```
기존 멀티 kernel 방식:
  Kernel 1: 입력 → Reduction → 글로벌 메모리 쓰기
  Kernel 2: 글로벌 메모리 읽기 → Broadcast → 출력
  총 글로벌 메모리 전송: 배열 크기의 3배

블록 연산 단일 kernel:
  입력 → block.sum() → block.broadcast() → 출력
  총 글로벌 메모리 전송: 배열 크기의 2배 (33% 개선)
```

### **각 블록 연산의 최적 사용 시나리오:**

**`block.sum()` 최적 시나리오:**

- **데이터 집계**: 합계, 평균, 최댓값/최솟값 계산
- **Reduction 패턴**: 전체→하나 통신이 필요한 모든 경우
- **통계 연산**: 평균, 분산, 상관관계 계산

**`block.prefix_sum()` 최적 시나리오:**

- **병렬 파티셔닝**: stream compaction, 히스토그램 구간 분류
- **쓰기 위치 계산**: 병렬 출력 생성
- **병렬 알고리즘**: 정렬, 검색, 데이터 재구성

**`block.broadcast()` 최적 시나리오:**

- **매개변수 분배**: 계산된 값을 모든 스레드에 공유
- **설정 전파**: 모드 플래그, 스케일링 팩터, 임계값
- **조율된 처리**: 모든 스레드가 동일한 계산된 매개변수가 필요할 때

### **조합의 이점:**

```
개별 연산:   좋은 성능, 제한된 범위
결합 연산:   탁월한 성능, 포괄적인 알고리즘

실제 응용에서 볼 수 있는 조합 예시:
• block.sum() + block.broadcast():       정규화 알고리즘
• block.prefix_sum() + block.sum():      고급 파티셔닝
• 세 가지 모두 결합:                      복잡한 병렬 알고리즘
• 기존 패턴과 함께:                       하이브리드 최적화 전략
```

## 다음 단계

완전한 블록 연산 3부작을 배웠으니, 다음으로 진행할 수 있습니다:

- **멀티 블록 알고리즘**: 여러 스레드 블록에 걸친 연산 조율
- **고급 병렬 패턴**: 복잡한 알고리즘을 위한 블록 연산 결합
- **메모리 계층 구조 최적화**: 효율적인 데이터 이동 패턴
- **알고리즘 설계**: 블록 연산 빌딩 블록을 사용한 병렬 알고리즘 구조화
- **성능 최적화**: 최적의 블록 크기와 연산 조합 선택

💡 **핵심 요점**: 블록 연산 3부작(`sum`, `prefix_sum`, `broadcast`)은 블록 레벨 병렬 프로그래밍을 위한 완전한 통신 기본 요소를 제공합니다. 이 연산들을 조합하면 GPU 하드웨어 최적화를 활용하는 깔끔하고 유지보수하기 쉬운 코드로 고급 병렬 알고리즘을 구현할 수 있습니다. 평균 정규화는 이 연산들이 함께 작동하여 실제 연산 문제를 효율적으로 해결하는 방법을 보여줍니다.
