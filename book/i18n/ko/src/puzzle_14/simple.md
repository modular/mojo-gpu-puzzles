<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

# 기본 버전

1D LayoutTensor `a`에 대해 prefix sum을 계산하고 결과를 1D LayoutTensor `output`에 저장하는 kernel을 구현하세요.

**참고:** _`a`의 크기가 블록 크기보다 큰 경우, 각 블록의 합계만 저장합니다._

## 구성

- 배열 크기: `SIZE = 8`
- 블록당 스레드 수: `TPB = 8`
- 블록 수: 1
- 공유 메모리: `TPB`개 원소

참고:

- **데이터 로딩**: 각 스레드가 LayoutTensor 접근을 통해 원소 하나를 로드
- **메모리 패턴**: address_space를 지정한 LayoutTensor로 중간 결과를 공유 메모리에 저장
- **스레드 동기화**: 연산 단계 간 조율
- **접근 패턴**: stride 기반 병렬 연산
- **타입 안전성**: LayoutTensor의 타입 시스템 활용

## 작성할 코드

```mojo
{{#include ../../../../../problems/p14/p14.mojo:prefix_sum_simple}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p14/p14.mojo" class="filename">전체 파일 보기: problems/p14/p14.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 데이터를 `shared[local_i]`에 로드
2. `offset = 1`에서 시작해 매 단계마다 2배로 증가
3. `local_i >= offset`인 원소에 대해 덧셈 수행
4. 각 단계 사이에 `barrier()` 호출

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
pixi run p14 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p14 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p14 --simple
```

  </div>
  <div class="tab-content">

```bash
uv run poe p14 --simple
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력은 다음과 같습니다:

```txt
out: DeviceBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p14/p14.mojo:prefix_sum_simple_solution}}
```

<div class="solution-explanation">

병렬 (inclusive) prefix sum 알고리즘은 다음과 같이 동작합니다:

### 설정 및 구성

- `TPB` (블록당 스레드 수) = 8
- `SIZE` (배열 크기) = 8

### 경쟁 상태 방지

이 알고리즘은 명시적 동기화를 통해 읽기-쓰기 충돌을 방지합니다:

- **읽기 단계**: 모든 스레드가 먼저 필요한 값을 로컬 변수 `current_val`에 읽어둠
- **동기화**: `barrier()`로 모든 읽기가 완료된 후에야 쓰기가 시작되도록 보장
- **쓰기 단계**: 모든 스레드가 계산된 값을 안전하게 공유 메모리에 기록

이렇게 하면 여러 스레드가 동시에 같은 공유 메모리 위치를 읽고 쓸 때 발생하는 경쟁 상태를 방지할 수 있습니다.

**대안적 접근**: 경쟁 상태를 방지하는 또 다른 방법은 _double-buffering_ 입니다. 공유 메모리를 2배로 할당한 뒤, 한 버퍼에서 읽고 다른 버퍼에 쓰는 것을 번갈아 수행하는 방식입니다. 이 방법은 경쟁 상태를 완전히 제거하지만, 공유 메모리 사용량이 늘어나고 복잡도가 올라갑니다. 학습 목적으로는 이해하기 더 쉬운 명시적 동기화 방식을 사용합니다.

### 스레드 매핑

- `thread_idx.x`: \\([0, 1, 2, 3, 4, 5, 6, 7]\\) (`local_i`)
- `block_idx.x`: \\([0, 0, 0, 0, 0, 0, 0, 0]\\)
- `global_i`: \\([0, 1, 2, 3, 4, 5, 6, 7]\\) (`block_idx.x * TPB + thread_idx.x`)

### 공유 메모리에 초기 로드

```txt
Threads:      T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
Input array:  [0    1    2    3    4    5    6    7]
shared:       [0    1    2    3    4    5    6    7]
               ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
              T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
```

### Offset = 1: 첫 번째 병렬 단계

활성 스레드: \\(T_1 \ldots T_7\\) (`local_i ≥ 1`인 스레드)

**읽기 단계**: 각 스레드가 필요한 값을 읽음:

```txt
T₁ reads shared[0] = 0    T₅ reads shared[4] = 4
T₂ reads shared[1] = 1    T₆ reads shared[5] = 5
T₃ reads shared[2] = 2    T₇ reads shared[6] = 6
T₄ reads shared[3] = 3
```

**동기화**: `barrier()`로 모든 읽기 완료를 보장

**쓰기 단계**: 각 스레드가 읽은 값을 현재 위치에 더함:

```txt
Before:      [0    1    2    3    4    5    6    7]
Add:              +0   +1   +2   +3   +4   +5   +6
                   |    |    |    |    |    |    |
Result:      [0    1    3    5    7    9    11   13]
                   ↑    ↑    ↑    ↑    ↑    ↑    ↑
                  T₁   T₂   T₃   T₄   T₅   T₆   T₇
```

### Offset = 2: 두 번째 병렬 단계

활성 스레드: \\(T_2 \ldots T_7\\) (`local_i ≥ 2`인 스레드)

**읽기 단계**: 각 스레드가 필요한 값을 읽음:

```txt
T₂ reads shared[0] = 0    T₅ reads shared[3] = 5
T₃ reads shared[1] = 1    T₆ reads shared[4] = 7
T₄ reads shared[2] = 3    T₇ reads shared[5] = 9
```

**동기화**: `barrier()`로 모든 읽기 완료를 보장

**쓰기 단계**: 각 스레드가 읽은 값을 더함:

```txt
Before:      [0    1    3    5    7    9    11   13]
Add:                   +0   +1   +3   +5   +7   +9
                        |    |    |    |    |    |
Result:      [0    1    3    6    10   14   18   22]
                        ↑    ↑    ↑    ↑    ↑    ↑
                       T₂   T₃   T₄   T₅   T₆   T₇
```

### Offset = 4: 세 번째 병렬 단계

활성 스레드: \\(T_4 \ldots T_7\\) (`local_i ≥ 4`인 스레드)

**읽기 단계**: 각 스레드가 필요한 값을 읽음:

```txt
T₄ reads shared[0] = 0    T₆ reads shared[2] = 3
T₅ reads shared[1] = 1    T₇ reads shared[3] = 6
```

**동기화**: `barrier()`로 모든 읽기 완료를 보장

**쓰기 단계**: 각 스레드가 읽은 값을 더함:

```txt
Before:      [0    1    3    6    10   14   18   22]
Add:                              +0   +1   +3   +6
                                  |    |    |    |
Result:      [0    1    3    6    10   15   21   28]
                                  ↑    ↑    ↑    ↑
                                  T₄   T₅   T₆   T₇
```

### 최종 결과를 output에 기록

```txt
Threads:      T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
global_i:     0    1    2    3    4    5    6    7
output:       [0    1    3    6    10   15   21   28]
              ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
              T₀   T₁   T₂   T₃   T₄   T₅   T₆   T₇
```

### 주요 구현 상세

**동기화 패턴**: 각 반복은 엄격한 읽기 → 동기화 → 쓰기 패턴을 따릅니다:

1. `var current_val: out.element_type = 0` - 로컬 변수 초기화
2. `current_val = shared[local_i - offset]` - 읽기 단계 (조건 충족 시)
3. `barrier()` - 경쟁 상태 방지를 위한 명시적 동기화
4. `shared[local_i] += current_val` - 쓰기 단계 (조건 충족 시)
5. `barrier()` - 다음 반복 전 동기화

**경쟁 상태 방지**: 읽기와 쓰기를 명시적으로 분리하지 않으면 여러 스레드가 동시에 같은 공유 메모리 위치에 접근하여 미정의 동작이 발생할 수 있습니다. 명시적 동기화를 사용한 2단계 접근 방식이 정확성을 보장합니다.

**메모리 안전성**: 알고리즘은 다음을 통해 메모리 안전성을 유지합니다:

- `if local_i >= offset and local_i < size`로 경계 검사
- 임시 변수의 적절한 초기화
- 데이터 레이스를 방지하는 조율된 접근 패턴

이 솔루션은 `barrier()`를 사용해 단계 간 올바른 동기화를 보장하고, `if global_i < size`로 배열 경계 검사를 처리합니다. 최종 결과는 각 원소 \\(i\\)가 \\(\sum_{j=0}^{i} a[j]\\)를 포함하는 inclusive prefix sum입니다.
</div>
</details>
