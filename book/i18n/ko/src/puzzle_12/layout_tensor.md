<!-- i18n-source-commit: 51143596e241ae5954474ecb3133b1d7b147f6fc -->

## 개요

1D LayoutTensor `a`와 1D LayoutTensor `b`의 내적을 계산하여 1D LayoutTensor `output`(단일 값)에 저장하는 kernel을 구현하세요.

**참고:** _각 위치마다 스레드 1개가 있습니다. 스레드당 global read 2회, 블록당 global write 1회만 필요합니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- [Puzzle 8](../puzzle_08/layout_tensor.md), [Puzzle 11](../puzzle_11/layout_tensor.md)에서 이어지는 LayoutTensor 기반 병렬 reduction
- 주소 공간(address_space)을 활용한 공유 메모리 관리
- 여러 스레드가 협력해 하나의 결과를 만들어가는 과정
- 레이아웃을 인식하는 텐서 연산

핵심은 LayoutTensor가 메모리 관리를 간소화하면서도, 병렬 reduction의 효율은 그대로 살리는 방식을 이해하는 것입니다.

## 구성

- 벡터 크기: `SIZE = 8`
- 블록당 스레드 수: `TPB = 8`
- 블록 수: 1
- 출력 크기: 1
- 공유 메모리: `TPB`개

참고:

- **LayoutTensor 할당**: `LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()` 사용
- **요소 접근**: 경계 검사가 자동으로 따라오는 자연스러운 인덱싱
- **레이아웃 처리**: 입력용과 출력용 레이아웃을 따로 구성
- **스레드 조율**: Raw 버전과 동일하게 `barrier()`로 동기화

## 완성할 코드

```mojo
{{#include ../../../../../problems/p12/p12_layout_tensor.mojo:dot_product_layout_tensor}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p12/p12_layout_tensor.mojo" class="filename">전체 파일 보기: problems/p12/p12_layout_tensor.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. LayoutTensor와 주소 공간(address_space)으로 공유 메모리 생성
2. `shared[local_i]`에 `a[global_i] * b[global_i]`를 저장
3. `barrier()`와 함께 병렬 reduction 패턴 적용
4. 스레드 0이 최종 결과를 `output[0]`에 기록

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
pixi run p12_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p12_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p12_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p12_layout_tensor
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력은 다음과 같습니다:

```txt
out: HostBuffer([0.0])
expected: HostBuffer([140.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p12/p12_layout_tensor.mojo:dot_product_layout_tensor_solution}}
```

<div class="solution-explanation">

LayoutTensor를 활용한 병렬 reduction으로 내적을 계산하는 솔루션입니다. 단계별로 살펴보겠습니다:

### 1단계: 요소별 곱셈

각 스레드가 직관적인 인덱싱으로 곱셈 연산을 하나씩 처리합니다:

```mojo
shared[local_i] = a[global_i] * b[global_i]
```

### 2단계: 병렬 Reduction

레이아웃을 인식하는 트리 기반 reduction입니다:

```txt
초기값:    [0*0  1*1  2*2  3*3  4*4  5*5  6*6  7*7]
        = [0    1    4    9    16   25   36   49]

Step 1:   [0+16 1+25 4+36 9+49  16   25   36   49]
        = [16   26   40   58   16   25   36   49]

Step 2:   [16+40 26+58 40   58   16   25   36   49]
        = [56   84   40   58   16   25   36   49]

Step 3:   [56+84  84   40   58   16   25   36   49]
        = [140   84   40   58   16   25   36   49]
```

### 구현의 핵심 특징

1. **메모리 관리**:
   - 주소 공간(address_space) 파라미터 하나로 공유 메모리를 깔끔하게 할당
   - 타입 안전한 연산이 보장되고
   - 경계 검사가 자동으로 따라오며
   - 인덱싱도 레이아웃을 인식

2. **스레드 동기화**:
   - 초기 곱셈이 끝나면 `barrier()`
   - Reduction 단계 사이마다 `barrier()`
   - 스레드 간 안전한 조율 보장

3. **Reduction 로직**:

   ```mojo
   stride = TPB // 2
   while stride > 0:
       if local_i < stride:
           shared[local_i] += shared[local_i + stride]
       barrier()
       stride //= 2
   ```

4. **성능상 이점**:
   - \\(O(\log n)\\) 시간 복잡도
   - 병합(coalesced) 메모리 접근
   - 최소한의 스레드 분기
   - 공유 메모리의 효율적 활용

LayoutTensor 버전은 병렬 reduction의 효율은 그대로 유지하면서, 여기에 더해:

- 타입 안전성이 한층 강화되고
- 메모리 관리가 더 깔끔해지며
- 레이아웃을 자동으로 인식하고
- 인덱싱 문법도 자연스러워집니다

### Barrier 동기화의 중요성

Reduction 단계 사이의 `barrier()`는 정확한 결과를 위해 반드시 필요합니다. 그 이유를 살펴보겠습니다:

`barrier()`가 없으면 경쟁 상태가 발생합니다:

```text
초기 공유 메모리: [0 1 4 9 16 25 36 49]

Step 1 (stride = 4):
Thread 0 읽기: shared[0] = 0, shared[4] = 16
Thread 1 읽기: shared[1] = 1, shared[5] = 25
Thread 2 읽기: shared[2] = 4, shared[6] = 36
Thread 3 읽기: shared[3] = 9, shared[7] = 49

barrier 없이:
- Thread 0 쓰기: shared[0] = 0 + 16 = 16
- Thread 1이 Thread 0보다 먼저 다음 단계(stride = 2)로 넘어가서
  16이 아닌 이전 값 shared[0] = 0을 읽어버립니다!
```

`barrier()`가 있으면:

```text
Step 1 (stride = 4):
모든 스레드가 합을 기록:
[16 26 40 58 16 25 36 49]
barrier()가 모든 스레드에게 이 값들이 보이도록 보장

Step 2 (stride = 2):
이제 업데이트된 값을 안전하게 읽을 수 있음:
Thread 0: shared[0] = 16 + 40 = 56
Thread 1: shared[1] = 26 + 58 = 84
```

`barrier()`는 다음을 보장합니다:

1. 현재 단계의 모든 쓰기가 끝난 뒤에야 다음으로 넘어감
2. 모든 스레드가 최신 값을 볼 수 있음
3. 어떤 스레드도 앞서 나가지 않음
4. 공유 메모리가 항상 일관된 상태를 유지

이런 동기화 지점이 없으면:

- 경쟁 상태가 발생하고
- 스레드가 이미 지난 값을 읽게 되며
- 실행할 때마다 결과가 달라지고
- 최종 합계가 틀어질 수 있습니다

</div>
</details>
