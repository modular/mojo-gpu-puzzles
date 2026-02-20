<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

## 개요

벡터 `a`와 벡터 `b`의 내적을 계산하여 `output`(단일 값)에 저장하는 kernel을 구현하세요.

**참고:** _각 위치마다 스레드 1개가 있습니다. 스레드당 global read 2회, 블록당 global write 1회만 필요합니다._

## 핵심 개념

이 퍼즐에서 다루는 내용:

- 여러 값을 하나로 합치는 병렬 reduction 구현하기
- 공유 메모리에 중간 결과 저장하기
- 스레드끼리 협력하여 하나의 결과 만들기

핵심은 공유 메모리와 병렬 연산을 활용해, 흩어져 있는 값들을 효율적으로 하나의 결과로 모아가는 과정을 이해하는 것입니다.

## 구성

- 벡터 크기: `SIZE = 8`
- 블록당 스레드 수: `TPB = 8`
- 블록 수: 1
- 출력 크기: 1
- 공유 메모리: `TPB`개

참고:

- **요소 접근**: 각 스레드가 `a`와 `b`에서 대응하는 요소를 읽음
- **부분 결과**: 중간 값을 계산하고 저장
- **스레드 조율**: 결과를 합치기 전에 동기화
- **최종 reduction**: 부분 결과를 스칼라 출력으로 변환

_참고: 이 문제에서는 공유 메모리 읽기 횟수를 신경 쓸 필요가 없습니다. 그 문제는 나중에 다루겠습니다._

## 완성할 코드

```mojo
{{#include ../../../../../problems/p12/p12.mojo:dot_product}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p12/p12.mojo" class="filename">전체 파일 보기: problems/p12/p12.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. `shared[local_i]`에 `a[global_i] * b[global_i]`를 저장
2. `barrier()`를 호출하여 동기화
3. 스레드 0이 공유 메모리의 모든 곱을 합산
4. 최종 합계를 `output[0]`에 기록

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
pixi run p12
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p12
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p12
```

  </div>
  <div class="tab-content">

```bash
uv run poe p12
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
{{#include ../../../../../solutions/p12/p12.mojo:dot_product_solution}}
```

<div class="solution-explanation">

공유 메모리를 활용한 병렬 reduction 알고리즘으로 내적을 계산하는 솔루션입니다. 단계별로 살펴보겠습니다:

### 1단계: 요소별 곱셈

각 스레드가 곱셈 하나를 수행합니다:

```txt
Thread i: shared[i] = a[i] * b[i]
```

### 2단계: 병렬 Reduction

활성 스레드를 매 단계마다 절반으로 줄이는 트리 기반 방식입니다:

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

1. **메모리 접근 패턴**:
   - 각 스레드가 글로벌 메모리에서 정확히 두 값을 로드 (`a[i]`, `b[i]`)
   - 중간 결과에 공유 메모리 사용
   - 최종 결과는 글로벌 메모리에 1회 기록

2. **스레드 동기화**:
   - 초기 곱셈 후 `barrier()`
   - 각 reduction 단계 후 `barrier()`
   - Reduction 단계 간 경쟁 상태 방지

3. **Reduction 로직**:

   ```mojo
   stride = TPB // 2
   while stride > 0:
       if local_i < stride:
           shared[local_i] += shared[local_i + stride]
       barrier()
       stride //= 2
   ```

   - 매 단계마다 stride를 절반으로
   - 활성 스레드만 덧셈 수행
   - 작업 효율성 유지

4. **성능 고려 사항**:
   - \\(n\\)개 요소에 대해 \\(\log_2(n)\\) 단계
   - 병합(coalesced) 메모리 접근 패턴
   - 최소한의 스레드 분기
   - 공유 메모리의 효율적 활용

이 구현은 순차 실행의 \\(O(n)\\)에 비해 \\(O(\log n)\\) 시간 복잡도를 달성하며, 병렬 reduction 알고리즘의 위력을 보여줍니다.

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
