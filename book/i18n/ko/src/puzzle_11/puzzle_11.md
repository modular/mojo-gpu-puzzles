<!-- i18n-source-commit: aed4a8442a1f84c61f13efd9a44b2ef240f22f1d -->

# Puzzle 11: 풀링

## 개요

1D TileTensor `a`에서 각 위치의 직전 3개 값의 합을 계산하여 1D TileTensor `output`에 저장하는 커널을 구현하세요.

**풀링(pooling)** 은 일정 영역의 값들을 하나의 요약 값(예: 합, 최댓값, 평균)으로 압축하는 연산입니다. **슬라이딩 윈도우(sliding window)** 는 입력 위로 고정 크기의 윈도우를 한 칸씩 옮겨 가며 이 압축을 반복 적용해, 윈도우 위치마다 출력값을 하나씩 만들어냅니다. 여기서는 윈도우 폭이 3이고 요약 함수가 합이므로, 각 출력 원소는 현재 원소와 그 앞 두 원소의 합이 됩니다(사용 가능한 원소가 3개보다 적은 경계 지점에서는 특수 케이스로 처리).

**참고:** _각 위치마다 스레드 1개가 있습니다. 스레드당 전역 읽기 1회, 전역 쓰기 1회만 필요합니다._

<img src="/puzzle_11/media/11-w.png" alt="Pooling 시각화" class="light-mode-img">
<img src="/puzzle_11/media/11-b.png" alt="Pooling 시각화" class="dark-mode-img">

## 핵심 개념

이 퍼즐에서 배울 내용:

- TileTensor로 슬라이딩 윈도우 연산 구현하기
- [Puzzle 8](../puzzle_08/puzzle_08.md)에서 다룬 TileTensor 주소 공간(address_space)으로 공유 메모리 관리하기
- 효율적인 이웃 접근 패턴
- 경계 조건 처리

핵심은 TileTensor가 효율적인 윈도우 기반 연산은 유지하면서도 공유 메모리 관리를 간소화하는 방법입니다.

## 구성

- 배열 크기: `SIZE = 8`
- 블록당 스레드 수: `TPB = 8`
- 윈도우 크기: 3
- 공유 메모리: `TPB`개

참고:

- **TileTensor 할당**: `stack_allocation[dtype=dtype, address_space=AddressSpace.SHARED](row_major[TPB]())` 사용
- **윈도우 접근**: 3개짜리 윈도우에 자연스러운 인덱싱
- **경계 처리**: 처음 두 위치는 특수 케이스
- **메모리 패턴**: 스레드당 공유 메모리 로드 1회

## 완성할 코드

```mojo
{{#include ../../../../../problems/p11/p11.mojo:pooling}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p11/p11.mojo" class="filename">전체 파일 보기: problems/p11/p11.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. TileTensor와 주소 공간(address_space)으로 공유 메모리 생성
2. 자연스러운 인덱싱으로 데이터 로드: `shared[local_i] = a[global_i]`
3. 처음 두 위치를 특수 케이스로 처리
4. 윈도우 연산에 공유 메모리 활용
5. 경계 초과 접근에 가드 추가

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
pixi run p11
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p11
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p11
```

  </div>
  <div class="tab-content">

```bash
uv run poe p11
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력은 다음과 같습니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p11/p11.mojo:pooling_solution}}
```

<div class="solution-explanation">

TileTensor를 활용한 슬라이딩 윈도우 합계 구현입니다. 주요 단계는 다음과 같습니다:

1. **공유 메모리 설정**
   - TileTensor가 주소 공간(address_space)으로 블록 로컬 저장소를 생성:

     ```txt
     shared = stack_allocation[dtype=dtype, address_space=AddressSpace.SHARED](row_major[TPB]())
     ```

   - 각 스레드가 하나씩 로드:

     ```txt
     Input array:  [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     Block shared: [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     ```

   - `barrier()`로 모든 데이터 로드 완료를 보장

2. **경계 케이스**
   - 위치 0: 하나만

     ```txt
     output[0] = shared[0] = 0.0
     ```

   - 위치 1: 처음 두 값의 합

     ```txt
     output[1] = shared[0] + shared[1] = 0.0 + 1.0 = 1.0
     ```

3. **메인 윈도우 연산**
   - 위치 2 이후:

     ```txt
     Position 2: shared[0] + shared[1] + shared[2] = 0.0 + 1.0 + 2.0 = 3.0
     Position 3: shared[1] + shared[2] + shared[3] = 1.0 + 2.0 + 3.0 = 6.0
     Position 4: shared[2] + shared[3] + shared[4] = 2.0 + 3.0 + 4.0 = 9.0
     ...
     ```

   - TileTensor의 자연스러운 인덱싱:

     ```txt
     # 3개짜리 슬라이딩 윈도우
     window_sum = shared[i-2] + shared[i-1] + shared[i]
     ```

> **단일 블록 전제:** 이 퍼즐이 `BLOCKS_PER_GRID = (1, 1)`과 `SIZE == TPB = 8`로 구성되어 있어서
> 모든 스레드가 같은 블록에 속하고 `global_i == local_i`가 보장되기 때문에 이 솔루션이 올바르게
> 동작합니다. 이 제약에서는 `global_i > 1`일 때마다 `local_i >= 2`이므로
> `shared[local_i - 2]`와 `shared[local_i - 1]`이 언제나 유효합니다.
>
> **다중 블록** 커널에서는 0번 블록 이후의 각 블록에서 첫 두 스레드가 `global_i > 1`인데도
> `local_i = 0` 또는 `local_i = 1`이 되어 공유 메모리 범위 초과 읽기가 발생합니다. 다중 블록
> 풀링에서 안정적으로 동작하는 패턴은 `local_i`로 가드를 걸고, 헤일로(halo) 원소에 대해서는
> 전역 읽기로 대체하는 것입니다:
>
> ```mojo
> if local_i >= 2:
>     output[global_i] = shared[local_i-2] + shared[local_i-1] + shared[local_i]
> elif local_i == 1 and global_i >= 2:
>     output[global_i] = a[global_i-2] + shared[0] + shared[1]
> elif local_i == 0 and global_i >= 2:
>     output[global_i] = a[global_i-2] + a[global_i-1] + shared[0]
> ```

4. **메모리 접근 패턴**
   - 스레드마다 공유 텐서로 전역 읽기 1회
   - 공유 메모리를 통한 효율적인 이웃 접근
   - TileTensor의 장점:
     - 자동 경계 검사
     - 자연스러운 윈도우 인덱싱
     - 레이아웃을 인식하는 메모리 접근
     - 전 과정에 걸친 타입 안전성

공유 메모리의 성능과 TileTensor의 안전성 및 편의성을 결합한 방식입니다:

- 전역 메모리 접근 최소화
- 윈도우 연산 간소화
- 깔끔한 경계 처리
- 병합 접근 패턴 유지

최종 출력은 누적 윈도우 합계입니다:

```txt
[0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
```

</div>
</details>
