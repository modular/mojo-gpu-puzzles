<!-- i18n-source-commit: 8726393ce8f2ba4d52d2ceec6352706da1f1806a -->

# Puzzle 8: 공유 메모리

## 개요

1D TileTensor `a`의 각 위치에 10을 더해 1D TileTensor `output`에 저장하는 커널을 구현해 보세요.

**참고:** _블록당 스레드 수가 `a`의 크기보다 작습니다._

<img src="/puzzle_08/media/08.png" alt="공유 메모리 시각화" class="light-mode-img">
<img src="/puzzle_08/media/08d.png" alt="공유 메모리 시각화" class="dark-mode-img">

## 핵심 개념

이 퍼즐에서 배울 내용:

- address_space를 활용한 TileTensor의 공유 메모리 기능
- 공유 메모리를 사용할 때의 스레드 동기화
- TileTensor로 블록 로컬 데이터 관리하기

핵심은 TileTensor가 블록 로컬 저장소의 성능은 그대로 유지하면서 공유 메모리 관리를 얼마나 간소화하는지 이해하는 것입니다.

## 구성

- 배열 크기: `SIZE = 8` 원소
- 블록당 스레드 수: `TPB = 4`
- 블록 수: 2
- 공유 메모리: 블록당 `TPB`개 원소

> **경고**: 각 블록에는 해당 블록의 스레드들이 읽고 쓸 수 있는 공유 메모리의 양이 _상수_로 고정되어 있습니다. 이 값은 파이썬 리터럴 상수여야 하며 변수를 사용할 수 없습니다. 공유 메모리에 쓴 후에는 [barrier](https://docs.modular.com/mojo/std/gpu/sync/sync/barrier/)를 호출해 스레드들이 교차하지 않도록 해야 합니다.

**학습 참고**: 이 퍼즐에서는 각 스레드가 자신의 공유 메모리 위치에만 접근하므로 `barrier()`가 엄밀히 필요하지 않습니다. 하지만 더 복잡한 상황에서 필요한 올바른 동기화 패턴을 익히기 위해 포함되어 있습니다.

## 완성할 코드

```mojo
{{#include ../../../../../problems/p08/p08.mojo:add_10_shared}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p08/p08.mojo" class="filename">전체 코드 보기: problems/p08/p08.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. address_space 파라미터로 TileTensor 공유 메모리 생성
2. 자연스러운 인덱싱으로 데이터 로드: `shared[local_i] = a[global_i]`
3. `barrier()`로 동기화 (학습용 - 여기서는 엄밀히 필요하지 않음)
4. 공유 메모리 인덱스로 데이터 처리
5. 범위를 벗어난 접근을 방지하는 가드

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
pixi run p08
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p08
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p08
```

  </div>
  <div class="tab-content">

```bash
uv run poe p08
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력이 다음과 같이 나타납니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p08/p08.mojo:add_10_shared_solution}}
```

<div class="solution-explanation">

TileTensor가 성능을 유지하면서 공유 메모리 사용을 얼마나 간소화하는지 보여주는 솔루션입니다:

1. **TileTensor를 사용한 메모리 계층 구조**
   - 전역 텐서: `a`와 `output` (느림, 모든 블록에서 보임)
   - 공유 텐서: `shared` (빠름, 스레드 블록 로컬)
   - 블록당 4개 스레드로 8개 원소를 처리하는 예시:

     ```txt
     전역 텐서 a: [1 1 1 1 | 1 1 1 1]  # 입력: 모두 1

     Block (0):         Block (1):
     shared[0..3]       shared[0..3]
     [1 1 1 1]          [1 1 1 1]
     ```

2. **스레드 조율**
   - 로드 단계 (자연스러운 인덱싱 사용):

     ```txt
     Thread 0: shared[0] = a[0]=1    Thread 2: shared[2] = a[2]=1
     Thread 1: shared[1] = a[1]=1    Thread 3: shared[3] = a[3]=1
     barrier()    ↓         ↓        ↓         ↓   # 모든 로드 완료 대기
     ```

   - 처리 단계: 각 스레드가 자신의 공유 텐서 값에 10을 더함
   - 결과: `output[global_i] = shared[local_i] + 10 = 11`

   **참고**: 이 경우에는 각 스레드가 자신의 공유 메모리 위치(`shared[local_i]`)에만 쓰고 읽으므로 `barrier()`가 엄밀히 필요하지 않습니다. 하지만 스레드들이 서로의 데이터에 접근하는 상황에서 필수적인 동기화 패턴을 익히기 위해 포함되어 있습니다.

3. **TileTensor의 장점**
   - 공유 메모리 할당:

     ```txt
     # address_space를 사용한 깔끔한 TileTensor API
     shared = stack_allocation[dtype=dtype, address_space=AddressSpace.SHARED](row_major[TPB]())
     ```

   - 전역과 공유 메모리 모두 자연스러운 인덱싱:

     ```txt
     Block 0 출력: [11 11 11 11]
     Block 1 출력: [11 11 11 11]
     ```

   - 내장된 레이아웃 관리와 타입 안전성

4. **메모리 접근 패턴**
   - 로드: 전역 텐서 → 공유 텐서 (최적화됨)
   - 동기화: 원시 메모리 버전과 동일한 `barrier()` 필요
   - 처리: 공유 메모리 값에 10 더하기
   - 저장: 결과(11)를 전역 텐서에 쓰기

이 패턴은 TileTensor가 공유 메모리의 성능 이점을 유지하면서 더 편리한 API와 내장 기능을 제공하는 방법을 보여줍니다.
</div>
</details>
