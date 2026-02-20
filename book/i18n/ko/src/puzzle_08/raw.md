<!-- i18n-source-commit: 9ac1b899ca05c1be26f2d9ee77fe97503d00cc0f -->

## 개요

벡터 `a`의 각 위치에 10을 더해 `output`에 저장하는 kernel을 구현해 보세요.

**참고:** _블록당 스레드 수가 `a`의 크기보다 작습니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- 스레드 블록 내에서 공유 메모리 사용하기
- barrier로 스레드 동기화하기
- 블록 로컬 데이터 저장소 관리하기

핵심은 공유 메모리가 블록 내 모든 스레드가 접근할 수 있는 빠른 로컬 저장소라는 점, 그리고 이를 사용할 때 스레드 간 조율이 필요하다는 점을 이해하는 것입니다.

## 구성

- 배열 크기: `SIZE = 8` 원소
- 블록당 스레드 수: `TPB = 4`
- 블록 수: 2
- 공유 메모리: 블록당 `TPB`개 원소

참고:

- **공유 메모리**: 블록 내 스레드들이 함께 사용하는 빠른 저장소
- **스레드 동기화**: `barrier()`를 사용한 조율
- **메모리 범위**: 공유 메모리는 블록 내에서만 보임
- **접근 패턴**: 로컬 인덱스 vs 전역 인덱스

> **주의**: 각 블록이 가질 수 있는 공유 메모리 크기는 _상수_ 로 정해져야 합니다. 이 값은 변수가 아닌 리터럴 Python 상수여야 합니다. 공유 메모리에 쓴 후에는 [barrier](https://docs.modular.com/mojo/stdlib/gpu/sync/barrier/)를 호출하여 스레드들이 서로 앞서가지 않도록 해야 합니다.

**학습 참고**: 이 퍼즐에서는 각 스레드가 자신의 공유 메모리 위치에만 접근하므로 `barrier()`가 엄밀히 필요하지 않습니다. 하지만 더 복잡한 상황에서 필요한 올바른 동기화 패턴을 익히기 위해 포함되어 있습니다.

## 완성할 코드

```mojo
{{#include ../../../../../problems/p08/p08.mojo:add_10_shared}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p08/p08.mojo" class="filename">전체 코드 보기: problems/p08/p08.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. `barrier()`로 공유 메모리 로드 완료 대기 (학습용 - 여기서는 엄밀히 필요하지 않음)
2. `local_i`로 공유 메모리 접근: `shared[local_i]`
3. `global_i`로 출력: `output[global_i]`
4. 가드 추가: `if global_i < size`

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

GPU 프로그래밍에서 공유 메모리 사용의 핵심 개념을 보여주는 솔루션입니다:

1. **메모리 계층 구조**
   - 글로벌 메모리: `a`와 `output` 배열 (느림, 모든 블록에서 보임)
   - 공유 메모리: `shared` 배열 (빠름, 스레드 블록 로컬)
   - 블록당 4개 스레드로 8개 원소를 처리하는 예시:

     ```txt
     글로벌 배열 a: [1 1 1 1 | 1 1 1 1]  # 입력: 모두 1

     Block (0):      Block (1):
     shared[0..3]    shared[0..3]
     [1 1 1 1]       [1 1 1 1]
     ```

2. **스레드 조율**
   - 로드 단계:

     ```txt
     Thread 0: shared[0] = a[0]=1    Thread 2: shared[2] = a[2]=1
     Thread 1: shared[1] = a[1]=1    Thread 3: shared[3] = a[3]=1
     barrier()    ↓         ↓        ↓         ↓   # 모든 로드 완료 대기
     ```

   - 처리 단계: 각 스레드가 자신의 공유 메모리 값에 10을 더함
   - 결과: `output[i] = shared[local_i] + 10 = 11`

   **참고**: 이 경우에는 각 스레드가 자신의 공유 메모리 위치(`shared[local_i]`)에만 쓰고 읽으므로 `barrier()`가 엄밀히 필요하지 않습니다. 하지만 스레드들이 서로의 데이터에 접근하는 상황에서 필수적인 동기화 패턴을 익히기 위해 포함되어 있습니다.

3. **인덱스 매핑**
   - 전역 인덱스: `block_dim.x * block_idx.x + thread_idx.x`

     ```txt
     Block 0 출력: [11 11 11 11]
     Block 1 출력: [11 11 11 11]
     ```

   - 로컬 인덱스: 공유 메모리 접근에 `thread_idx.x` 사용

     ```txt
     두 블록 모두 처리: 1 + 10 = 11
     ```

4. **메모리 접근 패턴**
   - 로드: 글로벌 → 공유 (coalesced 읽기로 1 값들 로드)
   - 동기화: `barrier()`로 모든 로드 완료 보장
   - 처리: 공유 메모리 값에 10 더하기
   - 저장: 결과(11)를 글로벌 메모리에 쓰기

이 패턴은 블록 내 스레드 조율을 유지하면서 공유 메모리로 데이터 접근을 최적화하는 방법을 보여줍니다.
</div>
</details>
