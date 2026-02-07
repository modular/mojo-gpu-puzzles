<!-- i18n-source-commit: db06539cab77774402e8a4bf955018fd853803d9 -->

## 개요

벡터 `a`에서 각 위치의 직전 3개 값의 합을 계산하여 벡터 `output`에 저장하는 kernel을 구현하세요.

**참고:** _각 위치마다 스레드 1개가 있습니다. 스레드당 global read 1회, global write 1회만 필요합니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- 공유 메모리로 슬라이딩 윈도우 연산 구현하기
- Pooling의 경계 조건 처리
- 이웃 데이터 접근을 위한 스레드 간 협력

핵심은 공유 메모리를 사용해 윈도우 내 값들에 효율적으로 접근하는 것입니다. 시퀀스 앞부분은 특별히 처리해야 합니다.

## 구성

- 배열 크기: `SIZE = 8`
- 블록당 스레드 수: `TPB = 8`
- 윈도우 크기: 3
- 공유 메모리: `TPB`개

참고:

- **윈도우 접근**: 각 출력은 이전 최대 3개 값에 의존합니다
- **경계 처리**: 처음 두 위치는 특별한 처리가 필요합니다
- **메모리 패턴**: 스레드당 공유 메모리 로드 1회
- **스레드 동기화**: 윈도우 연산 전에 조율 필요

## 작성할 코드

```mojo
{{#include ../../../../../problems/p11/p11.mojo:pooling}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p11/p11.mojo" class="filename">전체 파일 보기: problems/p11/p11.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 데이터를 로드하고 `barrier()` 호출
2. 특수 케이스: `output[0] = shared[0]`, `output[1] = shared[0] + shared[1]`
3. 일반 케이스: `if 1 < global_i < size`
4. 세 값의 합: `shared[local_i - 2] + shared[local_i - 1] + shared[local_i]`

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

공유 메모리를 활용한 슬라이딩 윈도우 합계 구현입니다. 주요 단계는 다음과 같습니다:

1. **공유 메모리 설정**
   - 공유 메모리에 `TPB`개 할당:

     ```txt
     Input array:  [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     Block shared: [0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0]
     ```

   - 각 스레드가 글로벌 메모리에서 하나씩 로드
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

   - 로컬 인덱스를 사용한 윈도우 계산:

     ```txt
     # 3개짜리 슬라이딩 윈도우
     window_sum = shared[i-2] + shared[i-1] + shared[i]
     ```

4. **메모리 접근 패턴**
   - 스레드당 공유 메모리로 global read 1회
   - 스레드당 공유 메모리에서 global write 1회
   - 이웃 접근을 위해 공유 메모리 활용
   - 병합(coalescing) 메모리 접근 패턴 유지

이 방식의 성능 최적화 포인트:

- 글로벌 메모리 접근 최소화
- 공유 메모리로 빠른 이웃 조회
- 깔끔한 경계 처리
- 효율적인 메모리 병합

최종 출력은 누적 윈도우 합계를 보여줍니다:

```txt
[0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
```

</div>
</details>
