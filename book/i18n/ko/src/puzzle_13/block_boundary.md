<!-- i18n-source-commit: 29894ca396ed4e90fc447bf66b8e2a216576d8a8 -->

# 블록 경계 버전

1D LayoutTensor `a`와 1D LayoutTensor `b`의 1D convolution을 계산하여 1D LayoutTensor `output`에 저장하는 kernel을 구현하세요.

**참고:** _일반적인 경우를 처리해야 합니다. 스레드당 global read 2회, global write 1회만 필요합니다._

## 구성

- 입력 배열 크기: `SIZE_2 = 15`
- Kernel 크기: `CONV_2 = 4`
- 블록당 스레드 수: `TPB = 8`
- 블록 수: 2
- 공유 메모리: 입력용 `TPB + CONV_2 - 1`개

참고:

- **확장 로딩**: 경계 겹침 영역을 고려
- **블록 가장자리**: 블록 경계를 넘는 데이터 처리
- **메모리 레이아웃**: 공유 메모리의 효율적 활용
- **동기화**: 적절한 스레드 간 조율

## 작성할 코드

```mojo
{{#include ../../../../../problems/p13/p13.mojo:conv_1d_block_boundary}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p13/p13.mojo" class="filename">전체 파일 보기: problems/p13/p13.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. `LayoutTensor[dtype, Layout.row_major(TPB + CONV_2 - 1), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`으로 공유 메모리 할당
2. 메인 데이터 로드: `shared_a[local_i] = a[global_i]`
3. 경계 데이터 로드: `if local_i < CONV_2 - 1`일 때 다음 블록의 데이터 처리
4. Kernel 로드: `shared_b[local_i] = b[local_i]`
5. 입력 범위 안에서 합산: `if global_i + j < SIZE_2`

</div>
</details>

### 코드 실행

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
pixi run p13 --block-boundary
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p13 --block-boundary
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p13 --block-boundary
```

  </div>
  <div class="tab-content">

```bash
uv run poe p13 --block-boundary
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력은 다음과 같습니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([14.0, 20.0, 26.0, 32.0, 38.0, 44.0, 50.0, 56.0, 62.0, 68.0, 74.0, 80.0, 41.0, 14.0, 0.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p13/p13.mojo:conv_1d_block_boundary_solution}}
```

<div class="solution-explanation">

확장된 공유 메모리를 사용해 블록 경계를 넘는 1D convolution을 처리하는 솔루션입니다. 자세히 분석해 보겠습니다:

### 메모리 레이아웃과 크기 계산

```txt
테스트 구성:
- 전체 배열 크기: SIZE_2 = 15
- 그리드: 2 블록 × 8 스레드
- Convolution kernel: CONV_2 = 4

Block 0 공유 메모리:  [0 1 2 3 4 5 6 7|8 9 10]  // TPB(8) + (CONV_2-1)(3) 패딩
Block 1 공유 메모리:  [8 9 10 11 12 13 14 0|0 0 0]  // 두 번째 블록. 데이터(7) + 그리드 채움용 패딩(1) + (CONV_2-1)(3) 패딩

크기 계산:
- 메인 데이터: TPB개 (8)
- 겹침 영역: CONV_2 - 1개 (4 - 1 = 3)
- 합계: TPB + CONV_2 - 1 = 8 + 4 - 1 = 11개
```

### 구현 상세

1. **공유 메모리 할당**:

   ```mojo
   # convolution 윈도우에 필요한 패딩을 먼저 고려
   shared_a = LayoutTensor[dtype, Layout.row_major(TPB + CONV_2 - 1), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   shared_b = LayoutTensor[dtype, Layout.row_major(CONV_2), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   ```

   이 할당 패턴은 블록의 데이터와 겹침 영역을 모두 수용할 수 있는 충분한 공간을 확보합니다.

2. **데이터 로딩 전략**:

   ```mojo
   # 메인 블록 데이터
   if global_i < SIZE_2:
       shared_a[local_i] = a[global_i]
   else:
       shared_a[local_i] = 0

   # 다음 블록의 경계 데이터
   if local_i < CONV_2 - 1:
       next_idx = global_i + TPB
       if next_idx < SIZE_2:
           shared_a[TPB + local_i] = a[next_idx]
       else:
           # 범위 밖 원소를 0으로 초기화하여
           # 미정의 동작을 유발하는 초기화되지 않은 메모리 읽기를 방지
           shared_a[TPB + local_i] = 0
   ```

   - `local_i < CONV_2 - 1`인 스레드만 경계 데이터를 로드
   - 불필요한 스레드 분기 방지
   - 메인 데이터 로드의 메모리 병합(coalescing) 유지
   - 범위 밖 원소를 명시적으로 0으로 초기화하여 미정의 동작 방지

3. **Kernel 로딩**:

   ```mojo
   if local_i < b_size:
       shared_b[local_i] = b[local_i]
   ```

   - 스레드당 1회 로드
   - Kernel 크기로 범위 제한

4. **Convolution 연산**:

   ```mojo
   if global_i < SIZE_2:
       var local_sum: output.element_type = 0
       @parameter
       for j in range(CONV_2):
           if global_i + j < SIZE_2:
               local_sum += shared_a[local_i + j] * shared_b[j]
   ```

   - `@parameter`로 컴파일 타임 루프 전개
   - `output.element_type`으로 적절한 타입 추론
   - 의미적으로 올바른 경계 검사: 유효한 입력 위치에서만 convolution 계산

### 메모리 접근 패턴 분석

1. **Block 0 접근 패턴**:

   ```txt
   Thread 0: [0 1 2 3] × [0 1 2 3]
   Thread 1: [1 2 3 4] × [0 1 2 3]
   Thread 2: [2 3 4 5] × [0 1 2 3]
   ...
   Thread 7: [7 8 9 10] × [0 1 2 3]  // 겹침 영역 데이터 사용
   ```

2. **Block 1 접근 패턴**:
Thread 4부터 `global_i + j < SIZE_2`가 `False`로 평가되어 해당 반복이 건너뛰어지는 점에 주목하세요.

   ```txt
   Thread 0: [8  9 10 11] × [0 1 2 3]
   Thread 1: [9 10 11 12] × [0 1 2 3]
   ...
   Thread 4: [12 13 14] × [0 1 2]       // 끝부분 제로 패딩
   Thread 5: [13 14]    × [0 1]
   Thread 6: [14]       × [0]
   Thread 7: 건너뜀                      // 모든 j에 대해 global_i + j < SIZE_2가 false, 연산 없음
   ```

### 성능 최적화

1. **메모리 병합(Coalescing)**:
   - 메인 데이터 로드: 인접 스레드가 연속된 메모리에 접근
   - 경계 데이터: 필요한 스레드만 참여
   - 단일 barrier 동기화 지점

2. **스레드 분기 최소화**:
   - 메인 로딩과 경계 로딩의 깔끔한 분리
   - Warp 내 균일한 연산 패턴
   - 효율적인 경계 검사

3. **공유 메모리 활용**:
   - 블록 경계 처리에 최적화된 크기 설정
   - 접근 패턴에서 bank conflict 없음
   - 로드한 데이터의 효율적 재사용

4. **경계 처리**:
   - 범위 밖 원소를 명시적으로 0으로 설정하여 초기화되지 않은 공유 메모리 읽기 방지
   - `global_i + j < SIZE_2`로 공유 메모리가 아닌 실제 입력 범위 기준의 경계 검사
   - 불필요한 연산 없이 적절한 엣지 케이스 처리

### 경계 조건 개선

이 솔루션은 공유 메모리 범위를 확인하는 대신 `if global_i + j < SIZE_2:`를 사용합니다. 이 패턴은:

- **수학적으로 정확**: 입력 데이터가 실제로 존재하는 위치에서만 convolution 계산
- **더 효율적**: 입력 배열을 넘어선 위치에 대한 불필요한 연산 회피
- **더 안전**: 공유 메모리의 제로 패딩 동작에 의존하지 않음

이 구현은 블록 간 convolution을 효율적으로 수행하면서 다음을 유지합니다:

- 적절한 경계 검사를 통한 메모리 안전성
- 최적화된 메모리 접근을 통한 높은 성능
- LayoutTensor 추상화를 활용한 깔끔한 코드 구조
- 최소한의 동기화 오버헤드
- 수학적으로 건전한 경계 처리

</div>
</details>
