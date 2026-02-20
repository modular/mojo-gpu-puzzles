<!-- i18n-source-commit: 1cd13cbe87682d50679d452938efab4cc79ddb78 -->

# 단일 블록을 사용한 기본 버전

1D LayoutTensor `a`와 1D LayoutTensor `b`의 1D convolution을 계산하여 1D LayoutTensor `output`에 저장하는 kernel을 구현하세요.

**참고:** _일반적인 경우를 처리해야 합니다. 스레드당 global read 2회, global write 1회만 필요합니다._

## 핵심 개념

이 퍼즐에서 다루는 내용:

- GPU에서 슬라이딩 윈도우 연산 구현하기
- 스레드 간 데이터 의존성 관리하기
- 겹치는 영역에 공유 메모리 활용하기

핵심은 경계 조건을 올바르게 유지하면서도 겹치는 원소에 효율적으로 접근하는 방법을 이해하는 것입니다.

## 구성

- 입력 배열 크기: `SIZE = 6`
- Kernel 크기: `CONV = 3`
- 블록당 스레드 수: `TPB = 8`
- 블록 수: 1
- 공유 메모리: `SIZE`와 `CONV` 크기의 배열 2개

참고:

- **데이터 로딩**: 각 스레드가 입력 배열과 kernel에서 원소를 하나씩 로드
- **메모리 패턴**: 입력 배열과 convolution kernel을 저장하는 공유 배열
- **스레드 동기화**: 연산 시작 전 스레드 간 조율

## 완성할 코드

```mojo
{{#include ../../../../../problems/p13/p13.mojo:conv_1d_simple}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p13/p13.mojo" class="filename">전체 파일 보기: problems/p13/p13.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. `LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()`으로 공유 메모리 할당
2. 입력을 `shared_a[local_i]`에, kernel을 `shared_b[local_i]`에 로드
3. 데이터 로드 후 `barrier()` 호출
4. 경계 안에서 곱을 합산: `if local_i + j < SIZE`
5. `global_i < SIZE`일 때만 결과 기록

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
pixi run p13 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p13 --simple
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p13 --simple
```

  </div>
  <div class="tab-content">

```bash
uv run poe p13 --simple
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력은 다음과 같습니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([5.0, 8.0, 11.0, 14.0, 5.0, 0.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p13/p13.mojo:conv_1d_simple_solution}}
```

<div class="solution-explanation">

공유 메모리를 활용해 겹치는 원소에 효율적으로 접근하는 1D convolution 구현입니다. 단계별로 살펴보겠습니다:

### 메모리 레이아웃

```txt
입력 배열 a:       [0  1  2  3  4  5]
Kernel b:        [0  1  2]
```

### 연산 과정

1. **데이터 로딩**:

   ```txt
   shared_a: [0  1  2  3  4  5]  // 입력 배열
   shared_b: [0  1  2]           // Convolution kernel
   ```

2. 각 위치 i에 대한 **convolution 연산**:

   ```txt
   output[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] = 0*0 + 1*1 + 2*2 = 5
   output[1] = a[1]*b[0] + a[2]*b[1] + a[3]*b[2] = 1*0 + 2*1 + 3*2 = 8
   output[2] = a[2]*b[0] + a[3]*b[1] + a[4]*b[2] = 2*0 + 3*1 + 4*2 = 11
   output[3] = a[3]*b[0] + a[4]*b[1] + a[5]*b[2] = 3*0 + 4*1 + 5*2 = 14
   output[4] = a[4]*b[0] + a[5]*b[1] + 0*b[2]    = 4*0 + 5*1 + 0*2 = 5
   output[5] = a[5]*b[0] + 0*b[1]   + 0*b[2]     = 5*0 + 0*1 + 0*2 = 0
   ```

### 구현 상세

1. **스레드 참여 범위와 효율성**:
   - 적절한 스레드 가드가 없는 비효율적 접근:

     ```mojo
     # 비효율적 버전 - 결과가 사용되지 않을 스레드도 모두 연산 수행
     local_sum = Scalar[dtype](0)
     for j in range(CONV):
         if local_i + j < SIZE:
             local_sum += shared_a[local_i + j] * shared_b[j]
     # 마지막 쓰기만 가드
     if global_i < SIZE:
         output[global_i] = local_sum
     ```

   - 효율적이고 올바른 구현:

     ```mojo
     if global_i < SIZE:
         var local_sum: output.element_type = 0  # var로 타입 추론 활용
         @parameter  # CONV가 상수이므로 컴파일 타임에 루프 전개
         for j in range(CONV):
             if local_i + j < SIZE:
                 local_sum += shared_a[local_i + j] * shared_b[j]
         output[global_i] = local_sum
     ```

   핵심적인 차이는 가드의 위치입니다. 비효율적 버전은 `global_i >= SIZE`인 스레드를 포함해 **모든 스레드가 convolution 연산을 수행**한 뒤, 마지막 쓰기에서만 가드를 적용합니다. 이로 인해:
   - **불필요한 연산**: 유효 범위 밖의 스레드가 쓸모없는 작업을 수행
   - **효율 저하**: 사용되지 않을 연산에 자원 소비
   - **GPU 활용도 저하**: 의미 없는 계산에 GPU 코어를 낭비

   효율적 버전은 유효한 `global_i` 값을 가진 스레드만 연산을 수행하므로 GPU 자원을 더 잘 활용합니다.

2. **주요 구현 특징**:
   - `var`와 `output.element_type`으로 적절한 타입 추론
   - `@parameter` 데코레이터로 convolution 루프를 컴파일 타임에 전개
   - 엄격한 경계 검사로 메모리 안전성 확보
   - LayoutTensor의 타입 시스템으로 코드 안전성 향상

3. **메모리 관리**:
   - 입력 배열과 kernel 모두 공유 메모리 사용
   - 스레드당 글로벌 메모리에서 1회 로드
   - 로드한 데이터의 효율적 재사용

4. **스레드 조율**:
   - `barrier()`로 모든 데이터 로드가 끝난 후 연산 시작을 보장
   - 각 스레드가 출력 원소 하나를 계산
   - 병합(coalesced) 메모리 접근 패턴 유지

5. **성능 최적화**:
   - 글로벌 메모리 접근 최소화
   - 공유 메모리로 빠른 데이터 접근
   - 메인 연산 루프에서 스레드 분기 회피
   - `@parameter` 데코레이터를 통한 루프 전개

</div>
</details>
