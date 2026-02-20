<!-- i18n-source-commit: 224fad345fe6e71377c89cdc596f8e28d58a1fa4 -->

# 공유 메모리 버전

## 개요

정방 행렬 \\(A\\)와 \\(B\\)의 행렬 곱셈을 구현하고 결과를 \\(\text{output}\\)에 저장하는 퍼즐입니다. 공유 메모리를 활용하여 메모리 접근 패턴을 최적화합니다. 연산 전에 행렬 블록을 공유 메모리에 미리 로드하는 방식입니다.

## 핵심 개념

이 퍼즐에서 다루는 내용:

- LayoutTensor를 사용한 블록 로컬 메모리 관리
- 스레드 동기화 패턴
- 공유 메모리를 활용한 메모리 접근 최적화
- 2D 인덱싱을 사용한 협력적 데이터 로딩
- 행렬 연산에 LayoutTensor를 효율적으로 활용하기

핵심은 LayoutTensor를 통해 빠른 공유 메모리를 활용하여 비용이 큰 글로벌 메모리 접근을 최소화하는 것입니다.

## 구성

- 행렬 크기: \\(\\text{SIZE} \\times \\text{SIZE} = 2 \\times 2\\)
- 블록당 스레드 수: \\(\\text{TPB} \\times \\text{TPB} = 3 \\times 3\\)
- 그리드 차원: \\(1 \\times 1\\)

레이아웃 구성:

- 입력 A: `Layout.row_major(SIZE, SIZE)`
- 입력 B: `Layout.row_major(SIZE, SIZE)`
- 출력: `Layout.row_major(SIZE, SIZE)`
- 공유 메모리: `TPB × TPB` 크기의 LayoutTensor 2개

메모리 구성:

```txt
Global Memory (LayoutTensor):          Shared Memory (LayoutTensor):
A[i,j]: Direct access                  a_shared[local_row, local_col]
B[i,j]: Direct access                  b_shared[local_row, local_col]
```

## 완성할 코드

```mojo
{{#include ../../../../../problems/p16/p16.mojo:single_block_matmul}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p16/p16.mojo" class="filename">전체 파일 보기: problems/p16/p16.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. 전역 인덱스와 로컬 인덱스를 사용하여 행렬을 공유 메모리에 로드
2. 로드 후 `barrier()` 호출
3. 공유 메모리 인덱스를 사용하여 내적 계산
4. 모든 연산에서 배열 경계 검사

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
pixi run p16 --single-block
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p16 --single-block
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p16 --single-block
```

  </div>
  <div class="tab-content">

```bash
uv run poe p16 --single-block
```

  </div>
</div>

퍼즐을 아직 풀지 않았다면 출력은 다음과 같습니다:

```txt
out: HostBuffer([0.0, 0.0, 0.0, 0.0])
expected: HostBuffer([4.0, 6.0, 12.0, 22.0])
```

## 솔루션

<details class="solution-details">
<summary></summary>

```mojo
{{#include ../../../../../solutions/p16/p16.mojo:single_block_matmul_solution}}
```

<div class="solution-explanation">

LayoutTensor를 활용한 공유 메모리 구현은 효율적인 메모리 접근 패턴을 통해 성능을 향상시킵니다:

### 메모리 구성

```txt
Input Tensors (2×2):                Shared Memory (3×3):
Matrix A:                           a_shared:
 [a[0,0] a[0,1]]                     [s[0,0] s[0,1] s[0,2]]
 [a[1,0] a[1,1]]                     [s[1,0] s[1,1] s[1,2]]
                                     [s[2,0] s[2,1] s[2,2]]
Matrix B:                           b_shared: (동일한 레이아웃)
 [b[0,0] b[0,1]]                     [t[0,0] t[0,1] t[0,2]]
 [b[1,0] b[1,1]]                     [t[1,0] t[1,1] t[1,2]]
                                     [t[2,0] t[2,1] t[2,2]]
```

### 구현 단계

1. **공유 메모리 설정**:

   ```mojo
   # address_space를 지정한 LayoutTensor로 2D 공유 메모리 tensor 생성
   a_shared = LayoutTensor[dtype, Layout.row_major(TPB, TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   b_shared = LayoutTensor[dtype, Layout.row_major(TPB, TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
   ```

2. **스레드 인덱싱**:

   ```mojo
   # 행렬 접근을 위한 전역 인덱스
   row = block_dim.y * block_idx.y + thread_idx.y
   col = block_dim.x * block_idx.x + thread_idx.x

   # 공유 메모리용 로컬 인덱스
   local_row = thread_idx.y
   local_col = thread_idx.x
   ```

3. **데이터 로딩**:

   ```mojo
   # LayoutTensor 인덱싱으로 데이터를 공유 메모리에 로드
   if row < size and col < size:
       a_shared[local_row, local_col] = a[row, col]
       b_shared[local_row, local_col] = b[row, col]
   ```

4. **공유 메모리를 사용한 연산**:

   ```mojo
   # 가드로 유효한 행렬 원소만 계산
   if row < size and col < size:
       # 출력 tensor의 타입으로 누적 변수 초기화
       var acc: output.element_type = 0

       # 컴파일 타임에 전개되는 행렬 곱셈 루프
       @parameter
       for k in range(size):
           acc += a_shared[local_row, k] * b_shared[k, local_col]

       # 행렬 경계 내의 스레드만 결과 기록
       output[row, col] = acc
   ```

   주요 포인트:
   - **경계 검사**: `if row < size and col < size`
     - 범위 밖 연산 방지
     - 유효한 스레드만 작업 수행
     - TPB (3×3) > SIZE (2×2)이므로 필수

   - **누적 변수 타입**: `var acc: output.element_type`
     - 출력 tensor의 원소 타입으로 타입 안전성 확보
     - 일관된 수치 정밀도 보장
     - 누적 전에 0으로 초기화

   - **루프 최적화**: `@parameter for k in range(size)`
     - 컴파일 타임에 루프 전개
     - 더 나은 명령어 스케줄링 가능
     - 크기가 작고 미리 알려진 행렬에 효과적

   - **결과 기록**: `output[row, col] = acc`
     - 동일한 가드 조건으로 보호
     - 유효한 스레드만 결과 기록
     - 행렬 경계 안전성 유지

### 스레드 안전성과 동기화

1. **가드 조건**:
   - 입력 로딩: `if row < size and col < size`
   - 연산: 동일한 가드로 스레드 안전성 보장
   - 출력 기록: 같은 조건으로 보호
   - 잘못된 메모리 접근과 경쟁 상태 방지

2. **메모리 접근 안전성**:
   - 공유 메모리: TPB 범위 내에서만 접근
   - 글로벌 메모리: 크기 검사로 보호
   - 출력: 가드된 쓰기로 데이터 손상 방지

### 주요 언어 기능

1. **LayoutTensor의 장점**:
   - 직접 2D 인덱싱으로 코드 단순화
   - `element_type`을 통한 타입 안전성
   - 효율적인 메모리 레이아웃 처리

2. **공유 메모리 할당**:
   - address_space를 지정한 LayoutTensor로 구조화된 할당
   - 입력 tensor와 동일한 row-major 레이아웃
   - 효율적 접근을 위한 적절한 메모리 정렬

3. **동기화**:
   - `barrier()`로 공유 메모리 일관성 보장
   - 로드와 연산 간 적절한 동기화
   - 블록 내 스레드 간 협력

### 성능 최적화

1. **메모리 접근 효율**:
   - 원소당 글로벌 메모리 로드 1회
   - 공유 메모리를 통한 다중 재사용
   - 병합된(coalesced) 메모리 접근 패턴

2. **스레드 협력**:
   - 협력적 데이터 로딩
   - 공유 데이터 재사용
   - 효율적인 스레드 동기화

3. **연산 이점**:
   - 글로벌 메모리 트래픽 감소
   - 캐시 활용도 향상
   - 명령어 처리량 개선

이 구현은 다음을 통해 기본 버전 대비 성능을 크게 향상시킵니다:

- 글로벌 메모리 접근 횟수 감소
- 공유 메모리를 통한 데이터 재사용
- LayoutTensor의 효율적인 2D 인덱싱 활용
- 적절한 스레드 동기화 유지

</div>
</details>
