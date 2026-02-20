<!-- i18n-source-commit: 51143596e241ae5954474ecb3133b1d7b147f6fc -->

## 개요

1D LayoutTensor `a`에서 각 위치의 직전 3개 값의 합을 계산하여 1D LayoutTensor `output`에 저장하는 kernel을 구현하세요.

**참고:** _각 위치마다 스레드 1개가 있습니다. 스레드당 global read 1회, global write 1회만 필요합니다._

## 핵심 개념

이 퍼즐에서 배울 내용:

- LayoutTensor로 슬라이딩 윈도우 연산 구현하기
- [Puzzle 8](../puzzle_08/layout_tensor.md)에서 다룬 LayoutTensor 주소 공간(address_space)으로 공유 메모리 관리하기
- 효율적인 이웃 접근 패턴
- 경계 조건 처리

핵심은 LayoutTensor가 효율적인 윈도우 기반 연산은 유지하면서도 공유 메모리 관리를 간소화하는 방법입니다.

## 구성

- 배열 크기: `SIZE = 8`
- 블록당 스레드 수: `TPB = 8`
- 윈도우 크기: 3
- 공유 메모리: `TPB`개

참고:

- **LayoutTensor 할당**: `LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()` 사용
- **윈도우 접근**: 3개짜리 윈도우에 자연스러운 인덱싱
- **경계 처리**: 처음 두 위치는 특수 케이스
- **메모리 패턴**: 스레드당 공유 메모리 로드 1회

## 완성할 코드

```mojo
{{#include ../../../../../problems/p11/p11_layout_tensor.mojo:pooling_layout_tensor}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p11/p11_layout_tensor.mojo" class="filename">전체 파일 보기: problems/p11/p11_layout_tensor.mojo</a>

<details>
<summary><strong>팁</strong></summary>

<div class="solution-tips">

1. LayoutTensor와 주소 공간(address_space)으로 공유 메모리 생성
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
pixi run p11_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p11_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p11_layout_tensor
```

  </div>
  <div class="tab-content">

```bash
uv run poe p11_layout_tensor
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
{{#include ../../../../../solutions/p11/p11_layout_tensor.mojo:pooling_layout_tensor_solution}}
```

<div class="solution-explanation">

LayoutTensor를 활용한 슬라이딩 윈도우 합계 구현입니다. 주요 단계는 다음과 같습니다:

1. **공유 메모리 설정**
   - LayoutTensor가 주소 공간(address_space)으로 블록 로컬 저장소를 생성:

     ```txt
     shared = LayoutTensor[dtype, Layout.row_major(TPB), MutAnyOrigin, address_space = AddressSpace.SHARED].stack_allocation()
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

   - LayoutTensor의 자연스러운 인덱싱:

     ```txt
     # 3개짜리 슬라이딩 윈도우
     window_sum = shared[i-2] + shared[i-1] + shared[i]
     ```

4. **메모리 접근 패턴**
   - 스레드마다 공유 텐서로 global read 1회
   - 공유 메모리를 통한 효율적인 이웃 접근
   - LayoutTensor의 장점:
     - 자동 경계 검사
     - 자연스러운 윈도우 인덱싱
     - 레이아웃을 인식하는 메모리 접근
     - 전 과정에 걸친 타입 안전성

공유 메모리의 성능과 LayoutTensor의 안전성 및 편의성을 결합한 방식입니다:

- 글로벌 메모리 접근 최소화
- 윈도우 연산 간소화
- 깔끔한 경계 처리
- 병합(coalesced) 접근 패턴 유지

최종 출력은 누적 윈도우 합계입니다:

```txt
[0.0, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
```

</div>
</details>
