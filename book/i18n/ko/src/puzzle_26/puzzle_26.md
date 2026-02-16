<!-- i18n-source-commit: 43fce1182f8029e7edc50157aed0e6ebb8129d42 -->

# Puzzle 26: 고급 Warp 패턴

## 개요

**Puzzle 26: 고급 Warp 통신 기본 요소**에서는 정교한 GPU **Warp 레벨 butterfly 통신과 병렬 scan 연산** - Warp 내에서 효율적인 트리 기반 알고리즘과 병렬 reduction을 가능하게 하는 하드웨어 가속 기본 요소를 소개합니다. [shuffle_xor](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_xor)을 사용한 butterfly 네트워크와 [prefix_sum](https://docs.modular.com/mojo/stdlib/gpu/warp/prefix_sum)을 사용한 하드웨어 최적화 병렬 scan을 배우며, 복잡한 다단계 공유 메모리 알고리즘 없이 이를 구현하는 방법을 익힙니다.

**달성 목표:** 복잡한 공유 메모리 + barrier + 다단계 reduction 패턴에서 벗어나, 하드웨어 최적화된 butterfly 네트워크와 병렬 scan 유닛을 활용하는 우아한 단일 함수 호출 알고리즘으로 전환합니다.

**핵심 통찰:** _GPU Warp는 하드웨어에서 정교한 트리 기반 통신과 병렬 scan 연산을 수행할 수 있습니다 - Mojo의 고급 Warp 기본 요소는 butterfly 네트워크와 전용 scan 유닛을 활용하여 \\(O(\\log n)\\) 알고리즘을 단일 명령 수준의 간결함으로 제공합니다._

## 배울 내용

### **고급 Warp 통신 모델**

GPU Warp 내 정교한 통신 패턴을 이해합니다:

```
GPU Warp Butterfly 네트워크 (32 스레드, XOR 기반 통신)
Offset 16: Lane 0 ↔ Lane 16, Lane 1 ↔ Lane 17, ..., Lane 15 ↔ Lane 31
Offset 8:  Lane 0 ↔ Lane 8,  Lane 1 ↔ Lane 9,  ..., Lane 23 ↔ Lane 31
Offset 4:  Lane 0 ↔ Lane 4,  Lane 1 ↔ Lane 5,  ..., Lane 27 ↔ Lane 31
Offset 2:  Lane 0 ↔ Lane 2,  Lane 1 ↔ Lane 3,  ..., Lane 29 ↔ Lane 31
Offset 1:  Lane 0 ↔ Lane 1,  Lane 2 ↔ Lane 3,  ..., Lane 30 ↔ Lane 31

하드웨어 Prefix Sum (병렬 scan 가속)
입력:  [1, 2, 3, 4, 5, 6, 7, 8, ...]
출력: [1, 3, 6, 10, 15, 21, 28, 36, ...] (inclusive scan)
```

**하드웨어 현실:**

- **Butterfly 네트워크**: XOR 기반 통신이 최적의 트리 토폴로지를 생성합니다
- **전용 scan 유닛**: 하드웨어 가속 병렬 prefix 연산
- **로그 복잡도**: \\(O(\\log n)\\) 알고리즘이 \\(O(n)\\) 순차 패턴을 대체합니다
- **단일 사이클 연산**: 복잡한 reduction이 전용 하드웨어에서 처리됩니다

### **Mojo의 고급 Warp 연산**

`gpu.primitives.warp`의 정교한 통신 기본 요소를 배웁니다:

1. **[`shuffle_xor(value, mask)`](https://docs.modular.com/mojo/stdlib/gpu/warp/shuffle_xor)**: 트리 알고리즘을 위한 XOR 기반 butterfly 통신
2. **[`prefix_sum(value)`](https://docs.modular.com/mojo/stdlib/gpu/warp/prefix_sum)**: 하드웨어 가속 병렬 scan 연산
3. **고급 조정 패턴**: 여러 기본 요소를 결합한 복잡한 알고리즘

> **참고:** 이 기본 요소들은 병렬 reduction, stream compaction, quicksort 파티셔닝, FFT 연산 등 공유 메모리 조정 코드가 수십 줄 필요했을 정교한 병렬 알고리즘을 가능하게 합니다.

### **성능 변환 예시**

```mojo
# 복잡한 병렬 reduction (기존 방식 - Puzzle 14 참고):
shared = LayoutTensor[
    dtype,
    Layout.row_major(WARP_SIZE),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
shared[local_i] = input[global_i]
barrier()
offset = 1
for i in range(Int(log2(Scalar[dtype](WARP_SIZE)))):
    var current_val: output.element_type = 0
    if local_i >= offset and local_i < WARP_SIZE:
        current_val = shared[local_i - offset]
    barrier()
    if local_i >= offset and local_i < WARP_SIZE:
        shared[local_i] += current_val
    barrier()
    offset *= 2

# 고급 Warp 기본 요소가 이 모든 복잡성을 제거합니다:
current_val = input[global_i]
scan_result = prefix_sum[exclusive=False](current_val)  # 단일 호출!
output[global_i] = scan_result
```

### **고급 Warp 연산이 빛나는 순간**

성능 특성을 이해합니다:

| 알고리즘 패턴 | 기존 방식 | 고급 Warp 연산 |
|------------------|-------------|-------------------------|
| 병렬 reduction | 공유 메모리 + barrier | 단일 `shuffle_xor` 트리 |
| Prefix/scan 연산 | 다단계 알고리즘 | 하드웨어 `prefix_sum` |
| Stream compaction | 복잡한 인덱싱 | `prefix_sum` + 조정 |
| Quicksort 파티션 | 수동 위치 계산 | 결합된 기본 요소 |
| 트리 알고리즘 | 재귀적 공유 메모리 | Butterfly 통신 |

## 선수 지식

고급 Warp 통신에 들어가기 전에 다음 내용에 익숙해야 합니다:

- **Part VII Warp 기초**: SIMT 실행과 기본 Warp 연산에 대한 이해 ([Puzzle 24: Warp 기초](../puzzle_24/puzzle_24.md)와 [Puzzle 25: Warp 통신](../puzzle_25/puzzle_25.md) 참고)
- **병렬 알고리즘 이론**: 트리 reduction, 병렬 scan, butterfly 네트워크
- **GPU 메모리 계층 구조**: 공유 메모리 패턴과 동기화 ([Puzzle 14: Prefix Sum](../puzzle_14/puzzle_14.md) 참고)
- **수학 연산**: XOR 연산과 로그 복잡도에 대한 이해

## 학습 경로

### **1. shuffle_xor을 이용한 butterfly 통신**

**→ [warp.shuffle_xor()와 Butterfly 네트워크](./warp_shuffle_xor.md)**

효율적인 트리 알고리즘과 병렬 reduction을 위한 XOR 기반 butterfly 통신 패턴을 배웁니다.

**배울 내용:**

- `shuffle_xor()`으로 butterfly 네트워크 토폴로지 구성하기
- 트리 통신을 활용한 \\(O(\\log n)\\) 병렬 reduction 구현
- XOR 기반 Lane 페어링과 통신 패턴 이해
- 다중 값 reduction을 위한 고급 조건부 butterfly 연산

**핵심 패턴:**

```mojo
max_val = input[global_i]
offset = WARP_SIZE // 2
while offset > 0:
    max_val = max(max_val, shuffle_xor(max_val, offset))
    offset //= 2
# 모든 Lane이 전역 최댓값을 가지게 됩니다
```

### **2. prefix_sum을 이용한 하드웨어 가속 병렬 scan**

**→ [warp.prefix_sum()과 Scan 연산](./warp_prefix_sum.md)**

복잡한 다단계 알고리즘을 단일 함수 호출로 대체하는 하드웨어 최적화 병렬 scan 연산을 배웁니다.

**배울 내용:**

- `prefix_sum()`을 활용한 하드웨어 가속 누적 연산
- Stream compaction과 병렬 파티셔닝 구현
- `prefix_sum`과 `shuffle_xor`을 결합한 고급 조정
- Inclusive vs exclusive scan 패턴 이해

**핵심 패턴:**

```mojo
current_val = input[global_i]
scan_result = prefix_sum[exclusive=False](current_val)
output[global_i] = scan_result  # 하드웨어 최적화 누적 합
```

## 핵심 개념

### **Butterfly 네트워크 통신**

XOR 기반 통신 토폴로지를 이해합니다:

- **XOR 페어링**: `lane_id ⊕ mask`가 대칭 통신 쌍을 생성합니다
- **트리 reduction**: 계층적 데이터 교환을 통한 로그 복잡도
- **병렬 조정**: 모든 Lane이 reduction에 동시에 참여합니다
- **동적 알고리즘**: 2의 거듭제곱 `WARP_SIZE` (32, 64 등) 어디서나 동작합니다

### **하드웨어 가속 병렬 scan**

전용 scan 유닛의 능력을 이해합니다:

- **Prefix sum 연산**: 하드웨어 가속을 활용한 누적 연산
- **Stream compaction**: 병렬 필터링과 데이터 재배치
- **단일 함수 간결성**: 복잡한 알고리즘이 단일 호출로 변환됩니다
- **동기화 불필요**: 하드웨어가 모든 조정을 내부적으로 처리합니다

### **알고리즘 복잡도 변환**

기존 패턴을 고급 Warp 연산으로 변환합니다:

- **순차 reduction** (\\(O(n)\\)) → **Butterfly reduction** (\\(O(\\log n)\\))
- **다단계 scan 알고리즘** → **단일 하드웨어 prefix_sum**
- **복잡한 공유 메모리 패턴** → **레지스터 전용 연산**
- **명시적 동기화** → **하드웨어 관리 조정**

### **고급 조정 패턴**

여러 기본 요소를 결합한 정교한 알고리즘:

- **이중 reduction**: Butterfly 패턴을 활용한 동시 min/max 추적
- **병렬 파티셔닝**: quicksort 스타일 연산을 위한 `shuffle_xor` + `prefix_sum`
- **조건부 연산**: 전역 조정을 통한 Lane 기반 출력 선택
- **다중 기본 요소 알고리즘**: 최적 성능의 복잡한 병렬 패턴

## 시작하기

고급 GPU Warp 레벨 통신을 활용할 준비가 되셨나요? Butterfly 네트워크 연산으로 트리 기반 통신을 이해한 다음, 하드웨어 가속 병렬 scan으로 나아가 최적의 알고리즘 성능을 달성하세요.

💡 **성공 팁**: 고급 Warp 연산을 **하드웨어 가속 병렬 알고리즘 빌딩 블록**으로 생각하세요. 이 기본 요소들은 복잡한 공유 메모리 알고리즘의 전체 범주를 단일 최적화 함수 호출로 대체합니다.

**학습 목표**: Puzzle 26을 마치면, 고급 Warp 기본 요소가 복잡한 다단계 알고리즘을 대체할 수 있는 상황을 인식하여 훨씬 간단하고 빠른 트리 기반 reduction, 병렬 scan, 조정 패턴을 작성할 수 있게 됩니다.

**시작하기**: **[warp.shuffle_xor()와 Butterfly 네트워크](./warp_shuffle_xor.md)** 에서 butterfly 통신을 배운 다음, **[warp.prefix_sum()과 Scan 연산](./warp_prefix_sum.md)** 에서 하드웨어 가속 병렬 scan 패턴으로 나아가세요!
