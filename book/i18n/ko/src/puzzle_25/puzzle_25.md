<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# Puzzle 25: 워프 통신

## 개요

**Puzzle 25: 워프 통신 기본 요소**에서는 고급 GPU **워프 레벨 통신 연산** - 워프 내에서 효율적인 데이터 교환과 조정 패턴을 가능하게 하는 하드웨어 가속 기본 요소를 소개합니다. [shuffle_down](https://docs.modular.com/mojo/std/gpu/primitives/warp/shuffle_down)과 [broadcast](https://docs.modular.com/mojo/std/gpu/primitives/warp/broadcast)를 사용하여 복잡한 공유 메모리 패턴 없이 이웃 통신과 집합 조정을 구현하는 방법을 배웁니다.

**Part VII: GPU 워프 통신**에서는 스레드 그룹 내 워프 레벨 데이터 이동 연산을 다룹니다. 복잡한 공유 메모리 + 인덱싱 + 경계 검사 패턴을 하드웨어 최적화된 데이터 이동을 활용하는 효율적인 워프 통신 호출로 대체하는 방법을 배웁니다.

**핵심 통찰:** _GPU 워프는 록스텝으로 실행됩니다 - Mojo의 워프 통신 연산은 이 동기화를 활용하여 자동 경계 처리와 명시적 동기화 없이 효율적인 데이터 교환 기본 요소를 제공합니다._

## 배울 내용

### **워프 통신 모델**

GPU 워프 내 기본 통신 패턴을 이해합니다:

```
GPU 워프 (32 스레드, SIMT 록스텝 실행)
├── 레인 0  ──shuffle_down──> 레인 1  ──shuffle_down──> 레인 2
├── 레인 1  ──shuffle_down──> 레인 2  ──shuffle_down──> 레인 3
├── 레인 2  ──shuffle_down──> 레인 3  ──shuffle_down──> 레인 4
│   ...
└── 레인 31 ──shuffle_down──> undefined (경계)

브로드캐스트 패턴:
레인 0 ──broadcast──> 모든 레인 (0, 1, 2, ..., 31)
```

**하드웨어 현실:**

- **레지스터 간 직접 통신**: 데이터가 스레드 레지스터 사이를 직접 이동합니다
- **메모리 오버헤드 제로**: 공유 메모리 할당이 필요하지 않습니다
- **자동 경계 처리**: 하드웨어가 워프 경계의 예외 상황을 관리합니다
- **단일 사이클 연산**: 하나의 명령 사이클에서 통신이 완료됩니다

### **Mojo의 워프 통신 연산**

`gpu.primitives.warp`의 핵심 통신 기본 요소를 배웁니다:

1. **[`shuffle_down(value, offset)`](https://docs.modular.com/mojo/std/gpu/primitives/warp/shuffle_down)**: 더 높은 인덱스의 레인에서 값을 가져오기 (이웃 접근)
2. **[`broadcast(value)`](https://docs.modular.com/mojo/std/gpu/primitives/warp/broadcast)**: 레인 0의 값을 모든 레인에 공유 (일대다)
3. **[`shuffle_idx(value, lane)`](https://docs.modular.com/mojo/std/gpu/primitives/warp/shuffle_idx)**: 특정 레인에서 값을 가져오기 (임의 접근)
4. **[`shuffle_up(value, offset)`](https://docs.modular.com/mojo/std/gpu/primitives/warp/shuffle_up)**: 더 낮은 인덱스의 레인에서 값을 가져오기 (역방향 이웃)

> **참고:** 이 퍼즐은 가장 많이 사용되는 통신 패턴인 `shuffle_down()`과 `broadcast()`에 초점을 맞춥니다. 모든 워프 연산에 대한 전체 내용은 [Mojo GPU 워프 문서](https://docs.modular.com/mojo/std/gpu/primitives/warp/)를 참고하세요.

### **성능 변환 예시**

```mojo
# 복잡한 이웃 접근 패턴 (기존 방식):
shared = TileTensor[
    dtype,
    row_major[WARP_SIZE](),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
].stack_allocation()
shared[local_i] = input[global_i]
barrier()
if local_i < WARP_SIZE - 1:
    next_value = shared[local_i + 1]  # 이웃 접근
    result = next_value - shared[local_i]
else:
    result = 0  # 경계 처리
barrier()

# 워프 통신은 이 모든 복잡성을 제거합니다:
current_val = input[global_i]
next_val = shuffle_down(current_val, 1)  # 이웃에 직접 접근
if lane < WARP_SIZE - 1:
    result = next_val - current_val
else:
    result = 0
```

### **워프 통신이 빛나는 순간**

성능 특성을 이해합니다:

| 통신 패턴 | 기존 방식 | 워프 연산 |
|----------------------|-------------|-----------------|
| 이웃 접근 | 공유 메모리 | 레지스터 간 직접 통신 |
| 스텐실 연산 | 복잡한 인덱싱 | 간단한 셔플 패턴 |
| 블록 조정 | 배리어 + 공유 메모리 | 단일 브로드캐스트 |
| 경계 처리 | 수동 검사 | 하드웨어 자동 처리 |

## 선수 지식

워프 통신에 들어가기 전에 다음 내용에 익숙해야 합니다:

- **Part VII 워프 기초**: SIMT 실행과 기본 워프 연산에 대한 이해 ([Puzzle 24: 워프 기초](../puzzle_24/puzzle_24.md) 참고)
- **GPU 스레드 계층 구조**: 블록, 워프, 레인 번호 매기기
- **TileTensor 연산**: 로드, 저장, 텐서 조작
- **경계 조건 처리**: 병렬 알고리즘의 가장자리 케이스 관리

## 학습 경로

### **1. shuffle_down을 이용한 이웃 통신**

**→ [warp.shuffle_down()](./warp_shuffle_down.md)**

스텐실 연산과 유한 차분을 위한 이웃 기반 통신 패턴을 배웁니다.

**배울 내용:**

- `shuffle_down()`으로 인접 레인 데이터 접근하기
- 유한 차분과 이동 평균 구현
- 워프 경계 자동 처리
- 확장된 이웃 접근을 위한 다중 오프셋 셔플

**핵심 패턴:**

```mojo
current_val = input[global_i]
next_val = shuffle_down(current_val, 1)
if lane < WARP_SIZE - 1:
    result = compute_with_neighbors(current_val, next_val)
```

### **2. 브로드캐스트를 이용한 집합 조정**

**→ [warp.broadcast()](./warp_broadcast.md)**

블록 레벨 조정과 집합적 의사결정을 위한 일대다 통신 패턴을 배웁니다.

**배울 내용:**

- `broadcast()`로 계산된 값을 모든 레인에 공유
- 블록 레벨 통계와 집합적 의사결정 구현
- 브로드캐스트와 조건부 로직 결합
- 고급 브로드캐스트-셔플 조정 패턴

**핵심 패턴:**

```mojo
var shared_value = 0.0
if lane == 0:
    shared_value = compute_block_statistic()
shared_value = broadcast(shared_value)
result = use_shared_value(shared_value, local_data)
```

## 핵심 개념

### **통신 패턴**

워프 통신의 기본 패러다임을 이해합니다:

- **이웃 통신**: 레인 간 인접 데이터 교환
- **집합 조정**: 하나의 레인에서 모든 레인으로 정보 공유
- **스텐실 연산**: 고정된 패턴으로 이웃 데이터 접근
- **경계 처리**: 워프 가장자리에서의 통신 관리

### **하드웨어 최적화**

워프 통신이 GPU 하드웨어에 매핑되는 방식을 이해합니다:

- **레지스터 파일 통신**: 스레드 간 레지스터 직접 접근
- **SIMT 실행**: 모든 레인이 통신을 동시에 실행합니다
- **제로 지연 시간**: 실행 유닛 내에서 통신이 완료됩니다
- **자동 동기화**: 명시적 배리어가 필요하지 않습니다

### **알고리즘 변환**

기존 병렬 패턴을 워프 통신으로 변환합니다:

- **배열 이웃 접근** → `shuffle_down()`
- **공유 메모리 조정** → `broadcast()`
- **복잡한 경계 로직** → 하드웨어 자동 처리
- **다단계 동기화** → 단일 통신 연산

## 시작하기

이웃 기반 셔플 연산으로 기초를 다진 다음, 고급 조정을 위한 집합 브로드캐스트 패턴으로 나아갑니다.

💡 **성공 팁**: 워프 통신을 같은 워프 내 스레드 간의 **하드웨어 가속 메시지 패싱**으로 생각하세요. 이 멘탈 모델이 GPU의 SIMT 아키텍처를 활용하는 효율적인 통신 패턴으로 안내할 것입니다.

**학습 목표**: Puzzle 25를 마치면, 워프 통신이 복잡한 공유 메모리 패턴을 대체할 수 있는 상황을 인식하여 더 간단하고 빠른 이웃 기반 알고리즘과 조정 알고리즘을 작성할 수 있게 됩니다.

**시작하기**: **[warp.shuffle_down()](./warp_shuffle_down.md)** 에서 이웃 통신을 배운 다음, **[warp.broadcast()](./warp_broadcast.md)** 에서 집합 조정 패턴으로 나아가세요.
