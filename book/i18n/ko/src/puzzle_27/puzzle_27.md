<!-- i18n-source-commit: 43fce1182f8029e7edc50157aed0e6ebb8129d42 -->

# Puzzle 27: 블록 전체 패턴

## 개요

**Puzzle 27: 블록 전체 패턴**에 오신 것을 환영합니다! 이 퍼즐은 GPU 병렬 프로그래밍의 핵심 구성 요소인 **블록 레벨 통신 기본 요소**를 소개합니다. 전체 스레드 블록에 걸친 고급 병렬 알고리즘을 구현할 수 있게 해주는 세 가지 핵심 통신 패턴을 탐구하며, 복잡한 수동 동기화를 간결하고 하드웨어에 최적화된 연산으로 대체합니다.

**목표:** 복잡한 공유 메모리 + barrier + 트리 reduction 패턴(Puzzle 12)에서 벗어나, 여러 Warp에 걸친 하드웨어 최적화 블록 전체 통신 기본 요소를 활용하는 간결한 단일 함수 호출 알고리즘으로 전환합니다.

**핵심 통찰:** _GPU 스레드 블록은 정교한 하드웨어 조율로 실행됩니다 - Mojo의 블록 연산은 크로스 Warp 통신과 전용 하드웨어 유닛을 활용하여 완벽한 병렬 프로그래밍 빌딩 블록을 제공합니다: reduction(전체→하나), scan(전체→각각), broadcast(하나→전체)._

## 배울 내용

### **블록 레벨 통신 모델**
GPU 스레드 블록 내 세 가지 기본 통신 패턴을 이해합니다:

```
GPU 스레드 블록 (128 스레드, 4개 또는 2개 Warp, 하드웨어 조율)
전체→하나 (Reduction):     모든 스레드 → 스레드 0에 단일 결과
전체→각각 (Scan):         모든 스레드 → 각 스레드가 누적 위치를 받음
하나→전체 (Broadcast):     스레드 0 → 모든 스레드가 같은 값을 받음

크로스 Warp 조율:
├── Warp 0 (스레드 0-31)   ──block.sum()──┐
├── Warp 1 (스레드 32-63)  ──block.sum()──┼→ 스레드 0 결과
├── Warp 2 (스레드 64-95)  ──block.sum()──┤
└── Warp 3 (스레드 96-127) ──block.sum()──┘
```

**하드웨어 현실:**
- **크로스 Warp 동기화**: 블록 내 여러 Warp 간 자동 조율
- **전용 하드웨어 유닛**: 특화된 scan 유닛과 butterfly reduction 네트워크
- **명시적 barrier 불필요**: 하드웨어가 모든 동기화를 내부적으로 관리
- **로그 복잡도**: \\(O(\\log n)\\) 알고리즘을 단일 명령의 단순함으로

### **Mojo의 블록 연산**
`gpu.primitives.block`의 완전한 병렬 프로그래밍 도구 모음을 배웁니다:

1. **[`block.sum(value)`](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/sum)**: 합계, 평균, 최댓값/최솟값을 위한 전체→하나 reduction
2. **[`block.prefix_sum(value)`](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/prefix_sum)**: 병렬 필터링과 추출을 위한 전체→각각 scan
3. **[`block.broadcast(value)`](https://docs.modular.com/mojo/stdlib/gpu/primitives/block/broadcast)**: 매개변수 공유와 조율을 위한 하나→전체 분배

> **참고:** 이 기본 요소들은 통계 연산, 히스토그램 구간 분류, 정규화 워크플로우와 같은 고급 병렬 알고리즘을 가능하게 합니다. 이런 알고리즘을 기본 요소 없이 구현하려면 수십 줄의 복잡한 공유 메모리 조율 코드가 필요합니다.

### **성능 변환 예시**
```mojo
# 복잡한 블록 전체 reduction (기존 방식 - Puzzle 12에서):
shared_memory[local_i] = my_value
barrier()
for stride in range(64, 0, -1):
    if local_i < stride:
        shared_memory[local_i] += shared_memory[local_i + stride]
    barrier()
if local_i == 0:
    output[block_idx.x] = shared_memory[0]

# 블록 연산으로 이 모든 복잡성을 제거:
my_partial = compute_local_contribution()
total = block.sum[block_size=128, broadcast=False](my_partial)  # 한 줄이면 끝!
if local_i == 0:
    output[block_idx.x] = total[0]
```

### **블록 연산이 빛나는 순간**
성능 특성을 이해합니다:

| 알고리즘 패턴 | 기존 방식 | 블록 연산 |
|-------------------|-------------|------------------|
| 블록 전체 reduction | 공유 메모리 + barrier | 단일 `block.sum` 호출 |
| 병렬 필터링 | 복잡한 인덱싱 | `block.prefix_sum` 조율 |
| 매개변수 공유 | 수동 동기화 | 단일 `block.broadcast` 호출 |
| 크로스 Warp 알고리즘 | 명시적 barrier 관리 | 하드웨어 관리 조율 |

## GPU 프로그래밍 패턴의 진화

### **출발점: 수동 조율 (Puzzle 12)**
복잡하지만 교육적 - 명시적 공유 메모리, barrier, 트리 reduction:
```mojo
# 수동 방식: 15줄 이상의 복잡한 동기화
shared_memory[local_i] = my_value
barrier()
# stride 기반 인덱싱을 사용한 트리 reduction...
for stride in range(64, 0, -1):
    if local_i < stride:
        shared_memory[local_i] += shared_memory[local_i + stride]
    barrier()
```

### **중간 단계: Warp 프로그래밍 (Puzzle 24)**
하드웨어 가속이지만 범위가 제한적 - 32 스레드 Warp 내의 `warp.sum()`:
```mojo
# Warp 방식: 1줄이지만 단일 Warp만
total = warp.sum[warp_size=WARP_SIZE](val=partial_product)
```

### **최종 목적지: 블록 프로그래밍 (이번 퍼즐)**
완전한 도구 모음 - 전체 블록에 걸친 하드웨어 최적화 기본 요소:
```mojo
# 블록 방식: 여러 Warp에 걸친 1줄 (128+ 스레드)
total = block.sum[block_size=128, broadcast=False](val=partial_product)
```

## 세 가지 기본 통신 패턴

블록 레벨 프로그래밍은 모든 병렬 통신 요구를 충족하는 세 가지 핵심 기본 요소를 제공합니다:

### **1. 전체→하나: Reduction (`block.sum()`)**
- **패턴**: 모든 스레드가 기여 → 하나의 스레드가 결과를 받음
- **용도**: 합계, 평균, 최댓값/최솟값 계산
- **예시**: 내적, 통계 집계
- **하드웨어**: 자동 barrier가 포함된 크로스 Warp butterfly reduction

### **2. 전체→각각: Scan (`block.prefix_sum()`)**
- **패턴**: 모든 스레드가 기여 → 각 스레드가 누적 위치를 받음
- **용도**: 병렬 필터링, stream compaction, 히스토그램 구간 분류
- **예시**: 병렬 데이터 추출을 위한 쓰기 위치 계산
- **하드웨어**: 크로스 Warp 조율을 포함한 병렬 scan

### **3. 하나→전체: Broadcast (`block.broadcast()`)**
- **패턴**: 하나의 스레드가 제공 → 모든 스레드가 같은 값을 받음
- **용도**: 매개변수 공유, 설정값 분배
- **예시**: 정규화 알고리즘을 위한 계산된 평균 공유
- **하드웨어**: 여러 Warp에 걸친 최적화된 분배


## 학습 경로

세 단계로 이 퍼즐을 완성하며, 단순한 것에서 복잡한 것으로 진행합니다:

### **Part 1: [block.sum()의 핵심](./block_sum.md)**
**복잡한 reduction을 단순한 함수 호출로 변환**

`block.sum()`으로 내적을 구현하며 블록 reduction의 기본 패턴을 배웁니다. 블록 연산이 15줄 이상의 수동 barrier를 단일 최적화 호출로 대체하는 방법을 보여줍니다.

**핵심 개념:**
- 여러 Warp에 걸친 블록 전체 동기화
- 하드웨어 최적화 reduction 패턴
- 스레드 0 결과 관리
- 기존 방식과의 성능 비교

**학습 목표:** `block.sum()`이 블록 규모에서 warp.sum()의 단순함을 제공하는 방법을 이해합니다.

---

### **Part 2: [block.prefix_sum()과 병렬 히스토그램 구간 분류](./block_prefix_sum.md)**
**고급 병렬 필터링과 추출**

히스토그램 구간 분류를 위해 `block.prefix_sum()`을 사용하여 고급 병렬 알고리즘을 구축합니다. prefix sum이 단순한 reduction으로는 구현하기 어려운 복잡한 데이터 재구성을 가능하게 하는 방법을 보여줍니다.

**핵심 개념:**
- 이진 프레디케이트를 이용한 병렬 필터링
- 조율된 쓰기 위치 계산
- 고급 파티셔닝 알고리즘
- 크로스 스레드 데이터 추출 패턴

**학습 목표:** `block.prefix_sum()`이 단순한 집계를 넘어서는 고급 병렬 알고리즘을 가능하게 하는 방법을 이해합니다.

---

### **Part 3: [block.broadcast()와 벡터 정규화](./block_broadcast.md)**
**모든 패턴을 결합하는 완전한 워크플로우**

블록 연산 도구 모음 전체를 사용하여 벡터 평균 정규화를 구현합니다. 세 가지 기본 요소가 어떻게 함께 작동하여 수학적 정확성을 갖춘 실제 연산 문제를 해결하는지 보여줍니다.

**핵심 개념:**
- 하나→전체 통신 패턴
- 조율된 다단계 알고리즘
- 완전한 블록 연산 워크플로우
- 실제 알고리즘 구현

**학습 목표:** 고급 병렬 알고리즘을 위해 블록 연산을 조합하는 방법을 이해합니다.

## 블록 연산이 중요한 이유

### **코드 단순화 변환:**
```
기존 방식:     20줄 이상의 barrier, 공유 메모리, 복잡한 인덱싱
블록 연산:     3-5줄의 조합 가능한 하드웨어 최적화 기본 요소
```

### **성능 이점:**
- **하드웨어 최적화**: GPU 아키텍처별 최적화를 활용
- **자동 동기화**: 수동 barrier 배치 오류 제거
- **조합 가능성**: 연산들이 매끄럽게 함께 동작
- **이식성**: 동일한 코드가 다양한 GPU 아키텍처에서 작동

### **교육적 가치:**
- **개념적 명확성**: 각 연산이 명확한 통신 목적을 가짐
- **점진적 복잡성**: 단순한 reduction에서 복잡한 알고리즘으로 발전
- **실제 응용**: 과학 연산, 그래픽, AI에서 광범위하게 사용되는 패턴

## 선수 지식

이 퍼즐을 시작하기 전에 다음을 완료해야 합니다:
- **[Puzzle 12: 내적](../puzzle_12/puzzle_12.md)**: 수동 GPU 동기화에 대한 이해
- **[Puzzle 24: Warp 기초](../puzzle_24/puzzle_24.md)**: Warp 레벨 프로그래밍 경험

## 학습 성과

세 파트를 모두 완료하면 다음을 이해하게 됩니다:

1. **각 블록 연산의 용도** - 다양한 병렬 통신 요구에 맞는 선택
2. **연산 조합 방법** - 고급 알고리즘 구축
3. **성능 트레이드오프** - 수동 방식과 자동화 방식 간의 비교
4. **실제 응용** - 블록 레벨 프로그래밍 패턴의 활용
5. **아키텍처 독립적 프로그래밍** - 하드웨어 최적화 기본 요소 활용

## 시작하기

**권장 순서:** 각 파트가 이전 파트의 개념을 기반으로 하므로 순서대로 완성하세요. 단순한 reduction → 고급 파티셔닝 → 완전한 워크플로우로 이어지는 진행이 블록 레벨 GPU 프로그래밍을 이해하는 최적의 학습 경로를 제공합니다.

💡 **핵심 통찰**: 블록 연산은 프로그래머 생산성과 하드웨어 성능 사이의 최적 지점을 나타냅니다 - 고수준 연산의 단순함과 세심하게 최적화된 저수준 구현의 효율성을 동시에 제공합니다. 이 퍼즐은 현대 GPU 프로그래밍에 적합한 추상화 수준에서 사고하는 법을 가르칩니다.
