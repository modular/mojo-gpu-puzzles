<!-- i18n-source-commit: 23f5ec0530b1cd15f85ce27e39f855a879987d36 -->

# 성능: Coalesced vs Non-coalesced 메모리 접근

메모리 접근 패턴을 이해하는 것은 GPU 성능 최적화의 핵심입니다. 이 섹션에서는 embedding 조회와 같은 메모리 바운드 연산에서 coalesced 메모리 접근 패턴이 왜 non-coalesced 패턴보다 뛰어난 성능을 보이는지 설명합니다.

## 메모리 병합 기초

**메모리 병합**은 Warp 내 연속된 스레드가 연속된 메모리 주소에 접근할 때 발생합니다. GPU는 이러한 개별 메모리 요청을 더 적은 수의 대용량 메모리 트랜잭션으로 결합하여 대역폭 활용도를 크게 향상시킵니다.

### Coalesced vs Non-coalesced 접근

**Coalesced (효율적):**

```
- Thread 0 → Address 0x1000
- Thread 1 → Address 0x1004
- Thread 2 → Address 0x1008
- Thread 3 → Address 0x100C
- ...
```

**결과**: Warp 전체(32개 스레드)에 대해 1번의 메모리 트랜잭션

**Non-coalesced (비효율적):**

```
- Thread 0 → Address 0x1000
- Thread 1 → Address 0x2000
- Thread 2 → Address 0x3000
- Thread 3 → Address 0x4000
- ...
```

**결과**: 최대 32번의 개별 메모리 트랜잭션

## Embedding 연산이 메모리 바운드인 이유

Embedding 조회는 다음과 같은 특성 때문에 **메모리 바운드**입니다:

- **최소한의 연산**: 하는 일이라곤 입력 데이터를 출력으로 복사하는 것뿐
- **큰 메모리 풋프린트**: Embedding 테이블은 수 기가바이트에 달할 수 있음
- **높은 메모리 대역폭 요구**: 대량의 데이터 전송이 필요

이러한 연산에서는 연산 복잡도보다 **메모리 접근 효율**이 성능을 결정합니다.

## 커널 비교

### 1D Coalesced 커널

- **스레드 구성**: `[total_elements // 256]` 블록, 출력 요소당 하나의 스레드
- **메모리 패턴**: 연속된 스레드가 연속된 embedding 차원에 접근
- **왜 coalesced인가**: `Thread 0: output[0,0,0]`, `Thread 1: output[0,0,1]` → 연속된 주소

### 2D Non-coalesced 커널

- **스레드 구성**: `[batch*seq // 16, embed_dim // 16]` 블록, 16×16 스레드
- **메모리 패턴**: 스레드들이 서로 다른 embedding 벡터에 접근할 수 있음
- **왜 non-coalesced인가**: 스레드 접근 패턴이 메모리 전체에 흩어질 수 있음

## 성능 결과

일반적인 벤치마크 결과:

```
Performance Results:
   1D Coalesced:     2.145 ms
   2D Non-coalesced: 3.867 ms
   1D is 1.80x faster than 2D
```

## 메모리 접근 시각화

### Coalesced 패턴 (1D 커널)

**output[0,0,0:32]에 대한 Warp 실행:**

| 요소 | 스레드 ID | 메모리 접근 | 주소 패턴 |
|---------|-----------|---------------|-----------------|
| `output[0,0,0]` | 0 | `[0,0]` | Base + 0 |
| `output[0,0,1]` | 1 | `[0,1]` | Base + 4 |
| `output[0,0,2]` | 2 | `[0,2]` | Base + 8 |
| `output[0,0,3]` | 3 | `[0,3]` | Base + 12 |
| ... | ... | ... | ... |
| `output[0,0,31]` | 31 | `[0,31]` | Base + 124 |

**결과**: 연속된 주소 → Warp 전체에 대해 **1번의 메모리 트랜잭션**

### Non-coalesced 패턴 (2D 커널)

**16×16 블록의 Warp 실행:**

```
Block organization (16×16):
    X-dim: batch*seq positions (0-15)
    Y-dim: embed dimensions (0-15)

Warp threads might access:
    Thread 0:  batch=0, seq=0, embed=0  → Address A
    Thread 1:  batch=0, seq=1, embed=0  → Address B (different row)
    Thread 2:  batch=0, seq=2, embed=0  → Address C (different row)
    ...
    Thread 31: batch=1, seq=15, embed=0 → Address Z (scattered)
```

**결과**: 흩어진 주소 → **여러 번의 메모리 트랜잭션**

## 핵심 최적화 전략

1. 메모리 바운드 연산에서는 가능한 한 **1D 인덱싱을 선호**하세요
2. 병합에 유리하도록 **데이터 구조를 정렬**하세요
3. 커널 설계 시 **메모리 접근 패턴을 고려**하세요
4. 병목 지점을 파악하기 위해 **메모리 대역폭을 프로파일링**하세요
5. 최적화 효과를 검증하기 위해 **메모리 바운드 벤치마크를 활용**하세요

핵심 통찰: 특히 embedding과 같은 메모리 바운드 연산에서는 연산 복잡도보다 **메모리 접근 패턴**이 GPU 성능을 결정하는 경우가 많습니다.
