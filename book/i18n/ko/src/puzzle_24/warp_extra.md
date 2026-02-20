<!-- i18n-source-commit: 43fce1182f8029e7edc50157aed0e6ebb8129d42 -->

# 언제 Warp 프로그래밍을 사용할까

## 빠른 판단 가이드

**✅ Warp 연산을 사용할 때:**

- 32개 이상의 요소에 대한 reduction 연산 (`sum`, `max`, `min`)
- 규칙적인 메모리 접근 패턴 (인접 Lane → 인접 주소)
- 크로스 아키텍처 이식성이 필요한 경우 (NVIDIA/RDNA 32 vs CDNA 64 스레드)
- 더 간단하고 유지보수하기 쉬운 코드를 원할 때

**❌ 기존 방식을 사용할 때:**

- 복잡한 Warp 간 동기화가 필요한 경우
- 불규칙하거나 산발적인 메모리 접근 패턴
- 스레드별 작업량이 다른 경우 (Warp 분기 발생)
- 문제 크기가 `size < WARP_SIZE`인 경우

## 성능 특성

### 문제 크기별 확장성

| 요소 수 | Warp 이점 | 비고 |
|----------|---------------|-------|
| < 32 | 없음 | 기존 방식이 유리 |
| 32-1K | 1.2-1.5배 | 이점이 나타나기 시작 |
| 1K-32K | 1.5-2.5배 | **Warp 연산이 탁월** |
| > 32K | 메모리 바운드 | 양쪽 모두 대역폭에 의해 제한 |

### Warp의 핵심 이점

- **동기화 오버헤드 제로**: barrier 비용 제거
- **최소한의 메모리 사용**: 공유 메모리 할당 불필요
- **우수한 확장성**: Warp 수가 늘어날수록 성능 향상
- **간결한 코드**: 더 적은 줄 수, 더 적은 오류 가능성

## 알고리즘별 가이드

| 알고리즘 | 권장 사항 | 이유 |
|-----------|---------------|--------|
| **내적** | Warp 연산 (1K+ 요소) | 단일 reduction, 규칙적 접근 |
| **행렬 행/열 합계** | Warp 연산 | 자연스러운 reduction 패턴 |
| **Prefix sum** | 항상 `prefix_sum()` 사용 | 하드웨어 최적화된 기본 요소 |
| **Pooling (max/min)** | Warp 연산 (규칙적 윈도우) | 효율적인 윈도우 reduction |
| **구간이 많은 히스토그램** | 기존 방식 | 불규칙한 쓰기, atomic 업데이트 |

## 코드 예시

### ✅ Warp에 적합한 경우

```mojo
# Reduction 연산
from gpu.primitives.warp import sum, max
var total = sum(partial_values)
var maximum = max(partial_values)

# 통신 패턴
from gpu.primitives.warp import shuffle_idx, prefix_sum
var broadcast = shuffle_idx(my_value, 0)
var running_sum = prefix_sum(my_value)
```

### ❌ 기존 방식이 나은 경우

```mojo
# 복잡한 다단계 동기화
stage1_compute()
barrier()  # 모든 스레드가 완료될 때까지 대기
stage2_depends_on_stage1()

# 불규칙한 메모리 접근
var value = input[random_indices[global_i]]  # 산발적 읽기

# 데이터 의존적 작업
if input[global_i] > threshold:
    result = expensive_computation()  # Warp 분기 발생
```

## 성능 측정

```bash
# 항상 양쪽 방식을 벤치마크하세요
mojo p22.mojo --benchmark

# 확장 패턴을 확인하세요:
# traditional_1x:  X.XX ms
# warp_1x:         Y.YY ms  # 더 빨라야 함
# warp_32x:        Z.ZZ ms  # 이점이 커져야 함
```

## 요약

**Warp 연산으로 시작하세요:**

- 규칙적인 접근 패턴을 가진 reduction
- 문제 ≥ 1 Warp 크기
- 크로스 플랫폼 호환성이 필요한 경우

**기존 방식을 사용하세요:**

- 복잡한 동기화가 필요한 경우
- 불규칙한 메모리 패턴
- 작은 문제 또는 심한 분기

**판단이 어려울 때:** 양쪽 모두 구현하고 벤치마크하세요. 성능 차이를 보면 답이 나옵니다.
