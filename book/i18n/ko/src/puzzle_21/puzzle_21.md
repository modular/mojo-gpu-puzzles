<!-- i18n-source-commit: 23f5ec0530b1cd15f85ce27e39f855a879987d36 -->

# Puzzle 21: Embedding Op

> ## 메모리 접근 패턴과 성능
>
> **메모리 바운드 연산**과 **GPU 메모리 접근 최적화**에 초점을 맞춰 Part V를 이어갑니다.
>
> [Puzzle 20: 1D Convolution Op](../puzzle_20/puzzle_20.md)에 이어, 동일한 연산의 서로 다른 커널 구현이 성능에 얼마나 극적인 차이를 가져올 수 있는지 알아봅니다. 배울 내용은 다음과 같습니다:
>
> - GPU 메모리 병합이 성능에 미치는 영향
> - 메모리 바운드 연산에서 그리드 구성이 중요한 이유
> - 최적의 메모리 접근 패턴으로 커널을 설계하는 방법
> - 서로 다른 스레딩 전략이 가져오는 성능 차이
>
> 이 퍼즐은 **어떤 연산을 수행하느냐**보다 **메모리에 어떻게 접근하느냐**가 더 중요할 수 있음을 보여줍니다.

## 개요

이 퍼즐에서는 신경망의 핵심 구성 요소인 embedding 연산을 위한 두 가지 GPU 커널을 구현합니다. 두 커널 모두 동일한 결과를 생성하지만, 서로 다른 메모리 접근 패턴을 사용하여 상당한 성능 차이를 보입니다.

비교할 두 커널:

- **1D coalesced 커널**: 연속적인 메모리 접근으로 메모리 대역폭에 최적화
- **2D non-coalesced 커널**: 비교를 위한 최적화되지 않은 메모리 접근 패턴

이 비교를 통해 GPU 커널 성능에서 메모리 병합이 얼마나 중요한지 체감할 수 있습니다.

## 배경: Embedding 연산

Embedding 연산은 이산적인 토큰 인덱스를 밀집 벡터 표현으로 변환합니다:

```python
# Input: token indices
indices = [[1, 5, 2], [7, 1, 9]]           # Shape: [batch_size, seq_len]

# Embedding table (learned parameters)
embedding_table = [                        # Shape: [vocab_size, embed_dim]
    [0.1, 0.2, 0.3, 0.4],  # Token 0
    [0.5, 0.6, 0.7, 0.8],  # Token 1
    [0.9, 1.0, 1.1, 1.2],  # Token 2
    # ... more tokens
]

# Output: embedded vectors
output[0,0] = embedding_table[1]  # [0.5, 0.6, 0.7, 0.8]
output[0,1] = embedding_table[5]  # lookup token 5's embedding
output[0,2] = embedding_table[2]  # [0.9, 1.0, 1.1, 1.2]
# ... and so on
```

이 연산은 **메모리 바운드**입니다. 성능은 embedding 테이블에서 얼마나 효율적으로 읽고 출력 텐서에 쓸 수 있느냐에 달려 있습니다.

## 학습 경로

이 퍼즐은 체계적인 이해를 위해 두 부분으로 구성되어 있습니다:

### **[Coalesced vs Non-Coalesced Kernel](./simple_embedding_kernel.md)**

여기서부터 시작하여 실제 퍼즐 코드를 구현하고 커널 구현을 이해합니다.

**무엇을 하게 될까요:**

- 두 가지 GPU embedding 커널 완성 (1D coalesced vs 2D non-coalesced)
- GPU 프로그래밍의 기본 메모리 접근 패턴 학습
- 동일한 알고리즘을 서로 다른 스레딩 전략으로 구현하는 사례 확인
- Mojo에서의 커스텀 연산 등록 이해

### **[성능 비교](./performance.md)**

커널 성능이 왜 다른지, 메모리 병합의 이론을 깊이 파고듭니다.

**무엇을 배울까요:**

- GPU 성능에서 메모리 병합이 중요한 이유
- 스레드 구성이 메모리 대역폭 활용에 미치는 영향
- 신경망 최적화에 대한 실제 시사점
- 메모리 바운드 연산을 위한 최적화 전략

## 시작하기

GPU 메모리 최적화를 탐구할 준비가 되셨나요? **[Coalesced vs Non-Coalesced Kernel](./simple_embedding_kernel.md)** 에서 코드를 구현한 후, **[성능 비교](./performance.md)** 로 넘어가 성능 차이의 원인을 이해해 보세요.

💡 **성공 팁:** 서로 다른 그리드 구성(1D vs 2D)이 메모리 접근 패턴에 어떤 영향을 미치는지 주의 깊게 살펴보세요. 이 통찰은 embedding을 넘어 다양한 GPU 프로그래밍 시나리오에 적용됩니다.
