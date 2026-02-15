<!-- i18n-source-commit: 9d44e8f2ab89f20eb789ee96c8ee86a0578245dd -->

# Puzzle 22: Kernel Fusion과 커스텀 역전파

> ## Kernel fusion과 autograd 통합
>
> **Kernel fusion**과 **autograd 통합**에 초점을 맞춰 Part IV를 이어갑니다.
>
> [Puzzle 21: Embedding Op](../puzzle_21/puzzle_21.md)에 이어, 여러 연산을 하나의 효율적인 커널로 결합하고 이를 PyTorch의 autograd 시스템과 통합하는 방법을 알아봅니다. 배울 내용은 다음과 같습니다:
>
> - Kernel fusion이 순전파와 역전파 모두에서 성능을 개선하는 원리
> - Fused 연산에 커스텀 역전파가 필수적인 이유
> - 적절한 기울기 흐름을 갖춘 fused 커널 설계 방법
> - 서로 다른 fusion 전략이 가져오는 성능 차이
>
> 이 퍼즐은 **연산을 어떻게 결합하느냐**가 **어떻게 구현하느냐**만큼 중요할 수 있음을 보여줍니다.

## 개요

이 퍼즐에서는 순전파와 역전파를 모두 포함하는 Fused LayerNorm + Linear 연산을 구현합니다. Fused와 unfused 구현 모두 동일한 결과를 생성하지만, 서로 다른 전략을 사용하여 상당한 성능 차이를 보입니다.

비교할 내용:

- **Unfused 방식**: LayerNorm과 Linear를 별도의 커널로 실행
- **Fused 커널**: 하나의 커널에서 두 연산을 결합하여 실행
- **커스텀 역전파**: Fused 연산을 위한 기울기 계산

이 비교를 통해 딥러닝 연산에서 kernel fusion과 적절한 기울기 계산이 얼마나 중요한지 체감할 수 있습니다.

## 배경: LayerNorm + Linear 연산

LayerNorm과 Linear는 트랜스포머 아키텍처의 핵심 연산으로, 특히 어텐션 메커니즘과 피드포워드 네트워크에서 빈번하게 사용됩니다. 일반적인 사용 방식은 다음과 같습니다:

```python
import torch
import torch.nn.functional as F

# Input: hidden states
x = torch.randn(batch_size, seq_len, hidden_dim)

# LayerNorm parameters
ln_weight = torch.ones(hidden_dim)  # scale parameter (γ)
ln_bias = torch.zeros(hidden_dim)   # shift parameter (β)

# Linear layer parameters
linear_weight = torch.randn(output_dim, hidden_dim)
linear_bias = torch.zeros(output_dim)

# Unfused operations (with autograd)
ln_output = F.layer_norm(x, [hidden_dim], weight=ln_weight, bias=ln_bias)
output = F.linear(ln_output, linear_weight, linear_bias)

# Fused operation (custom implementation)
# This is what you'll implement in this puzzle
output_fused = fused_layernorm_linear(x, ln_weight, ln_bias, linear_weight, linear_bias)
```

Fused 연산으로 결합하면 하나의 효율적인 커널에서 다음과 같은 이점을 얻을 수 있습니다:

- 메모리 대역폭 사용량 절감
- 커널 실행 오버헤드 최소화
- 캐시 활용도 향상
- 중간 결과 저장을 위한 메모리 할당 제거

실제로 이러한 fusion은 순전파와 역전파 모두에서 최대 1.5~2배의 속도 향상을 제공할 수 있으며, 이는 트랜스포머 학습 효율에 매우 중요합니다.

### 커스텀 역전파가 중요한 이유

PyTorch의 autograd 시스템은 개별 연산에 대한 기울기를 자동으로 계산하지만, fused 연산에는 다음과 같은 이유로 커스텀 역전파가 필요합니다:

- 수치 안정성 유지
- 적절한 기울기 흐름 보장
- 메모리 접근 패턴 최적화
- 기울기 누적을 위한 atomic 연산 처리

## 학습 경로

이 퍼즐은 체계적인 이해를 위해 두 부분으로 구성되어 있습니다:

### **[Fused vs Unfused Kernels](./forward_pass.md)**

여기서부터 시작하여 fused 순전파 커널을 구현하고 kernel fusion의 이점을 이해합니다.

**무엇을 하게 될까요:**

- Unfused와 fused 순전파 커널 모두 구현
- 핵심 kernel fusion 기법 학습
- 동일한 연산을 서로 다른 전략으로 구현하는 사례 확인
- Fusion이 가져오는 성능 차이 이해
- 최적 성능을 위한 메모리 접근 패턴 학습

### **[Autograd 통합과 역전파](./backward_pass.md)**

Autograd 통합과 기울기 계산을 깊이 파고듭니다.

**무엇을 배울까요:**

- 커스텀 역전파 구현 방법
- 적절한 기울기 흐름이 중요한 이유
- 학습 효율에 대한 실제 시사점
- 역전파 연산을 위한 최적화 전략
- 기울기 계산의 수학적 기초
- 기울기 누적을 위한 atomic 연산
- 역전파에서의 수치 안정성

## 시작하기

Kernel fusion과 autograd 통합을 탐구할 준비가 되셨나요? **[Fused vs Unfused Kernels](./forward_pass.md)** 에서 fused 커널을 구현한 후, **[Autograd 통합과 역전파](./backward_pass.md)** 로 넘어가 기울기 계산을 이해해 보세요.

이 퍼즐에는 다음을 검증하는 종합 테스트 프레임워크가 포함되어 있습니다:

- 순전파와 역전파 모두에서 PyTorch 구현과의 수치적 정확도
- CPU와 GPU 구현 간의 성능 비교
- 모든 파라미터(입력, LayerNorm 가중치/바이어스, Linear 가중치/바이어스)에 대한 기울기 계산 정확도
- Kernel fusion을 통한 메모리 사용량 최적화

💡 **성공 팁:** 서로 다른 구현 방식(fused vs unfused)이 순전파와 역전파 성능 모두에 어떤 영향을 미치는지 주의 깊게 살펴보세요. 이 통찰은 LayerNorm + Linear를 넘어 다양한 딥러닝 연산에 적용됩니다. 특히 역전파 구현은 학습 효율과 수치 안정성에 직접적인 영향을 미치므로 매우 중요합니다.
