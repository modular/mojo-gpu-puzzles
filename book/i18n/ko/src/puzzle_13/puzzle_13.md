<!-- i18n-source-commit: c851c500cec0955e101b5dae8db281cced543065 -->

# Puzzle 13: 1D Convolution

> ## LayoutTensor로 전환하기
>
> 지금까지 GPU 퍼즐 여정에서 GPU 메모리 관리에 대한 두 가지 접근 방식을 함께 살펴보았습니다:
>
> 1. [UnsafePointer](https://docs.modular.com/mojo/stdlib/memory/unsafe_pointer/UnsafePointer/)를 사용한 포인터 직접 조작 방식의 raw 메모리 관리
> 2. 강력한 주소 공간(address_space) 파라미터로 메모리를 할당하는, 보다 구조화된 [LayoutTensor](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/)
>
> 이 퍼즐부터는 `LayoutTensor`로 완전히 전환합니다. 이 추상화는 다음과 같은 이점을 제공합니다:
>
> - 타입 안전한 메모리 접근 패턴
> - 데이터 레이아웃의 명확한 표현
> - 코드 유지보수성 향상
> - 메모리 관련 버그 발생 가능성 감소
> - 내부 연산의 의도를 더 잘 드러내는 표현력 있는 코드
> - 앞으로 차차 알아갈 더 많은 것들!
>
> 이러한 전환은 Mojo 🔥의 현대적 GPU 프로그래밍 모범 사례와 맞닿아 있습니다. 높은 수준의 추상화로 복잡성을 관리하면서도 성능은 그대로 유지할 수 있습니다.

## 개요

신호 처리와 이미지 분석에서 convolution은 두 시퀀스를 결합해 새로운 시퀀스를 만들어내는 핵심 연산입니다. 이 퍼즐에서는 입력 배열 위로 kernel을 슬라이딩하면서 각 출력 원소를 계산하는 1D convolution을 GPU에서 구현해 봅니다.

`LayoutTensor` 추상화를 사용하여 벡터 `a`와 벡터 `b`의 1D convolution을 계산하고, 결과를 `output`에 저장하는 kernel을 구현하세요.

**참고:** _일반적인 경우를 처리해야 합니다. 스레드당 global read 2회, global write 1회만 필요합니다._

<img src="/puzzle_13/media/13-w.gif" alt="1D convolution 시각화" class="light-mode-img">
<img src="/puzzle_13/media/13-b.gif" alt="1D convolution 시각화" class="dark-mode-img">

Convolution이 처음이라면, 가중치가 적용된 슬라이딩 윈도우 연산이라고 생각하면 됩니다. 각 위치에서 kernel 값과 대응하는 입력 값을 곱한 뒤 합산합니다. 수학적 표기로는 다음과 같습니다:

\\[\Large output[i] = \sum_{j=0}^{\text{CONV}-1} a[i+j] \cdot b[j] \\]

의사 코드로 표현한 1D convolution:

```python
for i in range(SIZE):
    for j in range(CONV):
        if i + j < SIZE:
            ret[i] += a_host[i + j] * b_host[j]
```

이 퍼즐은 단계적으로 이해를 쌓아갈 수 있도록 두 파트로 나뉩니다:

- [🔰 기본 버전](./simple.md)
  여기서부터 시작하세요. 단일 블록에서 LayoutTensor와 공유 메모리를 활용한 convolution 구현의 기초를 익힙니다.

- [⭐ 블록 경계 버전](./block_boundary.md)
  이어서 블록 경계를 넘어 데이터를 공유해야 하는 더 까다로운 경우에 도전합니다. LayoutTensor의 기능을 본격적으로 활용합니다.

각 버전은 메모리 접근 패턴과 스레드 간 협력 측면에서 서로 다른 도전 과제를 제시합니다. 기본 버전에서 convolution 연산의 원리를 익힌 다음, 블록 경계 버전에서는 실제 GPU 프로그래밍에서 마주치는 복잡한 상황을 다루는 능력을 시험해 봅니다.
