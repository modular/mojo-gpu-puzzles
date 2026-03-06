<!-- i18n-source-commit: f614177b516e68590fa807e66e31c9f20488c7e7 -->

## 왜 LayoutTensor를 고려해야 할까요?

아래 기존 구현을 보면 몇 가지 잠재적인 문제를 발견할 수 있습니다:

### 현재 방식

```mojo
i = thread_idx.x
output[i] = a[i] + 10.0
```

1D 배열에서는 잘 작동하지만, 다음과 같은 상황에서는 어떨까요?

- 2D나 3D 데이터를 다뤄야 할 때
- 다양한 메모리 레이아웃을 처리해야 할 때
- 병합(coalesced) 메모리 접근을 보장해야 할 때

### 앞으로의 도전 미리보기

퍼즐을 진행하면서 배열 인덱싱은 점점 복잡해집니다:

```mojo
# 이후 퍼즐에서 다룰 2D 인덱싱
idx = row * WIDTH + col

# 3D 인덱싱
idx = (batch * HEIGHT + row) * WIDTH + col

# 패딩이 있는 경우
idx = (batch * padded_height + row) * padded_width + col
```

### LayoutTensor 미리보기

[LayoutTensor](https://docs.modular.com/mojo/kernels/layout/layout_tensor/LayoutTensor/)를 사용하면 이런 경우를 훨씬 깔끔하게 처리할 수 있습니다:

```mojo
# 미리보기 - 지금은 이 문법을 몰라도 괜찮습니다!
output[i, j] = a[i, j] + 10.0  # 2D 인덱싱
output[b, i, j] = a[b, i, j] + 10.0  # 3D 인덱싱
```

Puzzle 4에서 LayoutTensor를 자세히 배울 예정입니다. 그때 이 개념들이 필수가 됩니다. 지금은 다음 내용을 이해하는 데 집중하세요:

- 기본 스레드 인덱싱
- 간단한 메모리 접근 패턴
- 스레드와 데이터의 일대일 매핑

💡 **핵심 포인트**: 직접 인덱싱은 간단한 경우에 잘 작동하지만, 복잡한 GPU 프로그래밍 패턴에서는 곧 더 정교한 도구가 필요해집니다.
