<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# Puzzle 20: 1D Convolution Op

> ## MAX Graph에서 PyTorch 커스텀 Op으로
>
> GPU 퍼즐 여정의 Part V에 진입했습니다: **PyTorch 커스텀 Op 통합하기**.
>
> [Puzzle 17: 1D Convolution Op](../puzzle_17/puzzle_17.md)에서 MAX Graph를 사용하여 Mojo GPU 커널을 Python과 연동하는 방법을 배웠습니다. 이제부터는 다음을 알아봅니다:
>
> - 동일한 Mojo 커널을 PyTorch의 CustomOpLibrary로 사용하기
> - PyTorch의 텐서 시스템 및 autograd와 통합하기
> - MAX Graph와 PyTorch 방식의 커스텀 연산 비교하기
> - 명시적 출력 텐서 할당이라는 핵심 패턴 이해하기
>
> 이 전환을 통해 동일한 최적화된 GPU 커널이 서로 다른 Python 통합 방식에서 어떻게 동작하는지 확인할 수 있습니다.

## 개요

이 퍼즐에서는 [Puzzle 17: 1D Convolution Op](../puzzle_17/puzzle_17.md)의 1D convolution kernel을 그대로 가져와서, MAX Graph 대신 [CustomOpLibrary](https://docs.modular.com/max/api/python/torch/)를 사용하여 PyTorch와 통합합니다.

여기서 핵심은 **동일한 Mojo 커널이 수정 없이 그대로 동작한다**는 것입니다. MAX Graph와 PyTorch 방식 사이에서 달라지는 것은 Python 통합 레이어뿐입니다.

## 완성할 코드

이 퍼즐을 완성하려면 커스텀 연산을 호출하는 한 줄만 채우면 됩니다:

```python
{{#include ../../../../../problems/p20/p20.py:conv1d_pytorch}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p20/p20.py" class="filename">전체 파일 보기: problems/p20/p20.py</a>

다음 명령으로 퍼즐을 실행할 수 있습니다:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p20
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p20
```

  </div>
  <div class="tab-content">

```bash
uv run poe p20
```

  </div>
</div>

성공하면 다음과 비슷한 출력을 볼 수 있습니다:

```
Puzzle 20: From MAX Graph to PyTorch Custom Ops
============================================================
Input array: [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]
Convolution kernel: [0. 1. 2. 3.]

NumPy reference result: [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]

Testing PyTorch Custom Op (device: cuda)
----------------------------------------
PyTorch custom op result: [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]
✅ PyTorch custom op verification PASSED

Comparing with MAX Graph approach (like p15)
--------------------------------------------
MAX Graph result: [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]
✅ MAX Graph verification PASSED
✅ PyTorch and MAX Graph results MATCH
```

## 풀이

<details class="solution-details">
<summary></summary>

컴파일된 커스텀 연산을 적절한 인자와 함께 호출하면 됩니다:

```python
{{#include ../../../../../solutions/p20/p20.py:conv1d_pytorch_call}}
```

<div class="solution-explanation">

이 풀이는 몇 가지 핵심 개념을 보여줍니다:

### 1. **torch.compile() 통합**

`torch.compile` 통합 방식은 다음과 같습니다:

```python
torch.compile(conv1d)(output_tensor, input_tensor, kernel_tensor)
```

### 2. **명시적 출력 텐서 할당**

```python
output_tensor = torch.empty_like(input_tensor)
```

- MAX Graph는 출력 할당을 자동으로 처리하지만
- PyTorch CustomOpLibrary는 **미리 할당된 출력 텐서**가 필요합니다
- Mojo 연산 시그니처는 `(out, input, kernel)` 순서를 기대합니다

### 3. **파라미터 딕셔너리**

```python
ops.conv1d[{"input_size": input_tensor.shape[0], "conv_size": kernel_tensor.shape[0]}]
```

- 파라미터는 딕셔너리 형태로 연산에 전달됩니다
- 이 값들은 Mojo 커널의 컴파일 타임 파라미터가 됩니다
- Mojo `@staticmethod fn execute` 시그니처의 파라미터 이름과 일치해야 합니다

### 4. **같은 커널, 다른 통합 방식**

내부의 Mojo 커널(`conv1d_kernel`)은 Puzzle 17과 동일합니다:

- 동일한 GPU 커널 코드
- 동일한 메모리 접근 패턴
- 동일한 연산 로직
- Python 래퍼 레이어만 달라짐

</div>

</details>

## 핵심 개념

이 퍼즐은 PyTorch 커스텀 연산의 주요 패턴을 보여줍니다:

| 개념 | MAX Graph (p15) | PyTorch CustomOpLibrary (p18) |
|---------|-----------------|-------------------------------|
| **출력 할당** | 자동 | 수동 (`torch.empty_like()`) |
| **연산 호출** | `ops.custom(...)` | `torch.compile(op)(...)` |
| **파라미터 전달** | `parameters={...}` | `op[{...}]` |
| **디바이스 관리** | 명시적 device context | PyTorch 텐서의 device |
| **메모리 관리** | MAX Graph 텐서 | PyTorch 텐서 |

### 핵심 패턴: 명시적 출력 텐서 할당

가장 중요한 차이점은 PyTorch CustomOpLibrary가 **명시적 출력 텐서 할당**을 요구한다는 것입니다:

```python
# ❌ 동작하지 않음 - 출력 텐서 없음
result = torch.compile(conv1d)(input_tensor, kernel_tensor)

# ✅ 동작함 - 미리 할당된 출력 텐서
output_tensor = torch.empty_like(input_tensor)
torch.compile(conv1d)(output_tensor, input_tensor, kernel_tensor)
```

이 패턴이 보장하는 것들:

- 올바른 디바이스에 메모리 할당
- 출력 텐서의 shape과 dtype이 정확
- Mojo 커널이 출력 버퍼에 직접 쓰기 가능

### torch.compile() 통합

`torch.compile()`이 필수적인 이유:

- PyTorch와 Mojo 사이의 메모리 레이아웃 변환 처리
- 디바이스 동기화 관리 (CPU ↔ GPU)
- 텐서 포맷 변환 최적화
- 메모리 연산에 대한 적절한 오류 처리 제공

_참고: `torch.compile()` 없이 사용하면 `std::bad_alloc` 오류가 발생할 수 있습니다. 이는 raw 연산이 PyTorch의 텐서 메모리 관리를 처리하지 못하기 때문입니다._

## 커스텀 연산 디버깅

자주 발생하는 문제와 해결 방법:

1. **메모리 할당 오류**: 항상 `torch.compile()`을 사용하세요
2. **잘못된 출력 Shape**: 출력 텐서가 기대하는 차원과 일치하는지 확인하세요
3. **디바이스 불일치**: 모든 텐서가 같은 디바이스에 있어야 합니다
4. **파라미터 오류**: 파라미터 이름이 Mojo 연산 시그니처와 일치하는지 확인하세요

디버깅 접근법: PyTorch 결과를 동일한 커널을 실행하는 MAX Graph 레퍼런스 구현과 비교해 보세요.
