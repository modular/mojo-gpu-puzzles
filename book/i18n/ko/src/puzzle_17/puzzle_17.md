<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# Puzzle 17: 1D Convolution Op

> ## MAX Graph로 Python 연동하기
>
> GPU 퍼즐 여정의 Part IV에 진입했습니다: **MAX Graph 커스텀 Op으로 Python 연동하기**.
>
> 이전 퍼즐들에서는 Mojo로 효율적인 GPU kernel을 작성하는 방법을 배웠습니다. 이제부터는 다음을 알아봅니다:
>
> - Kernel들을 Python에서 호출할 수 있는 커스텀 연산으로 패키징하기
> - MAX Graph 시스템과 통합하여 머신러닝을 가속하기
> - 하이레벨 Python API와 로우레벨 GPU 코드 사이의 간극 메우기
>
> 이를 통해 익숙한 Python 환경에서 작업하면서도 Mojo GPU kernel의 성능을 활용할 수 있습니다.

## 개요

[Puzzle 13: 1D Convolution](../puzzle_13/puzzle_13.md)에서 GPU에서 효율적으로 동작하는 1D convolution kernel을 구현했습니다. 이번에는 이 kernel을 [MAX Graph](https://docs.modular.com/max/api/python/graph/)를 통해 Python에서 직접 호출할 수 있는 커스텀 연산으로 변환합니다.

사용할 1D convolution kernel은 이미 구현되어 있습니다:

```mojo
{{#include ../../../../../problems/p17/op/conv1d.mojo:conv1d_kernel}}
```

이 퍼즐의 핵심 요소는 다음과 같습니다:

1. **커스텀 op 등록**: `@compiler.register` 데코레이터를 통해 Mojo 함수를 Python에 노출하는 방법 이해하기
2. **커스텀 op 패키징**: Mojo 코드를 MAX Graph에서 사용할 수 있도록 패키징하는 방법 익히기
3. **Python 통합**: MAX Graph를 통해 Python에서 커스텀 연산 호출하기
4. **크로스 언어 데이터 흐름**: Python과 GPU 사이의 데이터 타입과 메모리 관리하기

이 커스텀 연산은 다음과 같은 일을 수행합니다:

- Python에서 [NumPy](https://numpy.org/doc/stable/) 배열을 입력으로 받기
- 이 데이터를 GPU로 전송하기
- 최적화된 convolution kernel 실행하기
- 결과를 Python으로 반환하기

이 퍼즐을 완성하면 Python의 풍부한 생태계와 Mojo의 강력한 GPU 성능을 잇는 매끄러운 다리를 만들게 됩니다.

## 완성할 코드

이 퍼즐을 완성하려면 `conv1d.mojo`에서 `conv1d_kernel`을 호출하는 한 줄만 채우면 됩니다:

```mojo
{{#include ../../../../../problems/p17/op/conv1d.mojo:conv1d_custom_op}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p17/op/conv1d.mojo" class="filename">전체 파일 보기: problems/p17/op/conv1d.mojo</a>

다음 명령으로 퍼즐을 실행할 수 있습니다:

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
pixi run p17
```

  </div>
  <div class="tab-content">

```bash
pixi run -e amd p17
```

  </div>
  <div class="tab-content">

```bash
pixi run -e apple p17
```

  </div>
  <div class="tab-content">

```bash
uv run poe p17
```

  </div>
</div>

성공하면 다음과 비슷한 출력을 볼 수 있습니다:

```
Input array: [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]
Convolution kernel: [0. 1. 2. 3.]
Expected result (NumPy calculation): [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]
Compiling 1D convolution graph...
Executing 1D convolution...
1D Convolution result (custom Mojo kernel): [14. 20. 26. 32. 38. 44. 50. 56. 62. 68. 74. 80. 41. 14.  0.]
Verification passed: Custom kernel results match NumPy calculation
```

이 출력은 커스텀 MAX Graph 연산이 1D convolution 알고리즘을 올바르게 구현했음을 나타냅니다.

## 풀이

<details class="solution-details">
<summary></summary>

이 퍼즐을 풀려면 1D convolution kernel을 MAX Graph 시스템과 통합해야 합니다. 핵심은 `Conv1DCustomOp` 구조체의 `execute` 메서드에서 kernel을 올바르게 호출하는 것입니다.

풀이는 다음과 같습니다:

```mojo
{{#include ../../../../../solutions/p17/op/conv1d.mojo:conv1d_custom_op_solution}}
```

<div class="solution-explanation">
이 한 줄이 수행하는 중요한 작업들은 다음과 같습니다:

1. GPU 컨텍스트(`gpu_ctx`의 타입은 [DeviceContext](https://docs.modular.com/mojo/std/gpu/host/device_context/DeviceContext/))에서 [enqueue_function](https://docs.modular.com/mojo/std/gpu/host/device_context/DeviceContext/#enqueue_function)을 호출하여 kernel 실행 예약
2. 필요한 레이아웃과 크기 정보를 **컴파일 타임** 파라미터로 전달
3. 출력, 입력, kernel 텐서를 런타임 인자로 제공
4. 적절한 차원으로 실행 그리드 구성

전체 맥락에서 어떻게 동작하는지 살펴보겠습니다:

### Python-Mojo 통합 흐름

1. **Python 쪽 (<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p17/p17.py" class="filename">problems/p17/p17.py</a>)**:
   - 입력과 kernel용 NumPy 배열 생성
   - MAX Graph로 연산을 감싸는 `conv_1d()` 함수 호출
   - NumPy 배열을 `Buffer.from_numpy(input).to(device)`로 [MAX driver](https://docs.modular.com/max/api/python/driver) Buffer로 변환
   - `custom_extensions=[mojo_kernels]`로 커스텀 연산 패키지 로드

2. **그래프 구축**:
   - [TensorType](https://docs.modular.com/max/api/python/graph/type/#max.graph.type.TensorType)으로 입력 및 출력 텐서 타입 정의
   - `parameters={...}`를 통해 연산의 파라미터 지정
   - [`Graph("conv_1d_graph", ...)`](https://docs.modular.com/max/api/python/graph/Graph)로 연산 그래프 생성
   - [`ops.custom(name="conv1d", ...)`](https://docs.modular.com/max/api/python/graph/ops#custom)로 커스텀 연산 호출

3. **커스텀 op 등록**:
   - `@compiler.register("conv1d")` 데코레이터가 연산을 MAX Graph에 노출. [@compiler.register](https://docs.modular.com/mojo/manual/decorators/compiler-register/) 참고
   - `execute` 메서드의 파라미터가 인터페이스(입력, 출력, 컨텍스트) 정의
   - 입출력 텐서가 kernel에서 사용할 수 있도록 LayoutTensor로 변환
   - Device context가 GPU 메모리 할당과 kernel 실행 관리

4. **Kernel 실행**:
   - `model.execute(...)`가 호출되면 `conv1d_kernel`이 데이터 수신
   - `grid_dim`과 `block_dim`으로 GPU 스레드 구성 설정
   - `result.to(CPU())`로 결과를 CPU로 전송
   - NumPy 검증으로 기대 출력과 결과 비교

### 핵심 구성 요소 상세

1. **커스텀 Op 구조체**:

   ```mojo
   @compiler.register("conv1d")
   struct Conv1DCustomOp:
       @staticmethod
       fn execute[target: StaticString, input_size: Int, conv_size: Int, dtype: DType = DType.float32](
           output: OutputTensor[rank=1],
           input: InputTensor[dtype = output.dtype, rank = output.rank],
           kernel: InputTensor[dtype = output.dtype, rank = output.rank],
           ctx: DeviceContextPtr,
       ) raises:
           # 구현
   ```

   - `target`은 디바이스 타입("gpu" 또는 "cpu")을 나타냄
   - `input_size`와 `conv_size`는 Python에서 전달되는 파라미터
   - 텐서 타입이 올바른 shape과 타입 검사 보장
   - 반환 타입은 적절한 오류 처리 위해 `raises`

2. **텐서 변환**:

   ```mojo
   output_tensor = output.to_layout_tensor()
   input_tensor = input.to_layout_tensor()
   kernel_tensor = kernel.to_layout_tensor()
   ```

   - MAX Graph 텐서를 Mojo LayoutTensor로 변환
   - kernel이 텐서를 직접 다룰 수 있게 해줌
   - 컴파일 타임 최적화를 위해 레이아웃 추출

3. **Device Context 사용**:

   ```mojo
   gpu_ctx = ctx.get_device_context()
   gpu_ctx.enqueue_memset(...)  # 출력 버퍼 초기화
   gpu_ctx.enqueue_function[..., ...](...) # kernel 예약
   ```

   - Device context가 GPU 리소스 관리
   - 메모리 연산이 올바른 버퍼 상태 보장
   - Function enqueueing이 kernel 실행 예약

이 풀이는 Python 데이터가 MAX Graph를 거쳐 GPU에서 실행되고 다시 돌아오는 전체 흐름을 보여줍니다. Mojo의 강력한 타입 시스템과 파라미터 함수를 활용하여 효율적이고 타입 안전한 가속 연산을 만들어냅니다.

</details>

## MAX Graph 커스텀 op 이해하기

> 더 자세한 내용은 아래 튜토리얼을 참고하세요:
>
> - [Get started with MAX Graph in Python](https://docs.modular.com/max/tutorials/get-started-with-max-graph-in-python/)
> - [MAX Graph custom op for GPUs](https://docs.modular.com/max/tutorials/build-custom-ops/)

### 커스텀 op 등록

커스텀 연산을 만드는 핵심은 `@compiler.register` 데코레이터와 관련 구조체입니다:

```mojo
@compiler.register("conv1d")
struct Conv1DCustomOp:
    @staticmethod
    fn execute[...](
        output: OutputTensor[rank=1],
        input: InputTensor[dtype = output.dtype, rank = output.rank],
        kernel: InputTensor[type = output.dtype, rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # 구현
```

등록의 핵심 구성 요소:

- 데코레이터에 전달하는 **이름**(`"conv1d"`)이 Python 코드에서 이 연산을 호출할 때 사용하는 이름
- **구조체**에는 올바른 시그니처를 가진 `execute` 메서드가 있어야 함
- **OutputTensor**와 **InputTensor** 타입이 Python 데이터와의 인터페이스를 정의
- **DeviceContextPtr**이 실행 환경에 대한 접근을 제공

### 커스텀 op 패키징

커스텀 연산을 Python에서 사용하려면 먼저 패키징해야 합니다:

```bash
mojo package op -o op.mojopkg
```

이 명령은:

1. Mojo 코드를 배포 가능한 패키지로 컴파일
2. MAX Graph가 연산을 이해하는 데 필요한 메타데이터 생성
3. Python에서 로드할 수 있는 바이너리 아티팩트(`op.mojopkg`)를 생성

패키지는 MAX Graph가 찾을 수 있는 위치에 배치해야 하며, 보통 Python 코드에서 접근 가능한 디렉토리에 둡니다.

### Python 통합

Python 쪽에서 커스텀 연산을 사용하는 방법은 다음과 같습니다:

```python
# Mojo 연산이 포함된 디렉토리 경로
mojo_kernels = Path(__file__).parent / "op"

# 커스텀 conv1d 연산으로 그래프 구성
with Graph(
    "conv_1d_graph",
    input_types=[...],
    custom_extensions=[mojo_kernels],  # 커스텀 op 패키지 로드
) as graph:
    # 그래프의 입력 정의
    input_value, kernel_value = graph.inputs

    # 이름으로 커스텀 연산 사용
    output = ops.custom(
        name="conv1d",  # @compiler.register의 이름과 일치해야 함
        values=[input_value, kernel_value],
        out_types=[...],
        parameters={
            "input_size": input_tensor.shape[0],
            "conv_size": kernel_tensor.shape[0],
            "dtype": dtype,
        },
    )[0].tensor
```

핵심 요소는 다음과 같습니다:

1. `custom_extensions`로 커스텀 연산의 경로 지정
2. 등록된 연산 이름으로 `ops.custom` 호출
3. 연산의 시그니처에 맞는 입력 값과 파라미터 전달
