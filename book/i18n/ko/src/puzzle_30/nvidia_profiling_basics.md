<!-- i18n-source-commit: 08367c0bf8d5dce82c5cd30526d2924f809cecb1 -->

# 📚 NVIDIA 프로파일링 기초

## 개요

지금까지 GPU 프로그래밍의 기초와 고급 패턴을 배웠습니다. Part II에서는 `compute-sanitizer`와 `cuda-gdb`를 사용한 **정확성** 디버깅 기법을, 다른 파트에서는 Warp 프로그래밍, 메모리 시스템, 블록 레벨 연산 등 다양한 GPU 기능을 다뤘습니다. 커널이 올바르게 동작하긴 합니다 - 하지만 **빠르기도** 할까요?

> 이 튜토리얼은 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#profiling)에서 권장하는 NVIDIA 프로파일링 방법론을 따릅니다.

**핵심 통찰**: 올바른 커널이라도 최적의 성능보다 몇 배나 느릴 수 있습니다. 프로파일링은 동작하는 코드와 고성능 코드 사이의 격차를 좁힙니다.

## 프로파일링 도구 모음

pixi를 통해 `cuda-toolkit`이 설치되어 있으므로, NVIDIA의 전문 프로파일링 도구를 바로 사용할 수 있습니다:

### NSight Systems (`nsys`) - "전체 그림" 도구

**용도**: 시스템 전체 성능 분석 ([NSight Systems 문서](https://docs.nvidia.com/nsight-systems/))

- CPU-GPU 상호작용의 타임라인 뷰
- 메모리 전송 병목
- 커널 실행 오버헤드
- 멀티 GPU 조율
- API 호출 추적

**사용 가능한 인터페이스**: 커맨드라인 (`nsys`) 및 GUI (`nsys-ui`)

**사용 시점**:

- 전체 애플리케이션 흐름 파악
- CPU-GPU 동기화 문제 식별
- 메모리 전송 패턴 분석
- 커널 실행 병목 발견

```bash
# 도움말 보기
pixi run nsys --help

# 기본 시스템 전체 프로파일링
pixi run nsys profile --trace=cuda,nvtx --output=timeline mojo your_program.mojo

# 대화형 분석
pixi run nsys stats --force-export=true timeline.nsys-rep
```

### NSight Compute (`ncu`) - "커널 심층 분석" 도구

**용도**: 상세한 단일 커널 성능 분석 ([NSight Compute 문서](https://docs.nvidia.com/nsight-compute/))

- Roofline 모델 분석
- 메모리 계층 구조 활용도
- Warp 실행 효율
- 레지스터/공유 메모리 사용량
- 연산 유닛 활용도

**사용 가능한 인터페이스**: 커맨드라인 (`ncu`) 및 GUI (`ncu-ui`)

**사용 시점**:

- 특정 커널 성능 최적화
- 메모리 접근 패턴 파악
- 연산 바운드 vs 메모리 바운드 커널 분석
- Warp 분기 문제 식별

```bash
# 도움말 보기
pixi run ncu --help

# 상세 커널 프로파일링
pixi run ncu --set full --output kernel_profile mojo your_program.mojo

# 특정 커널에 집중
pixi run ncu --kernel-name regex:your_kernel_name mojo your_program.mojo
```

## 도구 선택 의사결정 트리

```
성능 문제 발생
      |
      v
어떤 커널인지 아는가?
    |           |
  아니오         예
    |           |
    v           v
NSight    커널 고유의 문제인가?
Systems       |         |
    |       아니오       예
    v         |         |
타임라인        |         v
분석    <------+   NSight Compute
                        |
                        v
                   커널 심층 분석
```

**빠른 의사결정 가이드**:

- 병목이 어디인지 모르겠으면 **NSight Systems (`nsys`)부터 시작**
- 최적화할 커널을 정확히 알면 **NSight Compute (`ncu`) 사용**
- 종합적인 분석이 필요하면 **둘 다 사용** (일반적인 워크플로우)

## 실습: NSight Systems로 시스템 전체 프로파일링

[Puzzle 16](../puzzle_16/puzzle_16.md)의 행렬 곱셈 구현들을 프로파일링하여 성능 차이를 파악해 봅시다.

> **GUI 참고**: NSight Systems와 Compute GUI (`nsys-ui`, `ncu-ui`)는 디스플레이와 OpenGL 지원이 필요합니다. X11 포워딩이 없는 헤드리스 서버나 원격 시스템에서는 커맨드라인 버전 (`nsys`, `ncu`)을 사용하여 `nsys stats`와 `ncu --import --page details`로 텍스트 기반 분석을 수행하세요. `.nsys-rep`와 `.ncu-rep` 파일을 로컬 머신으로 전송하여 GUI로 분석할 수도 있습니다.

### Step 1: 프로파일링을 위한 코드 준비

**중요**: 정확한 프로파일링을 위해 최적화를 유지하면서 전체 디버그 정보를 포함하여 빌드합니다:

```bash
pixi shell -e nvidia
# 최적화를 유지하면서 전체 디버그 정보 포함 빌드 (포괄적인 소스 매핑용)
mojo build --debug-level=full solutions/p16/p16.mojo -o solutions/p16/p16_optimized

# 최적화 빌드 테스트
./solutions/p16/p16_optimized --naive
```

**이것이 중요한 이유**:

- **전체 디버그 정보**: 프로파일러를 위한 완전한 심볼 테이블, 변수명, 소스 라인 매핑 제공
- **포괄적 분석**: NSight 도구가 성능 데이터를 특정 코드 위치와 연결 가능
- **최적화 유지**: 프로덕션 빌드와 일치하는 현실적인 성능 측정 보장

### Step 2: 시스템 전체 프로파일 수집

```bash
# 포괄적 추적으로 최적화 빌드 프로파일링
nsys profile \
  --trace=cuda,nvtx \
  --output=matmul_naive \
  --force-overwrite=true \
  ./solutions/p16/p16_optimized --naive
```

**명령어 분석**:

- `--trace=cuda,nvtx`: CUDA API 호출 및 커스텀 어노테이션 캡처
- `--output=matmul_naive`: 프로파일을 `matmul_naive.nsys-rep`로 저장
- `--force-overwrite=true`: 기존 프로파일 덮어쓰기
- 마지막 인수: Mojo 프로그램

### Step 3: 타임라인 분석

```bash
# 텍스트 기반 통계 생성
nsys stats --force-export=true matmul_naive.nsys-rep

# 주요 지표 확인:
# - GPU 활용률
# - 메모리 전송 시간
# - 커널 실행 시간
# - CPU-GPU 동기화 간격
```

**확인할 수 있는 결과** (2×2 행렬 곱셈의 실제 출력):

```txt
** CUDA API Summary (cuda_api_sum):
 Time (%)  Total Time (ns)  Num Calls  Avg (ns)   Med (ns)  Min (ns)  Max (ns)  StdDev (ns)          Name
 --------  ---------------  ---------  ---------  --------  --------  --------  -----------  --------------------
     81.9          8617962          3  2872654.0    2460.0      1040   8614462    4972551.6  cuMemAllocAsync
     15.1          1587808          4   396952.0    5965.5      3810   1572067     783412.3  cuMemAllocHost_v2
      0.6            67152          1    67152.0   67152.0     67152     67152          0.0  cuModuleLoadDataEx
      0.4            44961          1    44961.0   44961.0     44961     44961          0.0  cuLaunchKernelEx

** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------------------------------
    100.0             1920          1    1920.0    1920.0      1920      1920          0.0  p16_naive_matmul_Layout_Int6A6AcB6A6AsA6A6A

** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):
 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)           Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ----------------------------
     49.4             4224      3    1408.0    1440.0      1312      1472         84.7  [CUDA memcpy Device-to-Host]
     36.0             3072      4     768.0     528.0       416      1600        561.0  [CUDA memset]
     14.6             1248      3     416.0     416.0       416       416          0.0  [CUDA memcpy Host-to-Device]
```

**주요 성능 통찰**:

- **메모리 할당이 지배적**: 전체 시간의 81.9%가 `cuMemAllocAsync`에 소비
- **커널은 번개처럼 빠름**: 실행 시간 1,920 ns (0.000001920초)에 불과
- **메모리 전송 내역**: 49.4% Device→Host, 36.0% memset, 14.6% Host→Device
- **아주 작은 데이터**: 모든 메모리 연산이 0.001 MB 미만 (float32 4개 = 16바이트)

### Step 4: 구현 비교

다른 버전들을 프로파일링하고 비교합니다:

```bash
# pixi shell 상태를 유지하세요 `pixi run -e nvidia`

# 공유 메모리 버전 프로파일링
nsys profile --trace=cuda,nvtx --force-overwrite=true --output=matmul_shared ./solutions/p16/p16_optimized --single-block

# Tiled 버전 프로파일링
nsys profile --trace=cuda,nvtx --force-overwrite=true --output=matmul_tiled ./solutions/p16/p16_optimized --tiled

# 관용적 Tiled 버전 프로파일링
nsys profile --trace=cuda,nvtx --force-overwrite=true --output=matmul_idiomatic_tiled ./solutions/p16/p16_optimized --idiomatic-tiled

# 각 구현을 개별적으로 분석 (nsys stats는 한 번에 하나의 파일만 처리)
nsys stats --force-export=true matmul_shared.nsys-rep
nsys stats --force-export=true matmul_tiled.nsys-rep
nsys stats --force-export=true matmul_idiomatic_tiled.nsys-rep
```

**결과 비교 방법**:

1. **GPU Kernel Summary 확인** - 구현 간 실행 시간 비교
2. **Memory Operations 확인** - 공유 메모리가 글로벌 메모리 트래픽을 줄이는지 확인
3. **API 오버헤드 비교** - 모두 비슷한 메모리 할당 패턴을 가져야 함

**수동 비교 워크플로우**:

```bash
# 각 분석 결과를 저장하여 비교
nsys stats --force-export=true matmul_naive.nsys-rep > naive_stats.txt
nsys stats --force-export=true matmul_shared.nsys-rep > shared_stats.txt
nsys stats --force-export=true matmul_tiled.nsys-rep > tiled_stats.txt
nsys stats --force-export=true matmul_idiomatic_tiled.nsys-rep > idiomatic_tiled_stats.txt
```

**공정한 비교 결과** (실제 프로파일링 출력):

### 비교 1: 2 x 2 행렬

| 구현 | 메모리 할당 | 커널 실행 | 성능 |
|------|-----------|----------|------|
| **Naive** | 81.9% cuMemAllocAsync | ✅ 1,920 ns | 기준선 |
| **Shared** (`--single-block`) | 81.8% cuMemAllocAsync | ✅ 1,984 ns | **+3.3% 느림** |

### 비교 2: 9 x 9 행렬

| 구현 | 메모리 할당 | 커널 실행 | 성능 |
|------|-----------|----------|------|
| **Tiled** (수동) | 81.1% cuMemAllocAsync | ✅ 2,048 ns | 기준선 |
| **Idiomatic Tiled** | 81.6% cuMemAllocAsync | ✅ 2,368 ns | **+15.6% 느림** |

**공정 비교에서 얻은 핵심 통찰**:

**두 행렬 크기 모두 GPU 작업에는 너무 작음!**:

- **2×2 행렬**: 4개 요소 - 완전히 오버헤드가 지배
- **9×9 행렬**: 81개 요소 - 여전히 오버헤드가 지배
- **실제 GPU 워크로드**: 차원당 수천~수백만 개 요소

**이 결과가 실제로 보여주는 것**:

- **모든 변형이 메모리 할당에 지배됨** (시간의 81% 이상)
- **커널 실행은 의미 없음** - 설정 비용에 비하면 미미
- **"최적화"가 오히려 해로울 수 있음**: 공유 메모리가 3.3%, async_copy가 15.6% 오버헤드 추가
- **진짜 교훈**: 작은 워크로드에서는 알고리즘 선택이 무의미 - 오버헤드가 모든 것을 압도

**이런 결과가 나오는 이유**:

- GPU 설정 비용(메모리 할당, 커널 실행)은 문제 크기에 관계없이 고정
- 작은 문제에서는 이 고정 비용이 연산 시간을 무색하게 만듦
- 큰 문제를 위해 설계된 최적화가 작은 문제에서는 오버헤드가 됨

**실무 프로파일링 교훈**:

- **문제 크기 맥락이 중요**: 2×2와 9×9 모두 GPU에게는 작음
- **고정 비용이 작은 문제를 지배**: 메모리 할당, 커널 실행 오버헤드
- **"최적화"가 작은 워크로드에 해로울 수 있음**: 공유 메모리, 비동기 연산이 오버헤드 추가
- **작은 문제를 최적화하지 말 것**: 실제 워크로드로 확장 가능한 알고리즘에 집중
- **항상 벤치마킹할 것**: "더 좋은" 코드에 대한 가정은 흔히 틀림

**작은 커널 프로파일링의 이해**:
이 2×2 행렬 예제는 **전형적인 작은 커널 패턴**을 보여줍니다:

- 실제 연산(행렬 곱셈)은 극히 빠름 (1,920 ns)
- 메모리 설정 오버헤드가 전체 시간을 지배 (실행의 97% 이상)
- 이것이 **실무 GPU 최적화**가 다음에 집중하는 이유입니다:
  - **연산 일괄 처리**로 설정 비용 분산
  - **메모리 재사용**으로 할당 오버헤드 감소
  - 연산이 병목이 되는 **더 큰 문제 크기**

## 실습: NSight Compute로 커널 심층 분석

이제 특정 커널의 성능 특성을 심층적으로 들여다봅시다.

### Step 1: 특정 커널 프로파일링

```bash
# 활성 shell 상태인지 확인
pixi shell -e nvidia

# Naive MatMul 커널을 상세 프로파일링 (최적화 빌드 사용)
ncu \
  --set full \
  -o kernel_analysis \
  --force-overwrite \
  ./solutions/p16/p16_optimized --naive
```

> **흔한 문제: 권한 오류**
>
> `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters` 오류가 발생하면 다음 해결 방법을 시도하세요:
>
> ```bash
> # NVIDIA 드라이버 옵션 추가 (rmmod보다 안전)
> echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee -a /etc/modprobe.d/nvidia-kernel-common.conf
>
> # 커널 파라미터 설정
> sudo sysctl -w kernel.perf_event_paranoid=0
>
> # 영구 적용
> echo 'kernel.perf_event_paranoid=0' | sudo tee -a /etc/sysctl.conf
>
> # 드라이버 변경 사항 적용을 위해 재부팅 필요
> sudo reboot
>
> # 그런 다음 ncu 명령을 다시 실행
> ncu \
>   --set full \
>   -o kernel_analysis \
>   --force-overwrite \
>   ./solutions/p16/p16_optimized --naive
> ```

### Step 2: 주요 지표 분석

```bash
# 상세 보고서 생성 (올바른 구문)
ncu --import kernel_analysis.ncu-rep --page details
```

**실제 NSight Compute 출력** (2×2 Naive MatMul):

```txt
GPU Speed Of Light Throughput
----------------------- ----------- ------------
DRAM Frequency              Ghz         6.10
SM Frequency                Ghz         1.30
Elapsed Cycles            cycle         3733
Memory Throughput             %         1.02
DRAM Throughput               %         0.19
Duration                     us         2.88
Compute (SM) Throughput       %         0.00
----------------------- ----------- ------------

Launch Statistics
-------------------------------- --------------- ---------------
Block Size                                                     9
Grid Size                                                      1
Threads                           thread               9
Waves Per SM                                                0.00
-------------------------------- --------------- ---------------

Occupancy
------------------------------- ----------- ------------
Theoretical Occupancy                 %        33.33
Achieved Occupancy                    %         2.09
------------------------------- ----------- ------------
```

**실제 데이터에서 얻은 핵심 통찰**:

#### 성능 분석 - 냉혹한 현실

- **Compute Throughput: 0.00%** - GPU가 연산적으로 완전히 유휴 상태
- **Memory Throughput: 1.02%** - 메모리 대역폭을 거의 사용하지 않음
- **Achieved Occupancy: 2.09%** - GPU 능력의 2%만 사용 중
- **Grid Size: 1 블록** - 80개 멀티프로세서를 완전히 낭비!

#### 성능이 이렇게 낮은 이유

- **작은 문제 크기**: 2×2 행렬 = 총 4개 요소
- **잘못된 실행 구성**: 1개 블록에 9개 스레드 (32의 배수여야 함)
- **심각한 과소 활용**: SM당 0.00 wave (효율을 위해 수천 개 필요)

#### NSight Compute의 핵심 최적화 권고사항

- **"Est. Speedup: 98.75%"** - 80개 SM을 모두 사용하도록 그리드 크기 증가
- **"Est. Speedup: 71.88%"** - 스레드 블록을 32의 배수로 사용
- **"Kernel grid is too small"** - GPU 효율을 위해 훨씬 큰 문제 필요

### Step 3: 현실 직시

**이 프로파일링 데이터가 알려주는 것**:

1. **작은 문제는 GPU에게 독**: 2×2 행렬은 GPU 리소스를 완전히 낭비
2. **실행 구성이 중요**: 잘못된 스레드/블록 크기가 성능을 죽임
3. **규모가 알고리즘보다 중요**: 근본적으로 작은 문제는 어떤 최적화로도 해결 불가
4. **NSight Compute는 정직함**: 커널 성능이 낮을 때 그대로 알려줌

**진짜 교훈**:

- **토이 문제를 최적화하지 말 것** - 실제 GPU 워크로드를 대표하지 않음
- **현실적인 워크로드에 집중** - 최적화가 실제로 의미 있는 1000×1000+ 행렬
- **프로파일링으로 최적화를 안내** - 단, 최적화할 가치가 있는 문제에만

**2×2 예제의 경우**: 정교한 알고리즘(공유 메모리, tiling)이 이미 오버헤드가 지배적인 워크로드에 오버헤드만 추가합니다.

## 프로파일러 출력을 성능 탐정처럼 읽기

### 자주 나타나는 성능 패턴

#### 패턴 1: 메모리 바운드 커널

**NSight Systems가 보여주는 것**: 긴 메모리 전송 시간
**NSight Compute가 보여주는 것**: 높은 메모리 처리량, 낮은 연산 활용도
**해결책**: 메모리 접근 패턴 최적화, 공유 메모리 사용

#### 패턴 2: 낮은 점유율

**NSight Systems가 보여주는 것**: 짧은 커널 실행과 간격
**NSight Compute가 보여주는 것**: 실제 점유율이 낮음
**해결책**: 레지스터 사용량 줄이기, 블록 크기 최적화

#### 패턴 3: Warp 분기

**NSight Systems가 보여주는 것**: 불규칙한 커널 실행 패턴
**NSight Compute가 보여주는 것**: 낮은 Warp 실행 효율
**해결책**: 조건 분기 최소화, 알고리즘 재구성

### 프로파일링 탐정 워크플로우

```txt
성능 문제 발생
     |
     v
NSight Systems: 전체 그림
        |
        v
GPU를 잘 활용하고 있는가?
    |             |
  아니오           예
    |             |
    v             v
CPU-GPU    NSight Compute: 커널 상세
파이프라인          |
수정               v
        메모리 또는 연산 바운드인가?
          |       |       |
         메모리   연산    둘 다 아님
          |       |       |
          v       v       v
        메모리    산술     점유율
        접근     최적화    확인
        최적화
```

## 프로파일링 모범 사례

포괄적인 프로파일링 지침은 [Best Practices Guide - Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics)를 참고하세요.

### 이렇게 하세요

1. **대표적인 워크로드를 프로파일링**: 현실적인 데이터 크기와 패턴 사용
2. **전체 디버그 정보로 빌드**: 최적화와 함께 포괄적인 프로파일링 데이터 및 소스 매핑을 위해 `--debug-level=full` 사용
3. **GPU 워밍업**: 커널을 여러 번 실행한 후 후반 반복을 프로파일링
4. **대안 비교**: 항상 여러 구현을 프로파일링
5. **핫스팟에 집중**: 가장 시간이 오래 걸리는 커널을 최적화

### 이렇게 하지 마세요

1. **디버그 정보 없이 프로파일링하지 말 것**: 성능을 소스 코드에 매핑할 수 없음 (`mojo build --help`)
2. **단일 실행만 프로파일링하지 말 것**: GPU 성능은 실행마다 달라질 수 있음
3. **메모리 전송을 무시하지 말 것**: CPU-GPU 전송이 흔히 지배적
4. **섣불리 최적화하지 말 것**: 먼저 프로파일링, 그다음 최적화

### 흔한 함정과 해결책

#### 함정 1: 콜드 스타트 효과

```bash
# 잘못된 방법: 첫 번째 실행을 프로파일링
nsys profile mojo your_program.mojo

# 올바른 방법: 워밍업 후 프로파일링
nsys profile --delay=5 mojo your_program.mojo  # GPU 워밍업 대기
```

#### 함정 2: 잘못된 빌드 구성

```bash
# 잘못된 방법: 전체 디버그 빌드 (최적화 비활성화) 즉, `--no-optimization`
mojo build -O0 your_program.mojo -o your_program

# 잘못된 방법: 디버그 정보 없음 (소스 매핑 불가)
mojo build your_program.mojo -o your_program

# 올바른 방법: 프로파일링을 위한 전체 디버그 정보 포함 최적화 빌드
mojo build --debug-level=full your_program.mojo -o optimized_program
nsys profile ./optimized_program
```

#### 함정 3: 메모리 전송 무시

```txt
# NSight Systems에서 이 패턴을 찾아보세요:
CPU -> GPU transfer: 50ms
Kernel execution: 2ms
GPU -> CPU transfer: 48ms
# 총: 100ms (커널은 겨우 2%!)
```

**해결책**: 전송과 연산을 중첩하고 전송 빈도를 줄이기 (Part IX에서 다룸)

#### 함정 4: 단일 커널에만 집중

```bash
# 잘못된 방법: "느린" 커널만 프로파일링
ncu --kernel-name regex:slow_kernel program

# 올바른 방법: 먼저 전체 애플리케이션을 프로파일링
nsys profile mojo program.mojo  # 실제 병목 찾기
```

## 모범 사례와 고급 옵션

### 고급 NSight Systems 프로파일링

포괄적인 시스템 전체 분석을 위해 다음 고급 `nsys` 플래그를 사용합니다:

```bash
# 프로덕션급 프로파일링 명령
nsys profile \
  --gpu-metrics-devices=all \
  --trace=cuda,osrt,nvtx \
  --trace-fork-before-exec=true \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  --opengl-gpu-workload=false \
  --delay=2 \
  --duration=30 \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --output=comprehensive_profile \
  --force-overwrite=true \
  ./your_program
```

**플래그 설명**:

- `--gpu-metrics-devices=all`: 모든 디바이스에서 GPU 지표 수집
- `--trace=cuda,osrt,nvtx`: 포괄적 API 추적
- `--cuda-memory-usage=true`: 메모리 할당/해제 추적
- `--cuda-um-cpu/gpu-page-faults=true`: Unified Memory 페이지 폴트 모니터링
- `--delay=2`: 프로파일링 전 2초 대기 (콜드 스타트 회피)
- `--duration=30`: 최대 30초간 프로파일링
- `--sample=cpu`: 핫스팟 분석을 위한 CPU 샘플링 포함
- `--cpuctxsw=process-tree`: CPU 컨텍스트 스위치 추적

### 고급 NSight Compute 프로파일링

포괄적 지표를 포함한 상세 커널 분석:

```bash
# 모든 지표 세트로 전체 커널 분석
ncu \
  --set full \
  --import-source=on \
  --kernel-id=:::1 \
  --launch-skip=0 \
  --launch-count=1 \
  --target-processes=all \
  --replay-mode=kernel \
  --cache-control=all \
  --clock-control=base \
  --apply-rules=yes \
  --check-exit-code=yes \
  --export=detailed_analysis \
  --force-overwrite \
  ./your_program

# 특정 성능 측면에 집중
ncu \
  --set=@roofline \
  --section=InstructionStats \
  --section=LaunchStats \
  --section=Occupancy \
  --section=SpeedOfLight \
  --section=WarpStateStats \
  --metrics=sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:your_kernel_.* \
  --export=targeted_analysis \
  ./your_program
```

**주요 NSight Compute 플래그**:

- `--set full`: 사용 가능한 모든 지표 수집 (포괄적이지만 느림)
- `--set @roofline`: roofline 분석에 최적화된 세트
- `--import-source=on`: 결과를 소스 코드에 매핑
- `--replay-mode=kernel`: 정확한 측정을 위해 커널 리플레이
- `--cache-control=all`: 일관된 결과를 위한 GPU 캐시 제어
- `--clock-control=base`: 기본 주파수로 클럭 고정
- `--section=SpeedOfLight`: Speed of Light 분석 포함
- `--metrics=...`: 특정 지표만 수집
- `--kernel-name regex:pattern`: 정규식 패턴으로 커널 지정 (`--kernel-regex`가 아님)

### 프로파일링 워크플로우 모범 사례

#### 1. 점진적 프로파일링 전략

```bash
# Step 1: 빠른 개요 (빠름)
nsys profile --trace=cuda --duration=10 --output=quick_look ./program

# Step 2: 상세 시스템 분석 (중간)
nsys profile --trace=cuda,osrt,nvtx --cuda-memory-usage=true --output=detailed ./program

# Step 3: 커널 심층 분석 (느리지만 포괄적)
ncu --set=@roofline --kernel-name regex:hotspot_kernel ./program
```

#### 2. 신뢰성을 위한 다중 실행 분석

```bash
# 여러 번 프로파일링하고 비교
for i in {1..5}; do
  nsys profile --output=run_${i} ./program
  nsys stats run_${i}.nsys-rep > stats_${i}.txt
done

# 결과 비교
diff stats_1.txt stats_2.txt
```

#### 3. 타겟 커널 프로파일링

```bash
# 먼저 핫스팟 커널 식별
nsys profile --trace=cuda,nvtx --output=overview ./program
nsys stats overview.nsys-rep | grep -A 10 "GPU Kernel Summary"

# 그런 다음 특정 커널 프로파일링
ncu --kernel-name="identified_hotspot_kernel" --set full ./program
```

### 환경 및 빌드 모범 사례

#### 최적 빌드 구성

```bash
# 프로파일링용: 전체 디버그 정보 포함 최적화 빌드
mojo build --debug-level=full --optimization-level=3 program.mojo -o program_profile

# 빌드 설정 확인
mojo build --help | grep -E "(debug|optimization)"
```

#### 프로파일링 환경 설정

```bash
# 일관된 결과를 위해 GPU 부스트 비활성화
sudo nvidia-smi -ac 1215,1410  # 메모리 및 GPU 클럭 고정

# 결정론적 동작 설정
export CUDA_LAUNCH_BLOCKING=1  # 정확한 타이밍을 위한 동기식 실행

# 프로파일링을 위한 드라이버 제한 완화
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee -a /etc/modprobe.d/nvidia-kernel-common.conf
```

#### 메모리 및 성능 격리

```bash
# 프로파일링 전 GPU 메모리 초기화
nvidia-smi --gpu-reset

# 다른 GPU 프로세스 비활성화
sudo fuser -v /dev/nvidia*  # GPU 사용 중인 프로세스 확인
sudo pkill -f cuda  # 필요시 CUDA 프로세스 종료

# 높은 우선순위로 실행
sudo nice -n -20 nsys profile ./program
```

### 분석 및 보고 모범 사례

#### 종합 보고서 생성

```bash
# 여러 보고서 형식 생성
nsys stats --report=cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format=csv --output=. profile.nsys-rep

# 외부 분석을 위해 내보내기
nsys export --type=sqlite profile.nsys-rep
nsys export --type=json profile.nsys-rep

# 비교 보고서 생성
nsys stats --report=cuda_gpu_kern_sum baseline.nsys-rep > baseline_kernels.txt
nsys stats --report=cuda_gpu_kern_sum optimized.nsys-rep > optimized_kernels.txt
diff -u baseline_kernels.txt optimized_kernels.txt
```

#### 성능 회귀 테스트

```bash
#!/bin/bash
# CI/CD용 자동화 프로파일링 스크립트
BASELINE_TIME=$(nsys stats baseline.nsys-rep | grep "Total Time" | awk '{print $3}')
CURRENT_TIME=$(nsys stats current.nsys-rep | grep "Total Time" | awk '{print $3}')

REGRESSION_THRESHOLD=1.10  # 10% 성능 저하 임계값
if (( $(echo "$CURRENT_TIME > $BASELINE_TIME * $REGRESSION_THRESHOLD" | bc -l) )); then
    echo "Performance regression detected: ${CURRENT_TIME}ns vs ${BASELINE_TIME}ns"
    exit 1
fi
```

## 다음 단계

프로파일링 기초를 이해했으니:

1. **기존 커널로 연습**: 이미 풀었던 퍼즐들을 프로파일링해 보세요
2. **최적화 준비**: Puzzle 31에서 이 통찰을 점유율 최적화에 활용합니다
3. **도구 익히기**: 다양한 NSight Systems와 NSight Compute 옵션을 실험해 보세요

**기억하세요**: 프로파일링은 단순히 느린 코드를 찾는 것이 아닙니다 - 프로그램의 동작을 이해하고 근거 있는 최적화 결정을 내리는 것입니다.

추가 프로파일링 자료:

- [NVIDIA Profiler User's Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)
- [NSight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NSight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
