<!-- i18n-source-commit: 45b4d82c1f65818e5f47f7c2925ef0f3083e9875 -->

## 퍼즐 사용 가이드

각 퍼즐은 단계적으로 실력을 쌓을 수 있도록 다음과 같은 일관된 구조로 구성되어 있습니다:

- **개요**: 문제 정의와 핵심 개념 소개
- **구성**: 기술적 설정과 메모리 구성 설명
- **완성할 코드**: `problems/pXX/`에 채워야 할 부분이 표시된 구현 템플릿
- **힌트**: 필요할 때 참고할 수 있는 전략적 힌트로, 정답을 직접 알려주지 않습니다
- **풀이**: 성능 고려사항과 개념 설명을 포함한 종합 분석

퍼즐은 이전에 배운 개념 위에 새로운 개념을 쌓아가며 점차 복잡해집니다. 고급 퍼즐은 앞선 퍼즐의 개념을 알고 있다고 가정하므로, 순서대로 풀어나가는 것을 권장합니다.

## 코드 실행하기

모든 퍼즐에는 구현 결과를 예상 결과와 비교해주는 테스트 프레임워크가 포함되어 있습니다. 각 퍼즐별로 실행 방법과 검증 절차가 안내됩니다.

## 사전 준비

### 시스템 요구사항

먼저 시스템이 [시스템 요구사항](https://docs.modular.com/max/packages#system-requirements)을 충족하는지 확인하세요.

### 지원되는 GPU

퍼즐을 실행하려면 [지원되는 GPU](https://docs.modular.com/max/faq#gpu-requirements)가 필요합니다. 환경 설정을 마친 뒤 아래 환경 설정의 `gpu-specs` 명령어로 GPU 호환성을 확인할 수 있습니다.

## 운영체제

> [!NOTE]
> 운영체제별 GPU 지원 설정 방법을 안내합니다.
>
> - [NVIDIA를 사용하는 Windows WSL2 for Linux](#windows-wsl2-for-linux-with-nvidia)
> - [NVIDIA를 사용하는 Linux 네이티브](#linux-native-with-nvidia)
> - [macOS Apple Silicon](#macos-apple-silicon)

### Windows WSL2 for Linux with NVIDIA

Windows Subsystem for Linux(WSL2, 예: Ubuntu)에서 NVIDIA GPU를 설정하려면 [NVIDIA CUDA on WSL 가이드](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)를 참고하세요.

핵심은 *Windows*용 NVIDIA CUDA 드라이버를 설치하는 것입니다. 이 드라이버가 WSL2를 완벽히 지원합니다.
Windows에 NVIDIA GPU 드라이버를 설치하면 WSL 2 안에서 CUDA를 바로 사용할 수 있습니다.
Windows 호스트의 CUDA 드라이버가 WSL 2 내부에서 libcuda.so로 스텁(stub) 처리되므로, WSL 2 안에 별도의 NVIDIA GPU Linux 드라이버를 설치해서는 안 됩니다.

드라이버 설치 후 정상 동작을 확인합니다.

Windows에서 확인: PowerShell을 엽니다 (WSL이 아닙니다)

```bash
nvidia-smi
```

WSL 내부에서 확인: (먼저 WSL을 시작합니다. 예: wsl -d Ubuntu)

```bash
ls -l /usr/lib/wsl/lib/nvidia-smi
/usr/lib/wsl/lib/nvidia-smi
```

Pixi에서 설정을 확인하고, 필요시 누락된 요구사항을 설치합니다 (예: cuda-gdb 디버깅용)

```bash
pixi run nvidia-smi
pixi run setup-cuda-gdb
pixi run mojo debug --help
pixi run cuda-gdb --version
```

WSL에서는 VS Code를 에디터로 사용할 수 있습니다.

- Windows에서 [https://code.visualstudio.com/](https://code.visualstudio.com/)을 통해 VS Code를 설치합니다.
- 그런 다음 Remote - WSL 확장을 설치합니다.

> [!NOTE]
> 퍼즐 1-15는 모두 WSL과 Linux에서 작동합니다.

### Linux native with NVIDIA

먼저 GPU와 Ubuntu 버전을 확인합니다 (지원되는 Ubuntu LTS: 20.04, 22.04, 24.04)

```bash
lspci | grep -i nvidia
lsb_release -a
```

NVIDIA 드라이버를 설치합니다 (필수)

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

Linux에서는 VS Code를 에디터로 사용할 수 있습니다. VS Code APT 저장소를 통해 설치하는 방법은 다음과 같습니다.

Microsoft GPG 키 가져오기

```bash
wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/packages.microsoft.gpg > /dev/null
```

VS Code APT 저장소 추가

```bash
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] \
https://packages.microsoft.com/repos/code stable main" \
| sudo tee /etc/apt/sources.list.d/vscode.list
```

VS Code 설치 및 확인

```bash
sudo apt update
sudo apt install code
code --version
```

> [!NOTE]
> 퍼즐 1-15는 모두 Linux에서 작동합니다.

### macOS Apple Silicon

`osx-arm64` 사용자는 다음이 필요합니다:

- **macOS 15.0 이상** — 최적 호환성을 위해 권장됩니다. `pixi run check-macos`로 확인하고, 실패하면 업그레이드하세요.
- **Xcode 16 이상** — 최소 요구사항입니다. `xcodebuild -version`으로 확인합니다.

`xcrun -sdk macosx metal` 실행 시 `cannot execute tool 'metal' due to missing Metal toolchain` 오류가 나타나면 다음을 실행합니다.

```bash
xcodebuild -downloadComponent MetalToolchain
```

이후 `xcrun -sdk macosx metal`을 다시 실행하면 `no input files error`가 나타나야 정상입니다.

> [!NOTE]
> 현재 퍼즐 1-8과 11-15가 macOS에서 작동합니다. 더 많은 퍼즐 지원을 준비하고 있습니다!

## 프로그래밍 지식

다음에 대한 기본적인 이해가 있으면 좋습니다:

- 프로그래밍 기초 (변수, 반복문, 조건문, 함수)
- 병렬 컴퓨팅 개념 (스레드, 동기화, 경쟁 상태(race condition))
- [Mojo](https://docs.modular.com/mojo/manual/) 기본 문법 ([포인터 입문](https://docs.modular.com/mojo/manual/pointers/) 섹션 포함)
- [GPU 프로그래밍 기초](https://docs.modular.com/mojo/manual/gpu/fundamentals)를 미리 읽어두면 도움이 됩니다!

GPU 프로그래밍 경험이 없어도 괜찮습니다! 퍼즐을 풀어가며 자연스럽게 익힐 수 있습니다.

Mojo🔥와 함께 GPU 컴퓨팅의 세계로 떠나봅시다!

## 환경 설정하기

1. [GitHub 저장소](https://github.com/modular/mojo-gpu-puzzles)를 클론하고 해당 디렉토리로 이동합니다:

    ```bash
    # 저장소 클론
    git clone https://github.com/modular/mojo-gpu-puzzles
    cd mojo-gpu-puzzles
    ```

2. Mojo🔥 프로그램을 실행하기 위한 패키지 매니저를 설치합니다:

   #### **옵션 1 (강력 추천)**: [pixi](https://pixi.sh/latest/#installation)

    이 프로젝트에서 `pixi`를 **권장하는 이유**는 다음과 같습니다:
    - Modular의 MAX/Mojo 패키지에 쉽게 접근 가능
    - GPU 의존성을 자동으로 처리
    - conda + PyPI 생태계를 모두 지원

    > **참고: 일부 퍼즐은 `pixi`에서만 작동합니다**

    **설치:**

    ```bash
    curl -fsSL https://pixi.sh/install.sh | sh
    ```

    **업데이트:**

    ```bash
    pixi self-update
    ```

   #### **옵션 2**: [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

    **설치:**

    ```bash
    curl -fsSL https://astral.sh/uv/install.sh | sh
    ```

    **업데이트:**

    ```bash
    uv self update
    ```

    **가상 환경 생성:**

    ```bash
    uv venv && source .venv/bin/activate
    ```

3. **설정을 확인하고 첫 번째 퍼즐을 실행합니다:**

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi NVIDIA (default)</button>
    <button class="tab-button">pixi AMD</button>
    <button class="tab-button">pixi Apple</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
# GPU 사양 확인
pixi run gpu-specs

# 첫 번째 퍼즐 실행
# 아직 구현 전이므로 실패합니다! 본문을 따라 구현해 보세요
pixi run p01
```

  </div>
  <div class="tab-content">

```bash
# GPU 사양 확인
pixi run gpu-specs

# 첫 번째 퍼즐 실행
# 아직 구현 전이므로 실패합니다! 본문을 따라 구현해 보세요
pixi run -e amd p01
```

  </div>
  <div class="tab-content">

```bash
# GPU 사양 확인
pixi run gpu-specs

# 첫 번째 퍼즐 실행
# 아직 구현 전이므로 실패합니다! 본문을 따라 구현해 보세요
pixi run -e apple p01
```

  </div>
  <div class="tab-content">

```bash
# GPU별 의존성 설치
uv pip install -e ".[nvidia]"  # NVIDIA GPU용
# 또는
uv pip install -e ".[amd]"     # AMD GPU용

# GPU 사양 확인
uv run poe gpu-specs

# 첫 번째 퍼즐 실행
# 아직 구현 전이므로 실패합니다! 본문을 따라 구현해 보세요
uv run poe p01
```

  </div>
</div>

## 퍼즐 풀기

### 프로젝트 구조

- **[`problems/`](https://github.com/modular/mojo-gpu-puzzles/tree/main/problems)**: 풀이를 직접 구현하는 곳입니다 (여기서 작업합니다!)
- **[`solutions/`](https://github.com/modular/mojo-gpu-puzzles/tree/main/solutions)**: 비교와 학습을 위한 참고 풀이입니다. 책 전반에 걸쳐 활용됩니다

### 작업 흐름

1. `problems/pXX/`에서 퍼즐 템플릿을 엽니다
2. 제공된 프레임워크 안에 풀이를 작성합니다
3. 구현을 테스트합니다: `pixi run pXX` 또는 `uv run poe pXX` (플랫폼에 따라 `-e platform`을 추가합니다. 예: `-e amd`)
4. `solutions/pXX/`의 참고 풀이와 비교하며 다른 접근 방식을 배웁니다

### 주요 명령어

<div class="code-tabs" data-tab-group="package-manager">
  <div class="tab-buttons">
    <button class="tab-button">pixi</button>
    <button class="tab-button">uv</button>
  </div>
  <div class="tab-content">

```bash
# 퍼즐 실행 (필요시 -e로 플랫폼 지정)
pixi run pXX             # NVIDIA (기본값) `pixi run -e nvidia pXX`와 동일
pixi run -e amd pXX      # AMD GPU
pixi run -e apple pXX    # Apple GPU

# 풀이 테스트
pixi run tests           # 모든 풀이 테스트
pixi run tests pXX       # 특정 퍼즐 테스트

# 수동 실행
pixi run mojo problems/pXX/pXX.mojo     # 내 구현
pixi run mojo solutions/pXX/pXX.mojo    # 참고 풀이

# 인터랙티브 셸
pixi shell               # 환경 진입
mojo problems/p01/p01.mojo              # 직접 실행
exit                     # 셸 종료

# 개발
pixi run format         # 코드 포맷팅
pixi task list          # 사용 가능한 명령어
```

  </div>
  <div class="tab-content">

```bash
# 참고: uv는 제한적이며 일부 챕터는 pixi가 필요합니다
# GPU별 의존성 설치:
uv pip install -e ".[nvidia]"  # NVIDIA GPU용
uv pip install -e ".[amd]"     # AMD GPU용

# 풀이 테스트
uv run poe tests        # 모든 풀이 테스트
uv run poe tests pXX    # 특정 퍼즐 테스트

# 수동 실행
uv run mojo problems/pXX/pXX.mojo      # 내 구현
uv run mojo solutions/pXX/pXX.mojo     # 참고 풀이
```

  </div>
</div>

## GPU 지원 현황

아래 표는 퍼즐별 GPU 플랫폼 호환성을 정리한 것입니다. 퍼즐에 따라 필요한 GPU 기능과 벤더별 도구가 다릅니다.

| 퍼즐 | NVIDIA GPU | AMD GPU | Apple GPU | 비고 |
|------|------------|---------|-----------|------|
| **Part I: GPU 기초** | | | | |
| 1 - Map | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 2 - Zip | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 3 - Guard | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 4 - Map 2D | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 5 - Broadcast | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 6 - Blocks | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 7 - Shared Memory | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 8 - Stencil | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| **Part II: 디버깅** | | | | |
| 9 - GPU Debugger | ✅ | ❌ | ❌ | NVIDIA 전용 디버깅 도구 |
| 10 - Sanitizer | ✅ | ❌ | ❌ | NVIDIA 전용 디버깅 도구 |
| **Part III: GPU 알고리즘** | | | | |
| 11 - Reduction | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 12 - Scan | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 13 - Pool | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 14 - Conv | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 15 - Matmul | ✅ | ✅ | ✅ | 기본 GPU 커널 |
| 16 - Flashdot | ✅ | ✅ | ✅ | 고급 메모리 패턴 |
| **Part IV: MAX Graph** | | | | |
| 17 - Custom Op | ✅ | ✅ | ✅ | MAX Graph 통합 |
| 18 - Softmax | ✅ | ✅ | ✅ | MAX Graph 통합 |
| 19 - Attention | ✅ | ✅ | ✅ | MAX Graph 통합 |
| **Part V: PyTorch 통합** | | | | |
| 20 - Torch Bridge | ✅ | ✅ | ❌ | PyTorch 통합 |
| 21 - Autograd | ✅ | ✅ | ❌ | PyTorch 통합 |
| 22 - Fusion | ✅ | ✅ | ❌ | PyTorch 통합 |
| **Part VI: 함수형 패턴** | | | | |
| 23 - Functional | ✅ | ✅ | ✅ | 고급 Mojo 패턴 |
| **Part VII: Warp 프로그래밍** | | | | |
| 24 - Warp Sum | ✅ | ✅ | ✅ | Warp 수준 연산 |
| 25 - Warp Communication | ✅ | ✅ | ✅ | Warp 수준 연산 |
| 26 - Advanced Warp | ✅ | ✅ | ✅ | Warp 수준 연산 |
| **Part VIII: 블록 프로그래밍** | | | | |
| 27 - Block Operations | ✅ | ✅ | ✅ | 블록 단위 프로그래밍 패턴 |
| **Part IX: 메모리 시스템** | | | | |
| 28 - Async Memory | ✅ | ✅ | ✅ | 고급 메모리 연산 |
| 29 - Barriers | ✅ | ❌ | ❌ | NVIDIA 전용 고급 동기화 |
| **Part X: 성능 분석** | | | | |
| 30 - Profiling | ✅ | ❌ | ❌ | NVIDIA 프로파일링 도구 (NSight) |
| 31 - Occupancy | ✅ | ❌ | ❌ | NVIDIA 프로파일링 도구 |
| 32 - Bank Conflicts | ✅ | ❌ | ❌ | NVIDIA 프로파일링 도구 |
| **Part XI: 최신 GPU 기능** | | | | |
| 33 - Tensor Cores | ✅ | ❌ | ❌ | NVIDIA tensor core 전용 |
| 34 - Cluster | ✅ | ❌ | ❌ | NVIDIA cluster 프로그래밍 |

### 범례

- ✅ **지원**: 해당 플랫폼에서 퍼즐이 작동합니다
- ❌ **미지원**: 플랫폼별 고유 기능이 필요합니다

### 플랫폼별 참고사항

**NVIDIA GPU (전체 지원)**

- 모든 퍼즐(1-34)이 CUDA를 지원하는 NVIDIA GPU에서 작동합니다
- CUDA 툴킷과 호환 드라이버가 필요합니다
- 모든 기능을 사용할 수 있어 가장 완전한 학습 경험을 제공합니다

**AMD GPU (폭넓은 지원)**

- 대부분의 퍼즐(1-8, 11-29)이 ROCm을 통해 작동합니다
- 미지원: 디버깅 도구(9-10), 프로파일링(30-32), tensor core(33-34)
- 고급 알고리즘과 메모리 패턴까지 포함하여 GPU 프로그래밍을 폭넓게 학습할 수 있습니다

**Apple GPU (기본 지원)**

- 기초(1-8, 11-18) 및 고급(23-27) 퍼즐 일부를 지원합니다
- 미지원: 고급 기능 전반, 디버깅, 프로파일링 도구
- GPU 프로그래밍의 기본 패턴을 익히기에 적합합니다

> **향후 지원 계획**: AMD 및 Apple GPU에 대한 도구와 플랫폼 지원을 꾸준히 확대하고 있습니다. 디버깅 도구, 프로파일링 기능, 고급 GPU 연산 등 아직 지원되지 않는 기능은 향후 릴리스에 포함될 예정입니다. 크로스 플랫폼 호환성을 계속 개선하고 있으니 업데이트를 확인해 주세요.

## GPU 리소스

### 무료 클라우드 GPU 플랫폼

로컬 GPU가 없다면, 무료로 GPU를 사용할 수 있는 클라우드 플랫폼을 활용할 수 있습니다:

#### **Google Colab**

Google Colab은 무료 GPU 접근을 제공하지만, Mojo GPU 프로그래밍에는 일부 제한이 있습니다:

**사용 가능한 GPU:**

- Tesla T4 (구세대 Turing 아키텍처)
- Tesla V100 (제한적 가용)

**Mojo GPU Puzzles 사용 시 제한사항:**

- **구세대 GPU 아키텍처**: T4 GPU는 고급 Mojo GPU 기능과 호환되지 않을 수 있습니다
- **세션 시간 제한**: 최대 12시간 실행 후 자동으로 연결이 끊깁니다
- **제한적 디버깅 지원**: NVIDIA 디버깅 도구(퍼즐 9-10)를 완전히 사용하지 못할 수 있습니다
- **패키지 설치 제한**: Mojo/MAX 설치 시 우회 방법이 필요할 수 있습니다
- **성능 제한**: 공유 인프라 특성상 일관된 벤치마킹이 어렵습니다

**추천 용도:** 기본 GPU 프로그래밍 개념(퍼즐 1-8, 11-15)과 기초 패턴 학습.

#### **Kaggle Notebooks**

Kaggle은 Colab보다 넉넉한 무료 GPU 사용 시간을 제공합니다:

**사용 가능한 GPU:**

- Tesla T4 (주당 30시간 무료)
- P100 (제한적 가용)

**Colab 대비 장점:**

- **넉넉한 시간**: Colab의 일일 세션 제한과 달리 주당 30시간 사용 가능
- **자동 저장**: 노트북이 자동으로 저장됩니다
- **안정적인 환경**: 패키지 설치가 더 안정적입니다

**Mojo GPU Puzzles 사용 시 제한사항:**

- **GPU 아키텍처 제약**: T4의 고급 기능 호환성 문제는 Colab과 동일
- **제한적 디버깅 도구**: NVIDIA 프로파일링 및 디버깅 도구(퍼즐 9-10, 30-32) 사용 불가
- **Mojo 설치 복잡성**: Mojo 환경을 수동으로 설정해야 합니다
- **cluster 프로그래밍 미지원**: 고급 퍼즐(33-34) 작동 불가

**추천 용도:** 기본 GPU 프로그래밍(퍼즐 1-16)을 장시간에 걸쳐 학습할 때 적합합니다.

### 권장 사항

- **전체 학습 과정**: NVIDIA GPU가 있으면 모든 퍼즐을 학습할 수 있습니다 (전체 34개)
- **폭넓은 학습**: AMD GPU로도 대부분의 내용을 다룰 수 있습니다 (34개 중 27개)
- **기초 학습**: Apple GPU로 기본 개념을 익힐 수 있습니다 (34개 중 13개)
- **무료 플랫폼 학습**: Google Colab/Kaggle로 기초~중급 개념까지 학습 가능합니다 (퍼즐 1-16)
- **디버깅 및 프로파일링**: 디버깅 도구와 성능 분석에는 NVIDIA GPU가 필요합니다
- **최신 GPU 기능**: tensor core와 cluster 프로그래밍에는 NVIDIA GPU가 필요합니다

## 개발

자세한 내용은 [README](https://github.com/modular/mojo-gpu-puzzles#development)를 참고하세요.

## 커뮤니티 참여하기

<p align="center" style="display: flex; justify-content: center; gap: 10px;">
  <a href="https://www.modular.com/company/talk-to-us">
    <img src="https://img.shields.io/badge/Subscribe-Updates-00B5AD?logo=mail.ru" alt="업데이트 구독">
  </a>
  <a href="https://forum.modular.com/c/">
    <img src="https://img.shields.io/badge/Modular-Forum-9B59B6?logo=discourse" alt="Modular 포럼">
  </a>
  <a href="https://discord.com/channels/1087530497313357884/1098713601386233997">
    <img src="https://img.shields.io/badge/Discord-Join_Chat-5865F2?logo=discord" alt="Discord">
  </a>
</p>

커뮤니티에서 GPU 프로그래밍에 대해 이야기하고, 풀이를 공유하고, 서로 도움을 주고받을 수 있습니다.
