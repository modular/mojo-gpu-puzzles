<!-- i18n-source-commit: 477e5a0d3eed091b3dde0812977773f7dc97730a -->

# Puzzle 34: GPU 클러스터 프로그래밍 (SM90+)

## 소개

> **하드웨어 요구사항: ⚠️ NVIDIA SM90+ 전용**
>
> 이 퍼즐은 SM90+ 컴퓨트 능력을 갖춘 **NVIDIA Hopper 아키텍처** (H100, H200) 이상의 GPU가 필요합니다. 클러스터 프로그래밍 API는 하드웨어 가속 기반이며, 지원하지 않는 하드웨어에서는 오류가 발생합니다. 사용 중인 아키텍처가 확실하지 않다면 `pixi run gpu-specs`를 실행하여 최소 `Compute Cap: 9.0` 이상인지 확인하세요 (하드웨어 식별에 대한 자세한 내용은 [NVIDIA 프로파일링 기초](../puzzle_30/nvidia_profiling_basics.md)를 참고하세요)

**[Warp 레벨 프로그래밍 (Puzzle 24-26)](../puzzle_24/puzzle_24.md)** 에서 **[블록 레벨 프로그래밍 (Puzzle 27)](../puzzle_27/puzzle_27.md)** 까지의 여정을 이어, 이제 **클러스터 레벨 프로그래밍**을 배웁니다 - 단일 블록의 한계를 넘어서는 문제를 해결하기 위해 여러 스레드 블록을 조정하는 기법입니다.

## 스레드 블록 클러스터란?

스레드 블록 클러스터는 하드웨어 가속 동기화 및 통신 기본 요소를 통해 **여러 스레드 블록이 협력**하여 하나의 연산 작업을 수행할 수 있게 해주는 혁신적인 SM90+ 기능입니다.

**핵심 기능:**

- **블록 간 동기화**: [`cluster_sync`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_sync), [`cluster_arrive`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive), [`cluster_wait`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_wait)로 여러 블록을 조정합니다
- **블록 식별**: [`block_rank_in_cluster`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/block_rank_in_cluster)를 사용하여 고유한 블록 조정을 수행합니다
- **효율적인 조정**: [`elect_one_sync`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/elect_one_sync)로 최적화된 Warp 수준 협력을 구현합니다
- **고급 패턴**: [`cluster_mask_base`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_mask_base)로 선택적 블록 조정을 수행합니다

## 클러스터 프로그래밍 모델

### 기존 GPU 프로그래밍 계층 구조

```
Grid (Multiple Blocks)
├── Block (Multiple Warps) - barrier() synchronization
    ├── Warp (32 Threads) - SIMT lockstep execution
    │   ├── Lane 0  ─┐
    │   ├── Lane 1   │ All execute same instruction
    │   ├── Lane 2   │ at same time (SIMT)
    │   │   ...      │ warp.sum(), warp.broadcast()
    │   └── Lane 31 ─┘
        └── Thread (SIMD operations within each thread)
```

### **새로운 계층: 클러스터 프로그래밍 계층 구조:**

```
Grid (Multiple Clusters)
├── 🆕 Cluster (Multiple Blocks) - cluster_sync(), cluster_arrive()
    ├── Block (Multiple Warps) - barrier() synchronization
        ├── Warp (32 Threads) - SIMT lockstep execution
        │   ├── Lane 0  ─┐
        │   ├── Lane 1   │ All execute same instruction
        │   ├── Lane 2   │ at same time (SIMT)
        │   │   ...      │ warp.sum(), warp.broadcast()
        │   └── Lane 31 ─┘
            └── Thread (SIMD operations within each thread)
```

**실행 모델 상세:**

- **스레드 레벨**: 개별 스레드 내에서의 [SIMD 연산](../puzzle_23/gpu-thread-vs-simd.md)
- **Warp 레벨**: [SIMT 실행](../puzzle_24/warp_simt.md) - 32개 스레드의 lockstep 조정
- **블록 레벨**: 공유 메모리와 barrier를 활용한 [멀티 Warp 조정](../puzzle_27/puzzle_27.md)
- **🆕 클러스터 레벨**: SM90+ 클러스터 API를 활용한 멀티 블록 조정

## 학습 단계

이 퍼즐은 클러스터 프로그래밍 역량을 체계적으로 쌓아가는 **3단계 구성**으로 설계되었습니다:

### **[🔰 멀티 블록 조정 기초](./cluster_coordination_basics.md)**

**핵심**: 클러스터 동기화 패턴의 기본 이해

여러 스레드 블록이 [`cluster_arrive()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive)와 [`cluster_wait()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_wait)를 사용하여 기본적인 블록 간 통신과 데이터 분배를 위해 실행을 조정하는 방법을 배웁니다.

**주요 API**: [`block_rank_in_cluster()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/block_rank_in_cluster), [`cluster_arrive()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive), [`cluster_wait()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_wait)

---

### **[☸️ 클러스터 전체 집합 연산](./cluster_collective_ops.md)**

**핵심**: 블록 레벨 패턴을 클러스터 규모로 확장

익숙한 `block.sum()` 개념을 여러 스레드 블록에 걸쳐 확장하여 대규모 연산을 조정하는 클러스터 전체 reduction과 집합 연산을 배웁니다.

**주요 API**: [`cluster_sync()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_sync), 효율적인 클러스터 조정을 위한 [`elect_one_sync()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/elect_one_sync)

---

### **[🧠 고급 클러스터 알고리즘](./advanced_cluster_patterns.md)**

**핵심**: 프로덕션 수준의 다단계 조정 패턴

GPU 활용률을 극대화하고 복잡한 연산 워크플로우를 구현하기 위해 Warp 레벨, 블록 레벨, 클러스터 레벨의 조정을 결합하는 정교한 알고리즘을 구현합니다.

**주요 API**: [`elect_one_sync()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/elect_one_sync), [`cluster_arrive()`](https://docs.modular.com/mojo/std/gpu/primitives/cluster/cluster_arrive), 고급 조정 패턴

## 클러스터 프로그래밍이 중요한 이유

**문제 규모**: 현대 AI 및 과학 워크로드는 단일 스레드 블록의 능력을 초과하는 연산을 필요로 하는 경우가 많습니다:

- 블록 간 조정이 필요한 **대규모 행렬 연산** ([Puzzle 16의 행렬 곱셈](../puzzle_16/puzzle_16.md)과 같은)
- [Puzzle 29의 생산자-소비자 의존성](../puzzle_29/barrier.md)을 갖는 **다단계 알고리즘**
- [Puzzle 8의 공유 메모리](../puzzle_08/puzzle_08.md)보다 큰 데이터셋에 대한 **전역 통계**
- 이웃 블록 간 통신이 필요한 **고급 stencil 연산**

**하드웨어 발전**: GPU가 더 많은 연산 유닛을 갖추게 됨에 따라 ([Puzzle 30의 GPU 아키텍처 프로파일링](../puzzle_30/nvidia_profiling_basics.md) 참고), **클러스터 프로그래밍은 차세대 하드웨어를 효율적으로 활용하는 데 필수적**이 됩니다.

## 교육적 가치

이 퍼즐을 완료하면 완전한 **GPU 프로그래밍 계층 구조**를 학습하게 됩니다:

- **스레드 레벨**: [SIMD 연산을 수행하는 개별 연산 단위](../puzzle_23/gpu-thread-vs-simd.md)
- **[Warp 레벨](../puzzle_24/puzzle_24.md)**: [32개 스레드 SIMT 조정](../puzzle_24/warp_simt.md) (Puzzle 24-26)
- **[블록 레벨](../puzzle_27/puzzle_27.md)**: [공유 메모리를 활용한 멀티 Warp 조정](../puzzle_27/block_sum.md) (Puzzle 27)
- **🆕 클러스터 레벨**: 멀티 블록 조정 (Puzzle 34)
- **그리드 레벨**: [다수의 SM(Streaming Multiprocessor)](../puzzle_30/profile_kernels.md)에 걸친 독립적 블록 실행

이 과정은 [Puzzle 30-32의 성능 최적화 기법](../puzzle_30/puzzle_30.md)을 기반으로, **차세대 GPU 프로그래밍**과 **대규모 병렬 컴퓨팅** 도전에 대비할 수 있도록 준비시켜 줍니다.

## 시작하기

**선수 조건**:

- [블록 레벨 프로그래밍 (Puzzle 27)](../puzzle_27/puzzle_27.md)에 대한 완전한 이해
- [Warp 레벨 프로그래밍 (Puzzle 24-26)](../puzzle_24/puzzle_24.md) 경험
- [공유 메모리 개념 (Puzzle 8)](../puzzle_08/puzzle_08.md)을 통한 GPU 메모리 계층 구조 숙지
- [Barrier를 활용한 GPU 동기화 (Puzzle 29)](../puzzle_29/puzzle_29.md)에 대한 이해
- NVIDIA SM90+ 하드웨어 또는 호환 환경 접근

**권장 학습 방법**: 3단계 구성을 순서대로 따라가세요. 각 단계가 다음 단계의 복잡성을 위한 핵심 개념을 구축합니다.

**하드웨어 참고**: SM90+ 이외의 하드웨어에서 실행하는 경우, 이 퍼즐은 클러스터 프로그래밍 개념과 API 사용 패턴의 **교육적 예제**로 활용할 수 있습니다.

GPU 프로그래밍의 미래를 배울 준비가 되셨나요? **[멀티 블록 조정 기초](./cluster_coordination_basics.md)** 부터 시작하여 기본적인 클러스터 동기화 패턴을 배워보세요!
