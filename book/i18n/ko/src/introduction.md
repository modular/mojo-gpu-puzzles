<!-- i18n-source-commit: c9af1879392132dd9daba5704b3500ab642a5ae3 -->

# Mojo 🔥 GPU Puzzles, Edition 1

<div class="social-buttons">
  <a href="https://github.com/modular/mojo-gpu-puzzles" target="_blank">
    <span class="title">Puzzles Repo</span>
    <span class="icon icon-github"></span>
  </a>
  <a href="https://docs.modular.com/mojo/manual/" target="_blank">
    <span class="title">Mojo Manual</span>
    <span class="icon icon-book"></span>
  </a>
  <div class="secondary">
    <a href="https://discord.com/channels/1087530497313357884/1098713601386233997"
      data-tooltip="Chat on Discord"
      aria-label="Discord"
      target="_blank">
      <span class="icon icon-discord"></span>
    </a>
    <a href="https://forum.modular.com/c"
      data-tooltip="Join the forum"
      aria-label="Forum"
      target="_blank">
      <span class="icon icon-discourse"></span>
    </a>
    <a href="https://docs.modular.com/max/get-started/#stay-in-touch"
      data-tooltip="Subscribe to newsletter"
      aria-label="Newsletter"
      target="_blank">
      <span class="icon icon-mail"></span>
    </a>
  </div>
</div>

> _"우리가 할 수 있기 전에 배워야 하는 것들은, 하면서 배운다."_
> 아리스토텔레스 (니코마코스 윤리학)

[Mojo](https://docs.modular.com/mojo/manual/) 🔥를 사용한 GPU 프로그래밍 실습 가이드에 오신 것을 환영합니다. Mojo는 Python 문법과 시스템 수준의 성능을 결합한 프로그래밍 언어입니다.

아래 개요 영상을 먼저 시청하거나, 계속 읽어주세요.

{{ youtube -VsP4kT6DjA full-width }}

## 왜 GPU 프로그래밍인가?

GPU 프로그래밍은 전문 기술에서 현대 컴퓨팅의 핵심 인프라로 발전했습니다. 수십억 개의 매개변수를 처리하는 대규모 언어 모델부터 실시간 영상 스트림을 분석하는 컴퓨터 비전 시스템까지, GPU 가속이 오늘날의 연산 혁신을 이끌고 있습니다. 기후 모델링, 신약 발견, 양자 시뮬레이션 등 과학적 발전은 GPU만이 제공하는 대규모 병렬 처리 능력에 의존하고 있습니다. 금융 기관은 실시간 리스크 분석과 알고리즘 트레이딩에 GPU 컴퓨팅을 활용하며, 자율주행 차량은 GPU 가속 신경망을 통해 센서 데이터를 처리하여 중요한 의사결정을 내립니다.

경제적 파급력도 상당합니다. GPU 컴퓨팅을 효과적으로 활용하는 조직은 개발 주기 단축, 연산 비용 절감, 그리고 이전에는 풀기 어려웠던 계산 문제를 해결할 수 있는 능력 등 상당한 경쟁 우위를 확보합니다. 계산 능력이 비즈니스 가치와 직결되는 시대에, GPU 프로그래밍 역량은 엔지니어, 연구자, 조직에게 전략적 차별화 요소입니다.

## 왜 GPU 프로그래밍에 Mojo🔥를 사용하는가?

컴퓨팅 산업은 중대한 전환점에 도달했습니다. CPU 성능은 전력과 발열 제약으로 인해 클럭 속도 향상만으로는 한계에 이르렀습니다. 이에 따라 하드웨어 제조사들은 물리적 코어 수를 늘리는 방향으로 나아갔고, 이러한 멀티코어 접근 방식의 정점이 바로 수천 개의 코어가 병렬로 동작하는 현대 GPU입니다. NVIDIA H100을 예로 들면, 단일 클럭 사이클에 16,896개의 스레드를 동시에 실행하면서 270,000개 이상의 스레드를 대기시킬 수 있습니다.

Mojo는 GPU 프로그래밍에 대한 실용적인 접근 방식을 제공하여, 이러한 병렬성을 더 쉽게 활용할 수 있게 합니다:

- **Python 스타일 문법**으로 시스템 프로그래밍까지
- 추상화해도 성능 손실 없이 머신 코드로 컴파일되는 **제로 코스트 추상화**
- 컴파일 타임에 오류를 잡는 **강력한 타입 시스템**
- 하드웨어 최적화를 고려한 **Tensor 기본 지원**
- CPU·GPU 내장 함수를 직접 호출할 수 있는 **하드웨어 직접 제어**
- CPU와 GPU 모두에서 동작하는 **크로스 하드웨어 이식성**
- C/C++ 대비 **향상된 안전성**
- **낮은 진입 장벽**으로 더 많은 프로그래머가 GPU 성능을 활용

> **Mojo🔥는 GPU 프로그래밍을 누구나 할 수 있도록 만들어 혁신을 이끌고자 합니다.** >**익숙한 Python 문법을 바탕으로 GPU에 직접 접근할 수 있어, 깊은 전문 지식 없이도 CPU와 GPU를 함께 활용하는 고성능 애플리케이션을 만들 수 있습니다.**

## 왜 퍼즐로 배우는가?

대부분의 GPU 프로그래밍 자료는 실습에 앞서 방대한 이론을 먼저 다룹니다. 하지만 직접 해봐야 이해되는 추상적 개념들은 입문자에게 부담이 될 수 있습니다.

이 책은 다른 접근 방식을 택합니다. 실전 문제에 바로 뛰어들어, 단계적으로 개념을 발견해 나갑니다.

**퍼즐 기반 학습의 장점:**

- **직접 체험**: GPU에서 바로 실행하고 결과를 확인할 수 있습니다
- **점진적 복잡도**: 각 퍼즐이 이전에 배운 개념 위에 쌓여갑니다
- **실용적 초점**: 실제 계산 문제를 반영한 퍼즐로 구성되어 있습니다
- **디버깅 능력**: 체계적인 디버깅 연습을 통해 문제 해결 감각을 키웁니다
- **지식 정착**: 직접 풀어보는 것이 읽기만 하는 것보다 이해가 더 깊어집니다

암기가 아닌 발견에 중점을 둡니다. 직접 실험하면서 개념을 자연스럽게 익히고, 깊은 이해와 실전 역량을 함께 쌓아갈 수 있습니다.

> **감사의 말**: 이 책의 Part I과 III은 인터랙티브 NVIDIA GPU 학습 프로젝트인 [GPU Puzzles](https://github.com/srush/GPU-Puzzles)에서 큰 영감을 받았습니다. 이 책은 해당 개념들을 Mojo의 추상화와 성능을 활용하여 재구현하고,
> Mojo에 특화된 최적화로 고급 주제를 더 넓게 다룹니다.

## GPU 프로그래밍 사고방식

효과적인 GPU 프로그래밍을 위해서는 계산을 바라보는 방식 자체를 바꿔야 합니다. 앞으로의 학습에 길잡이가 될 핵심 사고 모델을 소개합니다:

### 순차에서 병렬로: 반복문을 스레드로 대체

기존 CPU 프로그래밍에서는 반복문으로 데이터를 하나씩 순서대로 처리합니다:

```python
# CPU 방식
for i in range(data_size):
    result[i] = process(data[i])
```

GPU 프로그래밍은 이 방식을 완전히 뒤집습니다. 데이터를 하나씩 순회하는 대신, 수천 개의 스레드를 할당하여 데이터 요소를 동시에 처리합니다:

```mojo
# GPU 방식 (개념적)
thread_id = get_global_id()
if thread_id < data_size:
    result[thread_id] = process(data[thread_id])
```

각 스레드가 하나의 데이터 요소를 맡아 처리하므로, 명시적인 반복문이 대규모 병렬 실행으로 바뀝니다. 순차 처리에서 동시 실행으로의 전환이 GPU 프로그래밍의 핵심 개념입니다.

### 데이터 위에 연산 그리드 맞추기

데이터를 구조화된 그리드로, GPU 스레드가 이에 대응하는 연산 그리드를 형성한다고 생각해 보세요. 효과적인 GPU 프로그래밍은 이 스레드 구성을 잘 설계하여 데이터 공간을 최적으로 처리하는 것입니다:

- **Thread**: 각각 특정 데이터 요소를 담당하는 개별 처리 단위
- **Block**: 공유 메모리 접근과 동기화 기능을 갖춘 스레드 그룹
- **Grid**: 전체 계산 문제를 아우르는 스레드 계층 구조

GPU 프로그래밍을 잘하려면 메모리 접근 패턴과 동기화 요구사항을 관리하면서 병렬 효율을 최대한 끌어올리도록 이 스레드 구성의 균형을 잡아야 합니다.

### 데이터 이동 vs. 연산

GPU 프로그래밍에서는 연산 자체보다 데이터를 옮기는 비용이 더 클 때가 많습니다:

- CPU와 GPU 간 데이터 이동은 느립니다
- 글로벌 메모리에서 공유 메모리로의 이동은 그보다 빠릅니다
- 레지스터나 공유 메모리에 이미 있는 데이터를 다루는 것은 매우 빠릅니다

이는 프로그래밍에서 흔히 가지는 가정을 뒤집습니다. 병목은 연산이 아니라 데이터 이동입니다.

이 책의 퍼즐들을 풀어가면서 이러한 원칙을 직관적으로 체득하고, 계산 문제에 접근하는 방식을 바꿔 나갈 수 있습니다.

## 학습 내용

이 책은 기초 원리부터 고급 GPU 프로그래밍 기법까지 다룹니다. GPU를 알 수 없는 블랙박스로 두지 않고, 개별 스레드의 동작부터 시작하여 정교한 병렬 알고리즘까지 단계별로 이해를 쌓아갑니다. 저수준 메모리 관리와 고수준 텐서 추상화를 모두 배움으로써, 어떤 GPU 프로그래밍 과제에도 유연하게 대응할 수 있게 됩니다.

### 현재 학습 과정

| 핵심 기술                          | 상태           | 퍼즐          |
| ---------------------------------- | -------------- | ------------- |
| 스레드/블록 기초                    | ✅ **제공 중** | Part I (1-8)     |
| GPU 프로그램 디버깅                 | ✅ **제공 중** | Part II (9-10)   |
| 핵심 알고리즘                       | ✅ **제공 중** | Part III (11-16) |
| MAX Graph 통합                     | ✅ **제공 중** | Part IV (17-19)  |
| PyTorch 통합                       | ✅ **제공 중** | Part V (20-22)   |
| 함수형 패턴 및 벤치마킹             | ✅ **제공 중** | Part VI (23)     |
| Warp 프로그래밍                     | ✅ **제공 중** | Part VII (24-26) |
| 블록 수준 프로그래밍                | ✅ **제공 중** | Part VIII (27)   |
| 고급 메모리 연산                    | ✅ **제공 중** | Part IX (28-29)  |
| 성능 분석                           | ✅ **제공 중** | Part X (30-32)   |
| 최신 GPU 기능                       | ✅ **제공 중** | Part XI (33-34)  |

### 상세 학습 목표

**Part I: GPU 기초 (퍼즐 1-8) ✅**

- 스레드 인덱싱과 블록 구성 배우기
- 메모리 접근 패턴과 가드 이해하기
- raw 포인터와 LayoutTensor 추상화 모두 다뤄보기
- 스레드 간 통신을 위한 공유 메모리 기초 익히기

**Part II: GPU 프로그램 디버깅 (퍼즐 9-10) ✅**

- GPU 디버거와 디버깅 기법 배우기
- sanitizer로 메모리 오류와 경쟁 상태(race condition) 찾기
- GPU 버그를 체계적으로 식별하고 수정하기
- 복잡한 GPU 프로그래밍 과제에 도전할 자신감 쌓기

> **참고**: 디버깅 퍼즐을 실행하려면 NVIDIA GPU 디버깅 도구 접근을 위한 `pixi`가 필요합니다. CUDA를 지원하는 NVIDIA GPU에서만 작동합니다.

**Part III: GPU 알고리즘 (퍼즐 11-16) ✅**

- 병렬 reduction과 pooling 연산 구현하기
- 효율적인 convolution 커널 만들기
- prefix sum(scan) 알고리즘 배우기
- tiling 전략으로 행렬 곱셈 최적화하기

**Part IV: MAX Graph 통합 (퍼즐 17-19) ✅**

- custom MAX Graph 연산 만들기
- GPU 커널과 Python 코드 연결하기
- softmax, attention 같은 프로덕션 수준의 연산 구현하기

**Part V: PyTorch 통합 (퍼즐 20-22) ✅**

- Mojo GPU 커널과 PyTorch 텐서 연결하기
- CustomOpLibrary로 텐서 marshalling을 매끄럽게 처리하기
- torch.compile과 통합하여 실행 최적화하기
- kernel fusion과 custom backward pass 배우기

**Part VI: Mojo 함수형 패턴 및 벤치마킹 (퍼즐 23) ✅**

- 함수형 패턴 배우기: elementwise, tiled 처리, vectorization
- 체계적인 성능 최적화와 트레이드오프 익히기
- 정량적 벤치마킹으로 성능 분석하기
- GPU 스레딩 vs SIMD 실행 계층 구조 이해하기

**Part VII: Warp 수준 프로그래밍 (퍼즐 24-26) ✅**

- Warp 기초와 SIMT 실행 모델 배우기
- 핵심 Warp 연산 익히기: sum, shuffle_down, broadcast
- shuffle_xor와 prefix_sum으로 고급 패턴 구현하기
- Warp 프로그래밍과 함수형 패턴을 효과적으로 결합하기

**Part VIII: 블록 수준 프로그래밍 (퍼즐 27) ✅**

- `block.sum()`과 `block.max()`로 블록 단위 reduction 배우기
- 블록 수준 prefix sum 패턴과 통신 익히기
- `block.broadcast()`로 블록 내 조율 효율적으로 구현하기

**Part IX: 고급 메모리 시스템 (퍼즐 28-29) ✅**

- 최적의 메모리 coalescing 패턴 구현하기
- 비동기 메모리 연산으로 연산과 전송을 겹쳐 latency 숨기기
- memory fence와 동기화 primitive 배우기
- prefetching과 캐시 최적화 전략 익히기

**Part X: 성능 분석 및 최적화 (퍼즐 30-32) ✅**

- GPU 커널 profiling으로 병목 지점 찾기
- occupancy와 리소스 활용도 최적화하기
- 공유 메모리 bank conflict 제거하기

**Part XI: 고급 GPU 기능 (퍼즐 33-34) ✅**

- AI 워크로드를 위한 tensor core 프로그래밍 배우기
- 현대 GPU의 cluster 프로그래밍 배우기

이 책은 기존 방식과 달리, 먼저 저수준 메모리 조작으로 이해를 쌓은 뒤 점진적으로 Mojo의 LayoutTensor 추상화로 전환합니다. 이를 통해 GPU 메모리 패턴에 대한 깊은 이해와 현대적 텐서 기반 접근법의 실용적 지식을 모두 얻을 수 있습니다.

## 시작할 준비가 되셨나요?

GPU 프로그래밍이 왜 중요한지, Mojo가 왜 적합한지, 그리고 퍼즐로 어떻게 배우는지 살펴보았습니다. 이제 시작해 봅시다.

**다음 단계**: [이 책의 사용법](howto.md)에서 환경 설정, 시스템 요구사항, 첫 번째 퍼즐 실행 방법을 확인하세요.
