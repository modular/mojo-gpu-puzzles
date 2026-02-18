# 용어집 (Glossary)

이 문서는 Mojo GPU Puzzles 한국어 번역에서 사용되는 용어를 정리한 것입니다.
번역 시 이 용어집에 정의된 표기를 따릅니다.

## 표기 원칙

- 국립국어원 외래어 표기법을 기준으로 하되, 업계에서 널리 사용되는 표기를 우선합니다
- 코드, 함수명, 파일 경로, 명령어, 제품명은 번역하지 않습니다
- 기술 용어는 문맥에서 의미가 명확할 때 영어 원문을 유지합니다

---

## 원문 유지 용어

아래 용어들은 번역에서 영어 원문을 그대로 사용합니다:

| English | 비고 |
| --- | --- |
| atomic | 중간 상태 없이 완전히 실행되거나 전혀 실행되지 않는 연산 |
| autograd | PyTorch의 자동 미분 시스템. 역전파 기울기를 자동 계산 |
| barrier | 스레드들이 특정 지점에서 만나는 동기화 장벽 |
| broadcast | 1. 작은 차원의 데이터를 큰 차원으로 확장하여 연산. 예: 벡터를 행렬로 확장<br>2. 데이터를 여러 스레드에 복사하는 연산 (`warp.broadcast()`) |
| butterfly (network) | 나비 모양의 데이터 교환 패턴. 병렬 알고리즘에서 사용 |
| chunk | 데이터를 나눈 덩어리. 각 블록이 처리하는 연속된 데이터 조각 |
| coalescing | 여러 스레드의 메모리 접근을 하나로 묶어서 효율을 높이는 방법 |
| column-major | 열을 연속으로 저장하는 메모리 레이아웃. Fortran 방식 |
| convolution | 합성곱. 이미지나 신경망에서 필터를 씌우는 연산 |
| deadlock | 스레드들이 서로를 기다리며 영원히 멈춘 상태 |
| DRAM | Dynamic Random Access Memory. GPU의 글로벌 메모리. 용량이 크지만 느림 |
| double-buffering | 두 버퍼를 번갈아 쓰는 최적화 기법 |
| embedding | 이산적인 토큰 인덱스를 밀집 벡터 표현으로 변환하는 연산 |
| fusion | kernel을 합쳐서 실행하는 최적화 기법 |
| guard | 경계 검사를 위한 조건문. `if i < size` 형태 |
| halo (region) | 타일 경계를 넘어 확장되는 추가 데이터 영역. ghost cell, guard cell이라고도 함 |
| hazard | 스레드 간 메모리 접근 충돌. read-after-write, write-after-write 등 |
| JIT (Just-In-Time) | 실행 시점에 코드를 컴파일하는 방식. 빌드 단계 없이 빠른 반복 가능 |
| kernel | GPU에서 여러 스레드가 함께 실행하는 함수 |
| lane | Warp 내 각 스레드의 위치 (0-31) |
| latency | 작업이 완료될 때까지 기다리는 시간 |
| lockstep | Warp 내 모든 스레드가 동일 명령을 동시에 실행하는 모드. SIMT의 핵심 동작 방식 |
| LayoutTensor | Mojo의 다차원 배열 추상화 타입 |
| mbarrier | Mojo의 memory barrier API. `mbarrier_init()`, `mbarrier_arrive()`, `mbarrier_test_wait()` 등. 기본 `barrier()`보다 세밀한 동기화 제어 |
| memory fence | 메모리 작업 순서가 뒤바뀌지 않도록 보장하는 장치 |
| memcheck | compute-sanitizer의 메모리 위반 탐지 도구 |
| mixed precision | FP16/BF16 입력 + FP32 누적처럼 정밀도를 혼합하는 기법 |
| off-by-one | 경계값이 1만큼 어긋나는 프로그래밍 오류. 반복문에서 흔히 발생 |
| offset | 메모리 시작 위치로부터의 거리. 인덱스 계산에 사용 |
| pooling | 윈도우 내 값들을 하나로 합치는 연산. max pooling, average pooling 등 |
| prefix sum | 배열에서 각 위치까지의 누적 합을 구하는 알고리즘 |
| PTX (Parallel Thread Execution) | NVIDIA GPU의 가상 어셈블리 언어. 컴파일러가 생성하는 중간 표현 |
| racecheck | compute-sanitizer의 경쟁 상태 탐지 도구 |
| reduction | 여러 값을 합계나 최댓값처럼 하나의 값으로 줄이는 연산 |
| roofline (model) | 하드웨어 한계 대비 성능을 분석하는 모델 |
| row-major | 행을 연속으로 저장하는 메모리 레이아웃. C/Mojo 기본 방식 |
| sanitizer | GPU 코드의 메모리 오류, 경쟁 상태 등을 탐지하는 검사 도구 |
| SAXPY | Single-precision Alpha times X plus Y. `y[i] = α * x[i] + y[i]` 형태의 BLAS Level 1 표준 연산 |
| segmentation fault | 접근 권한이 없는 메모리 영역에 접근할 때 발생하는 오류 |
| shuffle | Warp 내 스레드 간 데이터 교환 |
| softmax | 벡터를 확률 분포로 정규화하는 함수 |
| SIMD | Single Instruction Multiple Data. 벡터 연산 방식 |
| SRAM | Static Random Access Memory. GPU의 공유 메모리에 해당. 용량이 작지만 빠름 |
| stack trace | 오류 발생 시점까지의 함수 호출 경로 |
| SIMT | Single Instruction Multiple Thread. GPU 실행 모델 |
| SM (Streaming Multiprocessor) | GPU의 연산 단위. 여러 Warp를 동시에 실행하는 프로세서 |
| stencil | 이웃 데이터를 참조하는 연산 패턴 |
| stream compaction | 프레디케이트를 만족하는 요소만 연속으로 재배치하는 병렬 알고리즘 |
| stride | 메모리 접근이나 반복의 간격. reduction에서 매 단계마다 절반으로 줄이는 보폭 |
| synccheck | compute-sanitizer의 동기화 버그 탐지 도구 |
| tensor core | GPU의 행렬 연산 전용 하드웨어 |
| tiling | 큰 데이터를 작은 조각으로 나눠서 처리하는 방법 |
| Warp | 32개 스레드가 한 묶음으로 함께 움직이는 GPU의 기본 단위 |

---

## 용어 목록

| English | 한글 | 비고 |
| --- | --- | --- |
| address space | 주소 공간 | 메모리 영역 구분. Mojo에서 `AddressSpace.SHARED` 등으로 지정 |
| bank conflict | 뱅크 충돌 | 여러 스레드가 공유 메모리의 같은 뱅크에 한꺼번에 접근해서 생기는 충돌 |
| conflict-free (pattern) | 충돌 없는 (패턴) | 뱅크 충돌이 발생하지 않는 접근 패턴 |
| arithmetic intensity | 산술 강도 | 데이터 1바이트당 수행하는 연산량 (FLOP/B). Roofline 모델의 X축 |
| backward pass | 역전파 | 신경망에서 기울기를 뒤로 전달하는 과정 |
| binary | 바이너리 | 컴파일된 실행 파일 |
| binning | 구간 분류 | 데이터를 구간으로 나누는 것. 히스토그램 구간 분류 |
| boundary (check) | 경계 (검사) | 배열 인덱스가 유효한 경계 내에 있는지 확인하는 것 |
| block | 블록 | 공유 메모리를 함께 쓰고 서로 동기화할 수 있는 스레드 묶음 |
| buffer overflow | 버퍼 오버플로우 | 버퍼 경계를 넘어서 데이터를 쓰는 메모리 오류 |
| compute-bound | 연산 바운드 | 연산 처리량에 의해 성능이 제한되는 상태 |
| data locality | 데이터 지역성 | 자주 쓰는 데이터를 가까운 메모리에 두는 것 |
| dense vector | 밀집 벡터 | 대부분의 원소가 0이 아닌 벡터. embedding의 출력 형태 |
| dereference | 역참조 | 포인터가 가리키는 메모리의 값에 접근하는 것 |
| dot product | 내적 | 두 벡터의 원소별 곱의 합 |
| element-wise | 요소별 | 배열의 각 요소에 개별적으로 수행하는 연산 |
| global index | 전역 인덱스 | 전체 데이터에서의 위치. `block_dim * block_idx + thread_idx`로 계산 |
| global memory | 글로벌 메모리 | GPU 어디서든 접근할 수 있는 메모리 |
| grid | 그리드 | 전체 계산을 담당하는 블록들의 집합 |
| host code | 호스트 코드 | CPU에서 실행되는 코드. GPU 작업을 설정하는 부분 |
| in-place (computation) | 직접 저장 (연산) | 별도 메모리를 할당하지 않고 기존 버퍼에 결과를 직접 기록하는 방식 |
| kernel code | 커널 코드 | GPU에서 병렬로 실행되는 코드 |
| loop unrolling | 루프 전개 | 반복문을 펼쳐서 반복 오버헤드를 줄이는 컴파일러 최적화 기법 |
| matrix multiplication | 행렬 곱셈 | 두 행렬을 곱하는 연산 |
| memory-bound | 메모리 바운드 | 메모리 대역폭에 의해 성능이 제한되는 상태 |
| memory bandwidth | 메모리 대역폭 | 단위 시간당 전송할 수 있는 데이터 양 |
| memory alignment | 메모리 정렬 | 데이터를 특정 바이트 경계에 맞춰 배치하는 것 |
| memory hierarchy | 메모리 계층 구조 | GPU 메모리의 계층적 구조 (글로벌 → 공유 → 레지스터) |
| memory layout | 메모리 레이아웃 | 데이터가 메모리에 배치되는 방식 |
| memory leak | 메모리 누수 | 할당된 메모리를 해제하지 않아 발생하는 문제 |
| memory violation | 메모리 위반 | 잘못된 메모리 영역에 접근하는 오류 |
| normalization | 정규화 | 값을 일정 범위로 조정하는 것 |
| occupancy | 점유율 | SM당 활성 Warp 수 대비 최대 가능 Warp 수의 비율 |
| overlap | 중첩 | 여러 작업을 동시에 수행하는 것. 복사 중첩 |
| padding | 패딩 | 배열 끝을 0이나 특정 값으로 채워 크기를 맞추는 것 |
| parallel | 병렬 | 여러 작업을 동시에 처리하는 방식 |
| partial block | 부분 블록 | 데이터 끝에서 블록 크기를 다 채우지 못한 블록 |
| predicate | 프레디케이트 | 조건의 참/거짓을 나타내는 값. 병렬 알고리즘에서 파티션 소속을 결정 |
| profiling | 프로파일링 | 프로그램에서 느린 부분을 찾아내는 성능 분석 |
| primitive | 기본 요소 | 프로그래밍의 기본 도구. 동기화 기본 요소 |
| race condition | 경쟁 상태 | 여러 스레드가 같은 데이터에 동시에 접근해서 생기는 오류 |
| register blocking | 레지스터 블로킹 | 레지스터에 값을 누적하여 메모리 트래픽을 줄이는 최적화 기법 |
| shared memory | 공유 메모리 | 같은 블록 안의 스레드들이 함께 쓰는 빠른 메모리 |
| sliding window | 슬라이딩 윈도우 | 데이터 위를 이동하며 처리하는 고정 크기 창 |
| single writer pattern | 단일 writer 패턴 | 하나의 스레드만 쓰기를 담당하는 동기화 패턴 |
| synchronization | 동기화 | 스레드들이 발맞춰 실행되도록 맞추는 것 |
| thread | 스레드 | 하나의 데이터를 처리하는 가장 작은 실행 단위 |
| thread divergence | 스레드 분기 | 같은 Warp 내 스레드들이 서로 다른 분기를 타는 현상 |
| thread specialization | 스레드 특화 | 스레드 그룹마다 서로 다른 알고리즘을 실행하는 패턴. 데이터 병렬 처리와 대비 |
| topology | 토폴로지 | 통신 또는 연결 구조의 형태. butterfly 네트워크의 Lane 간 데이터 교환 패턴 |
| transpose | 전치 | 행렬의 행과 열을 뒤바꾸는 연산. \\(A^T\\)로 표기 |
| Undefined Behavior | 미정의 동작 | 프로그램의 동작이 정의되지 않은 상태 |
| zero padding | 제로 패딩 | 배열 경계 밖을 0으로 채우는 기법. convolution 경계 처리에 사용 |
| zero-cost abstraction | 제로 코스트 추상화 | 추상화해도 성능 손실 없이 머신 코드로 컴파일됨 |

---

## 기여하기

용어 추가나 수정이 필요한 경우 이슈나 PR을 통해 제안해 주세요.
