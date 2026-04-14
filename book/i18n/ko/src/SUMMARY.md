<!-- i18n-source-commit: 704b538202746549a7731989c173fdc49f00c5c2 -->

# 목차

# 시작하기

- [🔥 소개](./introduction.md)
- [🧭 퍼즐 사용 가이드](./howto.md)
- [🏆 보상을 받아가세요](./reward.md)

# Part I: GPU 기초

- [Puzzle 1: Map](./puzzle_01/puzzle_01.md)
  - [🔰 원시 메모리 방식](./puzzle_01/raw.md)
  - [💡 미리보기: TileTensor를 활용한 현대적 방식](./puzzle_01/tile_tensor_preview.md)
- [Puzzle 2: Zip](./puzzle_02/puzzle_02.md)
- [Puzzle 3: 가드](./puzzle_03/puzzle_03.md)
- [Puzzle 4: 2D Map](./puzzle_04/puzzle_04.md)
  - [🔰 원시 메모리 방식](./puzzle_04/raw.md)
  - [📚 TileTensor 알아보기](./puzzle_04/introduction_tile_tensor.md)
  - [🚀 현대적 2D 연산](./puzzle_04/tile_tensor.md)
- [Puzzle 5: 브로드캐스트](./puzzle_05/puzzle_05.md)
- [Puzzle 6: 블록](./puzzle_06/puzzle_06.md)
- [Puzzle 7: 2D 블록](./puzzle_07/puzzle_07.md)
- [Puzzle 8: 공유 메모리](./puzzle_08/puzzle_08.md)

# Part II: 🐞 GPU 프로그램 디버깅

- [Puzzle 9: GPU 디버깅 워크플로우](./puzzle_09/puzzle_09.md)
  - [📚 Mojo GPU 디버깅의 핵심](./puzzle_09/essentials.md)
  - [🧐 탐정 수사: 첫 번째 사례](./puzzle_09/first_case.md)
  - [🔍 탐정 수사: 두 번째 사례](./puzzle_09/second_case.md)
  - [🕵 탐정 수사: 세 번째 사례](./puzzle_09/third_case.md)
- [Puzzle 10: 새니타이저로 메모리 오류와 경쟁 상태 찾기](./puzzle_10/puzzle_10.md)
  - [👮🏼‍♂️ 메모리 위반 탐지](./puzzle_10/memcheck.md)
  - [🏁 경쟁 상태 디버깅](./puzzle_10/racecheck.md)

# Part III: 🧮 GPU 알고리즘

- [Puzzle 11: 풀링](./puzzle_11/puzzle_11.md)
- [Puzzle 12: 내적](./puzzle_12/puzzle_12.md)
- [Puzzle 13: 1D 합성곱](./puzzle_13/puzzle_13.md)
  - [🔰 기본 버전](./puzzle_13/simple.md)
  - [⭐ 블록 경계 버전](./puzzle_13/block_boundary.md)
- [Puzzle 14: 누적 합](./puzzle_14/puzzle_14.md)
  - [🔰 기본 버전](./puzzle_14/simple.md)
  - [⭐ 완성 버전](./puzzle_14/complete.md)
- [Puzzle 15: 축 합계](./puzzle_15/puzzle_15.md)
- [Puzzle 16: 행렬 곱셈 (MatMul)](./puzzle_16/puzzle_16.md)
  - [🔰 전역 메모리를 사용한 기본 버전](./puzzle_16/naïve.md)
  - [📚 루프라인 모델 알아보기](./puzzle_16/roofline.md)
  - [🤝 공유 메모리 버전](./puzzle_16/shared_memory.md)
  - [📐 타일링 버전](./puzzle_16/tiled.md)

# Part IV: 🐍 MAX 그래프 커스텀 Op으로 파이썬 연동하기

- [Puzzle 17: 1D 합성곱 Op](./puzzle_17/puzzle_17.md)
- [Puzzle 18: 소프트맥스 Op](./puzzle_18/puzzle_18.md)
- [Puzzle 19: 어텐션 Op](./puzzle_19/puzzle_19.md)
- [🎯 보너스 챌린지](./bonuses/part4.md)

# Part V: 🔥 PyTorch 커스텀 Op 통합하기

- [Puzzle 20: 1D 합성곱 Op](./puzzle_20/puzzle_20.md)
- [Puzzle 21: 임베딩 Op](./puzzle_21/puzzle_21.md)
  - [🔰 병합 vs 비병합 커널](./puzzle_21/simple_embedding_kernel.md)
  - [📊 성능 비교](./puzzle_21/performance.md)
- [Puzzle 22: 커널 퓨전과 커스텀 역방향 패스](./puzzle_22/puzzle_22.md)
  - [⚛️ 퓨전 vs 언퓨전 커널](./puzzle_22/forward_pass.md)
  - [⛓️ 오토그래드 통합과 역방향 패스](./puzzle_22/backward_pass.md)

# Part VI: 🌊 Mojo 함수형 패턴과 벤치마킹

- [Puzzle 23: GPU 함수형 프로그래밍 패턴](./puzzle_23/puzzle_23.md)
  - [elementwise - 기본 GPU 함수형 연산](./puzzle_23/elementwise.md)
  - [tile - 메모리 효율적인 타일링 처리](./puzzle_23/tile.md)
  - [vectorize - SIMD 제어](./puzzle_23/vectorize.md)
  - [🧠 GPU 스레딩 vs SIMD 개념](./puzzle_23/gpu-thread-vs-simd.md)
  - [📊 Mojo 벤치마킹](./puzzle_23/benchmarking.md)

# Part VII: ⚡ 워프 레벨 프로그래밍

- [Puzzle 24: 워프 기초](./puzzle_24/puzzle_24.md)
  - [🧠 워프 레인과 SIMT 실행](./puzzle_24/warp_simt.md)
  - [🔰 warp.sum()의 핵심](./puzzle_24/warp_sum.md)
  - [🤔 언제 워프 프로그래밍을 사용할까](./puzzle_24/warp_extra.md)
- [Puzzle 25: 워프 통신](./puzzle_25/puzzle_25.md)
  - [⬇️ warp.shuffle_down()](./puzzle_25/warp_shuffle_down.md)
  - [📢 warp.broadcast()](./puzzle_25/warp_broadcast.md)
- [Puzzle 26: 고급 워프 패턴](./puzzle_26/puzzle_26.md)
  - [🦋 warp.shuffle_xor()와 버터플라이 네트워크](./puzzle_26/warp_shuffle_xor.md)
  - [🔢 warp.prefix_sum()과 스캔 연산](./puzzle_26/warp_prefix_sum.md)

# Part VIII: 🧱 블록 레벨 프로그래밍

- [Puzzle 27: 블록 전체 패턴](./puzzle_27/puzzle_27.md)
  - [🔰 block.sum()의 핵심](./puzzle_27/block_sum.md)
  - [📈 block.prefix_sum()과 병렬 히스토그램 구간 분류](./puzzle_27/block_prefix_sum.md)
  - [📡 block.broadcast()와 벡터 정규화](./puzzle_27/block_broadcast.md)

# Part IX: 🧠 고급 메모리 시스템

- [Puzzle 28: 비동기 메모리 연산과 복사 중첩](./puzzle_28/puzzle_28.md)
- [Puzzle 29: GPU 동기화 기본 요소](./puzzle_29/puzzle_29.md)
  - [📶 다단계 파이프라인 조정](./puzzle_29/barrier.md)
  - [더블 버퍼링 스텐실 연산](./puzzle_29/memory_barrier.md)

# Part X: 📊 성능 분석과 최적화

- [Puzzle 30: GPU 프로파일링](./puzzle_30/puzzle_30.md)
  - [📚 NVIDIA 프로파일링 기초](./puzzle_30/nvidia_profiling_basics.md)
  - [🕵 캐시 히트의 역설](./puzzle_30/profile_kernels.md)
- [Puzzle 31: 점유율 최적화](./puzzle_31/puzzle_31.md)
- [Puzzle 32: 뱅크 충돌](./puzzle_32/puzzle_32.md)
  - [📚 공유 메모리 뱅크 이해하기](./puzzle_32/shared_memory_bank.md)
  - [충돌 없는 패턴](./puzzle_32/conflict_free_patterns.md)

# Part XI: 🚀 고급 GPU 기능

- [Puzzle 33: 텐서 코어 연산](./puzzle_33/puzzle_33.md)
  - [🎯 성능 보너스 챌린지](./bonuses/part5.md)
- [Puzzle 34: GPU 클러스터 프로그래밍 (SM90+)](./puzzle_34/puzzle_34.md)
  - [🔰 멀티 블록 조정 기초](./puzzle_34/cluster_coordination_basics.md)
  - [☸️ 클러스터 전체 집합 연산](./puzzle_34/cluster_collective_ops.md)
  - [🧠 고급 클러스터 알고리즘](./puzzle_34/advanced_cluster_patterns.md)
