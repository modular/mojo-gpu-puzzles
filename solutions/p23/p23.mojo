# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from max.gpu import thread_idx, block_dim, block_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.compile import get_gpu_target
from layout import TileTensor, TensorEngine
from layout.layout import row_major, TensorLayout
from layout.tile_tensor import stack_allocation
from std.utils.coord import Coord
from std.math import ceildiv, log2
from std.algorithm.functional import vectorize

from max.algorithm.functional import elementwise
from std.sys import simd_width_of, argv, align_of
from std.testing import assert_equal
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from max.benchmark import bencher_iter_custom

from harness.canary import PuzzleMemory

comptime SIZE = 1024
comptime rank = 1
comptime layout = row_major[SIZE]()
comptime LayoutType = type_of(layout)
comptime dtype = DType.float32
comptime SIMD_WIDTH = simd_width_of[dtype, target=get_gpu_target()]()


# ANCHOR: elementwise_add_solution
def elementwise_add[
    LayoutT: TensorLayout,
    dtype: DType,
    simd_width: Int,
    rank: Int,
    size: Int,
    Engine: TensorEngine,
](
    output: TileTensor[mut=True, dtype, LayoutT, MutAnyOrigin, Engine=Engine],
    a: TileTensor[mut=False, dtype, LayoutT, ImmutAnyOrigin, Engine=Engine],
    b: TileTensor[mut=False, dtype, LayoutT, ImmutAnyOrigin, Engine=Engine],
    ctx: DeviceContext[],
) raises where (Engine.element_size == 1):
    @inline(.always)
    @__allow_legacy_any_origin_fields
    def add[simd_width: Int, alignment: Int = 1](indices: Coord) {var} -> None:
        var idx = Int(indices[0])
        # Note: This is thread-local SIMD - each thread processes its own vector of data
        # we'll later better see this hierarchy in Mojo:
        # SIMD within threads, warp across threads, block across warps
        var a_simd = a.load[width=simd_width](Coord(idx))
        var b_simd = b.load[width=simd_width](Coord(idx))
        var ret = a_simd + b_simd
        output.store[simd_width](Coord(idx), ret)

    elementwise[simd_width=SIMD_WIDTH](add, Coord(size), ctx)


# ANCHOR_END: elementwise_add_solution


# ANCHOR: tiled_elementwise_add_solution
comptime TILE_SIZE = 32


def tiled_elementwise_add[
    LayoutT: TensorLayout,
    dtype: DType,
    simd_width: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
    Engine: TensorEngine,
](
    output: TileTensor[mut=True, dtype, LayoutT, MutAnyOrigin, Engine=Engine],
    a: TileTensor[mut=False, dtype, LayoutT, ImmutAnyOrigin, Engine=Engine],
    b: TileTensor[mut=False, dtype, LayoutT, ImmutAnyOrigin, Engine=Engine],
    ctx: DeviceContext[],
) raises where (Engine.element_size == 1):
    @inline(.always)
    @__allow_legacy_any_origin_fields
    def process_tiles[
        simd_width: Int, alignment: Int = 1
    ](indices: Coord) {var} -> None:
        var tile_id = Int(indices[0])

        var output_tile = output.tile[tile_size](tile_id)
        var a_tile = a.tile[tile_size](tile_id)
        var b_tile = b.tile[tile_size](tile_id)

        comptime for i in range(tile_size):
            var a_vec = a_tile.load[width=simd_width](Coord(i))
            var b_vec = b_tile.load[width=simd_width](Coord(i))
            var ret = a_vec + b_vec
            output_tile.store[simd_width](Coord(i), ret)

    var num_tiles = ceildiv(size, tile_size)
    elementwise[simd_width=1](process_tiles, Coord(num_tiles), ctx)


# ANCHOR_END: tiled_elementwise_add_solution


# ANCHOR: manual_vectorized_tiled_elementwise_add_solution
def manual_vectorized_tiled_elementwise_add[
    LayoutT: TensorLayout,
    dtype: DType,
    simd_width: Int,
    num_tiles_per_call: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
    Engine: TensorEngine,
](
    output: TileTensor[mut=True, dtype, LayoutT, MutAnyOrigin, Engine=Engine],
    a: TileTensor[mut=False, dtype, LayoutT, ImmutAnyOrigin, Engine=Engine],
    b: TileTensor[mut=False, dtype, LayoutT, ImmutAnyOrigin, Engine=Engine],
    ctx: DeviceContext[],
) raises where (Engine.element_size == 1):
    # Each tile contains tile_size groups of simd_width elements
    comptime chunk_size = tile_size * simd_width

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def process_manual_vectorized_tiles[
        tile_count: Int, alignment: Int = 1
    ](indices: Coord) {var} -> None:
        # This call owns tiles indices[0] .. indices[0] + tile_count - 1
        comptime for t in range(tile_count):
            var tile_id = Int(indices[0]) + t

            comptime for i in range(tile_size):
                var global_start = tile_id * chunk_size + i * simd_width

                var a_vec = a.load[width=simd_width](Coord(global_start))
                var b_vec = b.load[width=simd_width](Coord(global_start))
                var ret = a_vec + b_vec
                output.store[simd_width](Coord(global_start), ret)

    # Number of tiles needed: each tile processes chunk_size elements
    var num_tiles = ceildiv(size, chunk_size)
    # The index space is tile ids, not elements, so elementwise's simd_width
    # is how many consecutive tiles each call gets, not a SIMD width. It calls
    # process_manual_vectorized_tiles[num_tiles_per_call] on multiples of
    # num_tiles_per_call, and [1] on any leftover tiles at the end.
    elementwise[simd_width=num_tiles_per_call](
        process_manual_vectorized_tiles, Coord(num_tiles), ctx
    )


# ANCHOR_END: manual_vectorized_tiled_elementwise_add_solution


# ANCHOR: vectorize_within_tiles_elementwise_add_solution
def vectorize_within_tiles_elementwise_add[
    LayoutT: TensorLayout,
    dtype: DType,
    simd_width: Int,
    num_tiles_per_call: Int,
    rank: Int,
    size: Int,
    tile_size: Int,
    Engine: TensorEngine,
](
    output: TileTensor[mut=True, dtype, LayoutT, MutAnyOrigin, Engine=Engine],
    a: TileTensor[mut=False, dtype, LayoutT, ImmutAnyOrigin, Engine=Engine],
    b: TileTensor[mut=False, dtype, LayoutT, ImmutAnyOrigin, Engine=Engine],
    ctx: DeviceContext[],
) raises where (Engine.element_size == 1):
    # Each tile contains tile_size elements (not SIMD groups)
    @inline(.always)
    @__allow_legacy_any_origin_fields
    def process_tile_with_vectorize[
        tile_count: Int, alignment: Int = 1
    ](indices: Coord) {var} -> None:
        # This call owns tiles indices[0] .. indices[0] + tile_count - 1
        comptime for t in range(tile_count):
            var tile_id = Int(indices[0]) + t
            var tile_start = tile_id * tile_size
            var tile_end = min(tile_start + tile_size, size)
            var actual_tile_size = tile_end - tile_start

            def vectorized_add[
                width: Int
            ](i: Int) {imm tile_start, var a, var b, var output}:
                var global_idx = tile_start + i
                if global_idx + width <= size:
                    var a_vec = a.load[width](Coord(global_idx))
                    var b_vec = b.load[width](Coord(global_idx))
                    var result = a_vec + b_vec
                    output.store[width](Coord(global_idx), result)

            # Use vectorize within each tile
            vectorize[simd_width](actual_tile_size, vectorized_add)

    var num_tiles = ceildiv(size, tile_size)
    # The index space is tile ids, not elements, so elementwise's simd_width
    # is how many consecutive tiles each call gets, not a SIMD width. It calls
    # process_tile_with_vectorize[num_tiles_per_call] on multiples of
    # num_tiles_per_call, and [1] on any leftover tiles at the end.
    elementwise[simd_width=num_tiles_per_call](
        process_tile_with_vectorize, Coord(num_tiles), ctx
    )


# ANCHOR_END: vectorize_within_tiles_elementwise_add_solution


@inline(.always)
def benchmark_elementwise_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    var bench_ctx = DeviceContext()
    comptime bench_layout = row_major[test_size]()
    comptime BenchLayoutType = type_of(bench_layout)
    var out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)

    var a_tensor = TileTensor(a, bench_layout).as_unsafe_any_origin().as_imm()
    var b_tensor = (
        TileTensor(b_buf, bench_layout).as_unsafe_any_origin().as_imm()
    )
    var out_tensor = TileTensor(out, bench_layout)

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def elementwise_workflow(ctx: DeviceContext[]) raises {imm}:
        elementwise_add[
            BenchLayoutType,
            dtype,
            SIMD_WIDTH,
            rank,
            test_size,
            out_tensor.Engine,
        ](out_tensor, a_tensor, b_tensor, ctx)

    bencher_iter_custom(b, elementwise_workflow, bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


@inline(.always)
def benchmark_tiled_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    var bench_ctx = DeviceContext()
    comptime bench_layout = row_major[test_size]()
    comptime BenchLayoutType = type_of(bench_layout)
    var out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)

    var a_tensor = TileTensor(a, bench_layout).as_unsafe_any_origin().as_imm()
    var b_tensor = (
        TileTensor(b_buf, bench_layout).as_unsafe_any_origin().as_imm()
    )
    var out_tensor = TileTensor(out, bench_layout)

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def tiled_workflow(ctx: DeviceContext[]) raises {imm}:
        tiled_elementwise_add[
            BenchLayoutType,
            dtype,
            SIMD_WIDTH,
            rank,
            test_size,
            tile_size,
            out_tensor.Engine,
        ](out_tensor, a_tensor, b_tensor, ctx)

    bencher_iter_custom(b, tiled_workflow, bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


@inline(.always)
def benchmark_manual_vectorized_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    var bench_ctx = DeviceContext()
    comptime bench_layout = row_major[test_size]()
    comptime BenchLayoutType = type_of(bench_layout)
    var out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)

    var a_tensor = TileTensor(a, bench_layout).as_unsafe_any_origin().as_imm()
    var b_tensor = (
        TileTensor(b_buf, bench_layout).as_unsafe_any_origin().as_imm()
    )
    var out_tensor = TileTensor(out, bench_layout)

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def manual_vectorized_workflow(ctx: DeviceContext[]) raises {imm}:
        manual_vectorized_tiled_elementwise_add[
            BenchLayoutType,
            dtype,
            SIMD_WIDTH,
            1,
            rank,
            test_size,
            tile_size,
            out_tensor.Engine,
        ](out_tensor, a_tensor, b_tensor, ctx)

    bencher_iter_custom(b, manual_vectorized_workflow, bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


@inline(.always)
def benchmark_vectorized_parameterized[
    test_size: Int, tile_size: Int
](mut b: Bencher) raises:
    var bench_ctx = DeviceContext()
    comptime bench_layout = row_major[test_size]()
    comptime BenchLayoutType = type_of(bench_layout)
    var out = bench_ctx.enqueue_create_buffer[dtype](test_size)
    out.enqueue_fill(0)
    var a = bench_ctx.enqueue_create_buffer[dtype](test_size)
    a.enqueue_fill(0)
    var b_buf = bench_ctx.enqueue_create_buffer[dtype](test_size)
    b_buf.enqueue_fill(0)

    with a.map_to_host() as a_host, b_buf.map_to_host() as b_host:
        for i in range(test_size):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)

    var a_tensor = TileTensor(a, bench_layout).as_unsafe_any_origin().as_imm()
    var b_tensor = (
        TileTensor(b_buf, bench_layout).as_unsafe_any_origin().as_imm()
    )
    var out_tensor = TileTensor(out, bench_layout)

    @inline(.always)
    @__allow_legacy_any_origin_fields
    def vectorized_workflow(ctx: DeviceContext[]) raises {imm}:
        vectorize_within_tiles_elementwise_add[
            BenchLayoutType,
            dtype,
            SIMD_WIDTH,
            1,
            rank,
            test_size,
            tile_size,
            out_tensor.Engine,
        ](out_tensor, a_tensor, b_tensor, ctx)

    bencher_iter_custom(b, vectorized_workflow, bench_ctx)
    keep(out.unsafe_ptr())
    bench_ctx.synchronize()


def check_output(
    result: DeviceBuffer[dtype], expected: HostBuffer[dtype]
) raises:
    with result.map_to_host() as out_host:
        print("out:", out_host)
        print("expected:", expected)
        for i in range(SIZE):
            assert_equal(out_host[i], expected[i])


def check_manual_vectorized[
    num_tiles_per_call: Int
](
    mut mem: PuzzleMemory[dtype],
    a: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
    expected: HostBuffer[dtype],
    ctx: DeviceContext[],
) raises:
    print("num_tiles_per_call:", num_tiles_per_call)
    # A fresh zeroed buffer per run, so one run can't pass on another's output
    var out = mem.output(SIZE)
    var out_tensor = TileTensor(out, layout)
    var a_tensor = TileTensor(a, layout).as_unsafe_any_origin().as_imm()
    var b_tensor = TileTensor(b, layout).as_unsafe_any_origin().as_imm()
    manual_vectorized_tiled_elementwise_add[
        LayoutType,
        dtype,
        SIMD_WIDTH,
        num_tiles_per_call,
        rank,
        SIZE,
        TILE_SIZE,
        out_tensor.Engine,
    ](out_tensor, a_tensor, b_tensor, ctx)
    check_output(out, expected)


def check_vectorized[
    num_tiles_per_call: Int
](
    mut mem: PuzzleMemory[dtype],
    a: DeviceBuffer[dtype],
    b: DeviceBuffer[dtype],
    expected: HostBuffer[dtype],
    ctx: DeviceContext[],
) raises:
    print("num_tiles_per_call:", num_tiles_per_call)
    # A fresh zeroed buffer per run, so one run can't pass on another's output
    var out = mem.output(SIZE)
    var out_tensor = TileTensor(out, layout)
    var a_tensor = TileTensor(a, layout).as_unsafe_any_origin().as_imm()
    var b_tensor = TileTensor(b, layout).as_unsafe_any_origin().as_imm()
    vectorize_within_tiles_elementwise_add[
        LayoutType,
        dtype,
        SIMD_WIDTH,
        num_tiles_per_call,
        rank,
        SIZE,
        TILE_SIZE,
        out_tensor.Engine,
    ](out_tensor, a_tensor, b_tensor, ctx)
    check_output(out, expected)


def main() raises:
    var ctx = DeviceContext()
    var mem = PuzzleMemory[dtype](ctx)
    var out = mem.output(SIZE)
    var a = ctx.enqueue_create_buffer[dtype](SIZE)
    a.enqueue_fill(0)
    var b = ctx.enqueue_create_buffer[dtype](SIZE)
    b.enqueue_fill(0)
    var expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
    expected.enqueue_fill(0)

    with a.map_to_host() as a_host, b.map_to_host() as b_host:
        for i in range(SIZE):
            a_host[i] = Scalar[dtype](2 * i)
            b_host[i] = Scalar[dtype](2 * i + 1)
            expected[i] = a_host[i] + b_host[i]

    var a_tensor = TileTensor(a, layout).as_unsafe_any_origin().as_imm()
    var b_tensor = TileTensor(b, layout).as_unsafe_any_origin().as_imm()

    ctx.synchronize()

    print("SIZE:", SIZE)
    print("simd_width:", SIMD_WIDTH)

    if argv()[1] == "--elementwise":
        var out_tensor = TileTensor(out, layout)
        elementwise_add[
            LayoutType, dtype, SIMD_WIDTH, rank, SIZE, out_tensor.Engine
        ](out_tensor, a_tensor, b_tensor, ctx)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 23 complete ✅")

    elif argv()[1] == "--tiled":
        var out_tensor = TileTensor(out, layout)
        print("tile size:", TILE_SIZE)
        tiled_elementwise_add[
            LayoutType,
            dtype,
            SIMD_WIDTH,
            rank,
            SIZE,
            TILE_SIZE,
            out_tensor.Engine,
        ](out_tensor, a_tensor, b_tensor, ctx)

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
            print("Puzzle 23 complete ✅")

    elif argv()[1] == "--manual-vectorized":
        print("tile size:", TILE_SIZE)
        # 1 tile per call is the launch the explanations walk through. 4 checks
        # that a call covers every tile it is handed, and 3 doesn't divide the
        # tile count, so elementwise also makes 1-tile calls for the leftovers.
        check_manual_vectorized[1](mem, a, b, expected, ctx)
        check_manual_vectorized[3](mem, a, b, expected, ctx)
        check_manual_vectorized[4](mem, a, b, expected, ctx)
        print("Puzzle 23 complete ✅")

    elif argv()[1] == "--vectorized":
        print("tile size:", TILE_SIZE)
        # 1 tile per call is the launch the explanations walk through. 4 checks
        # that a call covers every tile it is handed, and 3 doesn't divide the
        # tile count, so elementwise also makes 1-tile calls for the leftovers.
        check_vectorized[1](mem, a, b, expected, ctx)
        check_vectorized[3](mem, a, b, expected, ctx)
        check_vectorized[4](mem, a, b, expected, ctx)
        print("Puzzle 23 complete ✅")

    elif argv()[1] == "--benchmark":
        print("Running P23 GPU Benchmarks...")
        print("SIMD width:", SIMD_WIDTH)
        print("-" * 80)
        var bench_config = BenchConfig(max_iters=10, num_warmup_iters=1)
        var bench = Bench(bench_config.copy())

        print("Testing SIZE=16, TILE=4")
        bench.bench_function(
            lambda (mut b: Bencher) raises: benchmark_elementwise_parameterized[
                16, 4
            ](b),
            BenchId("elementwise_16_4"),
        )
        bench.bench_function(
            lambda (mut b: Bencher) raises: benchmark_tiled_parameterized[
                16, 4
            ](b),
            BenchId("tiled_16_4"),
        )
        bench.bench_function(
            lambda (
                mut b: Bencher
            ) raises: benchmark_manual_vectorized_parameterized[16, 4](b),
            BenchId("manual_vectorized_16_4"),
        )
        bench.bench_function(
            lambda (mut b: Bencher) raises: benchmark_vectorized_parameterized[
                16, 4
            ](b),
            BenchId("vectorized_16_4"),
        )

        print("-" * 80)
        print("Testing SIZE=128, TILE=16")
        bench.bench_function(
            lambda (mut b: Bencher) raises: benchmark_elementwise_parameterized[
                128, 16
            ](b),
            BenchId("elementwise_128_16"),
        )
        bench.bench_function(
            lambda (mut b: Bencher) raises: benchmark_tiled_parameterized[
                128, 16
            ](b),
            BenchId("tiled_128_16"),
        )
        bench.bench_function(
            lambda (
                mut b: Bencher
            ) raises: benchmark_manual_vectorized_parameterized[128, 16](b),
            BenchId("manual_vectorized_128_16"),
        )

        print("-" * 80)
        print("Testing SIZE=128, TILE=16, Vectorize within tiles")
        bench.bench_function(
            lambda (mut b: Bencher) raises: benchmark_vectorized_parameterized[
                128, 16
            ](b),
            BenchId("vectorized_128_16"),
        )

        print("-" * 80)
        print("Testing SIZE=1048576 (1M), TILE=1024")
        bench.bench_function(
            lambda (mut b: Bencher) raises: benchmark_elementwise_parameterized[
                1048576, 1024
            ](b),
            BenchId("elementwise_1M_1024"),
        )
        bench.bench_function(
            lambda (mut b: Bencher) raises: benchmark_tiled_parameterized[
                1048576, 1024
            ](b),
            BenchId("tiled_1M_1024"),
        )
        bench.bench_function(
            lambda (
                mut b: Bencher
            ) raises: benchmark_manual_vectorized_parameterized[1048576, 1024](
                b
            ),
            BenchId("manual_vectorized_1M_1024"),
        )
        bench.bench_function(
            lambda (mut b: Bencher) raises: benchmark_vectorized_parameterized[
                1048576, 1024
            ](b),
            BenchId("vectorized_1M_1024"),
        )

        print(bench)
        print("Benchmarks completed!")

    else:
        print(
            "Usage: --elementwise | --tiled | --manual-vectorized |"
            " --vectorized | --benchmark"
        )
    mem.verify()
