# MFEM on XLA-GPU: setup, build, benchmark, reproduce the paper table

This branch (`xla-paper-2026-10`, from `dfem-dev` @ `93b234fdc`) is the MFEM tree
behind the native-CUDA vs XLA-GPU table of 2026-10-20 ("paper20"), with the
toolchain recipe, the scripts and the results in `xla-paper/`. Nothing here is
meant for upstream as is.

Contents of the branch: the benchmark patches (`bench_assembly_levels` initializes
the solution vector `FormLinearSystem` takes its Dirichlet values from; `bench_tmop`
has an `MFEM_BENCH_CHECK` hook that prints the norms of one call of each kernel),
the three small source edits the tree carried when it was built, the `.re_export`
files (each kernel as raised to StableHLO) as the toolchain regenerates them, and
`xla-paper/` (this file, `scripts/`, `shim/`, `results/`, the benchmark subset and
the configure flags).

## 1. Toolchain

Everything is built from pinned commits; `<work>` is a scratch root.

| piece | source | build |
|---|---|---|
| clang 24 (LLVM `b08e0059`) | llvm-project at that commit, as `Enzyme/llvm-project` `buildRelease` | an ordinary Release build with clang, lld, MLIR |
| Reactant clang plugin `libClangReactant` | EnzymeAD/Reactant, branch `tmp/perf-plugin` (`ff3dbe67`); it must contain EnzymeAD/Reactant #89 | `cmake` the Enzyme subdir against that LLVM; `ninja -C enzyme/buildRelease ClangReactant-24`; the product is `libClangReactant-24.so` (referenced below as `libClangReactant.so`) |
| raising library `libRaise.so` | EnzymeAD/Enzyme-JAX, branch `paper/mfem-raise-2026-10` (`4a2868384`) | `bazel build //:libRaise.so //:enzymexlamlir-opt` |
| runtime `libReactantExtra.so` | EnzymeAD/Reactant.jl, branch `paper/mfem-runtime-2026-10` (`6b13d7ad4`), against EnzymeAD/Enzyme-JAX branch `paper/mfem-rextra-2026-10` (`19c8cabce`) checked out at `<work>/rextra-src` | `cd deps/ReactantExtra && bazel build --config=cuda12 --check_visibility=false --repo_env=HERMETIC_PYTHON_VERSION=3.11 --copt=-Wno-error=unused-command-line-argument --host_copt=-Wno-error=unused-command-line-argument --override_repository=enzyme_ad=<work>/rextra-src //:libReactantExtra.so` (`scripts/build_paper20.sh`) |

Put the runtime where the benchmarks find it: a directory `<work>/rt` holding
`libReactantExtra_stable.so` (the built library, or a symlink to it) and
`_solib_local` (a symlink to bazel-bin's `_solib_local`, cuDNN and the CUDA
libraries resolve through it). `ldd libReactantExtra_stable.so | grep "not found"`
must be empty. The GPU box: one NVIDIA RTX 5090 (sm_120), driver 575, CUDA 12.9.

The plugin's `MLIRRoundTripOptions` struct must match `libRaise`'s `raise.h` field
for field (both 13 fields at these commits); a mismatch shows up as nonsense
options at compile time, not as a link error.

## 2. Building MFEM

Dependencies as in the configure flags: MPI (`mpicxx`), HYPRE (`../hypre`), METIS
4 (`../metis-4.0`), google-benchmark (`BENCHMARK_DIR`). Two build directories from
this tree:

**native** (MFEM's own CUDA kernels, compiled by clang with the ClangEnzyme plugin):

    cmake -S mfem -B build-native -DEnzyme_DIR=<shim-native> <flags from xla-paper/cmake-cache-native.txt>
    ninja -C build-native mfem benchmarks

**xla** (every kernel raised to StableHLO at compile time, run by XLA through the
runtime): copy `xla-paper/shim/EnzymeConfig.cmake` to a directory of its own and
edit its three paths (`-fplugin=.../libClangReactant.so`, `-raising-plugin-path
.../libRaise.so`, `INTERFACE_LINK_LIBRARIES` and the rpath to `<work>/rt`), then

    cmake -S mfem -B build-xla -DEnzyme_DIR=<that dir> \
      -DCMAKE_CXX_COMPILER_LAUNCHER=$PWD/xla-paper/scripts/mfem3_launch.sh \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=$PWD/xla-paper/scripts/mfem3_launch.sh \
      <flags from xla-paper/cmake-cache-xla.txt>
    env REACTANT_LOWER_INVOKE=0 REACTANT_SPECIALIZE_INDEX_STRIDES=1 REACTANT_SPECIALIZE_INT_SCALARS=1 \
        REACTANT_UNROLL_MAX_TRIP_COUNT=0 REACTANT_UNROLL_MAX_NESTED_OPS=4096 \
        MFEM3_FALLBACK=$PWD/xla-paper/mfem3_xla_fallback.txt \
        ninja -C build-xla -k 0 -j 24 mfem benchmarks

(`scripts/build_xla2sync.sh` is the script that did this.) The launcher compiles the
three TUs in `mfem3_xla_fallback.txt` without the plugin and retries a failed
compile once. `bench_bpl` and the two `bench_dfem*` targets do not build through
the plugin (`__enzyme_fwddiff` inside a kernel; a vdim mismatch in `fem/kernels.hpp`
`Eval2d` that fails on vanilla clang too) — bpl is native-only in the table; `-k 0`
carries on past them. The raising runs at compile time, so the XLA build takes
about 20 minutes at `-j 24` on 192 cores; a changed `libRaise` means a full
rebuild.

## 3. Running a benchmark

    export LD_LIBRARY_PATH=<work>/rt REACTANT_EXEC_DOT_FORMATION=0 REACTANT_EXEC_UNROLL=8 CUDA_VISIBLE_DEVICES=0
    build-xla/tests/benchmarks/bench_ceed --benchmark_context=device=cuda \
      --benchmark_filter='^BP1/2/96$' --benchmark_min_warmup_time=1 --benchmark_repetitions=3

The first call of each kernel compiles it (seconds to a minute for the large
ones); the 1 s warm-up keeps that out of the measurement. Runtime knobs used by the
table: `REACTANT_EXEC_DOT_FORMATION=0` (contractions stay as reductions, the GPU
default; `=1` forms `dot_general`s, which XLA's Triton GEMM miscompiles on BP6
unless `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`), `REACTANT_EXEC_UNROLL=8`
(the loop-unroll trip count of the exec pipeline), `REACTANT_EXEC_NOBARRIER=1`
(drops the barrier round), `REACTANT_EXEC_NOCONCATROUND=1` (drops
`concat_operand_barrier` from it), `REACTANT_EXEC_DUMP=1` (prints every kernel's
module before and after the exec pipeline), `REACTANT_EXEC_NO_SPECIALIZE=1`.

Every benchmark's `MFEM_DEVICE_SYNC` is honoured on both backends (the runtime's
`reactantXLASync` waits for the device). Without it — the lowering before
Enzyme-JAX #3641 erased the call — the XLA column reads the host's enqueue rate
wherever that is faster than the GPU (tmop `AddMultGradPA/0/4`: 1.23 ms instead of
2.58 ms).

## 4. The table

    xla-paper/scripts/bench_paper20pair.sh      # the full paired sweep, ~45 min

runs `mfem3_bench_pair.sh native xla2sync 0 mfem3_bench_final.txt bench_paper20pair`:
for each `<binary> <filter>` line of the subset the native binary and then the XLA
binary, back to back (the host-bound suites follow the machine's load, so both
columns must see the same state), each run gated on the GPU being idle and the
1-minute load being below 30, Google Benchmark with the flags above. Then

    python3 xla-paper/scripts/paper_table.py paper20 xla2off    # results/paper20.json
    python3 xla-paper/scripts/paper_tex.py paper20 xla2off      # the two-page table (\label{tab:mfem-performance})
    python3 xla-paper/scripts/paper_site.py paper20 page.html --meta results/meta_paper20.json
    python3 xla-paper/scripts/cmp_runs.py <runA dir> <runB dir> 1.1   # per-case B/A, geomean

(`paper_table.py` expects `results/mfem3/<tag>/{native,<xla build>}/bench_*.json`;
the scripts carry the paths of the machine they ran on at the top.) The
`bench_assembly_levels_x0` binary of the subset is the patched copy; on this branch
the stock `bench_assembly_levels` is the same source (`scripts/build_x0.sh` shows
how it was built out of tree).

## 5. Results (paper20)

2026-10-20 22:05–22:48, load 23–25 of 192 cores. XLA-GPU over native, geometric
mean of the per-case ratios: assembly_levels 1.05, ceed 1.62, dg_amr 1.73,
elasticity 1.74, nlvc 1.13, tmop 0.86, vector 0.98, virtuals 1.06. Per case with
all three repetitions: `results/paper20.json`; the method and the notes:
`results/meta_paper20.json`.

## 6. Checking results between the backends

    MFEM_BENCH_CHECK=1 build-xla/tests/benchmarks/bench_tmop --benchmark_context=device=cuda --benchmark_filter='^AddMultPA/0/2$' --benchmark_min_time=1x

prints the norms of one call of each tmop kernel; compare with the native binary's
line. For the other suites the benchmarks' own residual/norm output (e.g. ceed's CG
`nom`) serves the same purpose.
