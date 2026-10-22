# MFEM native CUDA vs XLA-GPU: how the paper table was built and measured

This branch is the tree state behind the "paper20" table (2026-10-20), plus the
scripts and data that produced it. Nothing here is meant for upstream as is.

## What is on the branch

1. `tests/benchmarks/bench_assembly_levels.cpp`: `x.Randomize(1)` before
   `FormLinearSystem` (the stock source never initializes x, which the Dirichlet
   values are taken from; both backends ran the patched copy, as
   `bench_assembly_levels_x0`). `tests/benchmarks/bench_tmop.cpp`: an
   `MFEM_BENCH_CHECK` hook that prints the norms of one call of each kernel, used
   to compare the two backends' results (off unless the variable is set).
2. The working tree as it was built: three small source edits that the tree
   carried, and the `.re_export` files (the kernels as raised to StableHLO) as the
   Oct 2026 toolchain regenerates them.
3. `xla-paper/`: this file, the build and benchmark scripts, the configure flags,
   the benchmark subset and the results.

## Toolchain (all commits reachable)

| piece | where |
|---|---|
| clang 24 / LLVM | `/home/wmoses/git/Enzyme/llvm-project` at LLVM `b08e0059`, `buildRelease` |
| Reactant clang plugin (`libClangReactant`) | EnzymeAD/Reactant branch `tmp/perf-plugin` (`ff3dbe67`), built against that LLVM; must contain EnzymeAD/Reactant #89 |
| raising library (`libRaise.so`) | EnzymeAD/Enzyme-JAX branch `paper/mfem-raise-2026-10` (`4a2868384`): `tmp/best4` = main `73603ba52` + the AffineCFG/scatter branches merged there + the `cudaDeviceSynchronize -> reactantXLASync` lowering (Enzyme-JAX #3641/#3668); `bazel build //:libRaise.so` |
| runtime (`libReactantExtra.so`) | EnzymeAD/Reactant.jl branch `paper/mfem-runtime-2026-10` (`6b13d7ad4`), built with `--override_repository=enzyme_ad=<EnzymeAD/Enzyme-JAX paper/mfem-rextra-2026-10, 19c8cabce>` (`scripts/build_paper20.sh` has the exact bazel command, `--config=cuda12`) |

The XLA build links the benchmarks against that runtime (`shim/EnzymeConfig.cmake`,
`INTERFACE_LINK_LIBRARIES`); at run time `LD_LIBRARY_PATH` selects it.

## Building MFEM

Both builds: `cmake -S mfem -B <dir> -DEnzyme_DIR=<shim dir>` with the flags in
`cmake-cache-{native,xla}.txt` (Release, `-O3 -DNDEBUG`, CUDA `sm_120`, MPI + HYPRE +
METIS, benchmarks on, google-benchmark at `BENCHMARK_DIR`), compiler launcher
`scripts/mfem3_launch.sh`, then `ninja mfem benchmarks`. The native build is the
same clang with the ClangEnzyme plugin (MFEM's CUDA kernels compiled by clang, not
nvcc). The XLA build is `shim/EnzymeConfig.cmake` (the Reactant plugin,
`-reactant-backend xla-gpu`, `libRaise`), with the environment
`scripts/build_xla2sync.sh` sets (`REACTANT_LOWER_INVOKE=0
REACTANT_SPECIALIZE_INDEX_STRIDES=1 REACTANT_SPECIALIZE_INT_SCALARS=1
REACTANT_UNROLL_MAX_TRIP_COUNT=0 REACTANT_UNROLL_MAX_NESTED_OPS=4096`); the three
TUs in `mfem3_xla_fallback.txt` are compiled without the plugin (the launcher
does that). `bench_bpl` and the two `bench_dfem*` do not build through the
plugin (`__enzyme_fwddiff` inside a kernel; a vdim mismatch in `fem/kernels.hpp`
`Eval2d` on vanilla clang too) — bpl is native-only in the table.
`bench_assembly_levels_x0` is the stock target's compile and link commands over
the patched source (`scripts/build_x0.sh`; on this branch the patch is in place,
so the stock target is the same binary).

## Measuring

`scripts/bench_paper20pair.sh` → `scripts/mfem3_bench_pair.sh native xla2sync 0
mfem3_bench_final.txt`: for each `<binary> <filter>` line of the subset, the
native binary and then the XLA binary, back to back on one RTX 5090 (so the
host-bound suites see the same machine state), each run gated on the GPU being
idle and the 1-minute load being below 30; Google Benchmark with
`--benchmark_context=device=cuda --benchmark_min_warmup_time=1
--benchmark_repetitions=3`; XLA runs with `REACTANT_EXEC_DOT_FORMATION=0
REACTANT_EXEC_UNROLL=8` and `LD_LIBRARY_PATH` at the runtime. The benchmarks'
`MFEM_DEVICE_SYNC` is honoured on both backends (the runtime's `reactantXLASync`);
without it the XLA column reads the host's enqueue rate wherever that is faster
than the GPU.

`scripts/paper_table.py paper20 xla2off` makes `results/paper20.json` from the
benchmark JSON, `paper_tex.py` the two-page table (`paper20_cases_{1,2}.tex`,
`\label{tab:mfem-performance}`), `paper_site.py` the data page with
`results/meta_paper20.json` (the method and the notes). `cmp_runs.py A B thresh`
compares two runs.

## Results (paper20, 2026-10-20 22:05–22:48, load 23–25 of 192 cores)

XLA-GPU over native, geometric mean of the per-case ratios: assembly_levels 1.05,
ceed 1.62, dg_amr 1.73, elasticity 1.74, nlvc 1.13, tmop 0.86, vector 0.98,
virtuals 1.06; bpl native-only. Per case: `results/paper20.json`.
