#!/bin/bash
# Build runtime paper20: paper19 with concat_operand_barrier on by default in the barrier round (REACTANT_EXEC_NOCONCATROUND=1 disables).
E=/home/wmoses/ejwork
cd /home/wmoses/git/Reactant.jl/deps/ReactantExtra && nice -n 19 bazel build --jobs=24 --config=cuda12 --check_visibility=false --repo_env=HERMETIC_PYTHON_VERSION=3.11 --copt=-Wno-error=unused-command-line-argument --host_copt=-Wno-error=unused-command-line-argument --override_repository=enzyme_ad=$E/wt/rextra-src //:libReactantExtra.so > $E/logs/build_rextra_paper20.log 2>&1 || { echo PAPER20-BUILD-FAILED; exit 1; }
B=/mnt3/wmoses/.cache/bazel/_bazel_wmoses/938daab9a72fc0d5443aa9a534262d60/execroot/__main__/bazel-out/k8-opt/bin
rm -f $B/libReactantExtra_paper20.so; cp $B/libReactantExtra.so $B/libReactantExtra_paper20.so && mkdir -p $E/rt/paper20 && ln -sf $B/libReactantExtra_paper20.so $E/rt/paper20/libReactantExtra_stable.so && ln -sfn $B/_solib_local $E/rt/paper20/_solib_local
cd $E
echo PAPER20-BUILT
