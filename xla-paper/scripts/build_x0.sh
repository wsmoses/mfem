#!/bin/bash
# build_x0.sh <build>: bench_assembly_levels_x0 in mfem_builds/<build> = the stock bench_assembly_levels compile and link
# commands (ninja -t commands) over the patched source mfem_bench_patch/bench_assembly_levels.cpp (x initialized).
# Log logs/build_x0_<build>.log, marker X0-BUILT.
E=/home/wmoses/ejwork; b=$1; B=$E/mfem_builds/$b; L=$E/logs/build_x0_$b.log; cd $B || exit 1
ninja -t commands tests/benchmarks/bench_assembly_levels > $E/scratch/x0_cmds_$b.txt
cc=$(grep -- '-o tests/benchmarks/CMakeFiles/bench_assembly_levels.dir/bench_assembly_levels.cpp.o' $E/scratch/x0_cmds_$b.txt | tail -1)
ln=$(grep -- '-o tests/benchmarks/bench_assembly_levels ' $E/scratch/x0_cmds_$b.txt | tail -1)
[ -n "$cc" ] && [ -n "$ln" ] || { echo "no commands" > $L; exit 1; }
cc=${cc//tests\/benchmarks\/CMakeFiles\/bench_assembly_levels.dir\/bench_assembly_levels.cpp.o/tests\/benchmarks\/CMakeFiles\/bench_assembly_levels.dir\/bench_assembly_levels_x0.cpp.o}
cc=$(sed "s#-c [^ ]*tests/benchmarks/bench_assembly_levels.cpp#-c $E/mfem_bench_patch/bench_assembly_levels.cpp#" <<< "$cc")
ln=${ln//tests\/benchmarks\/CMakeFiles\/bench_assembly_levels.dir\/bench_assembly_levels.cpp.o/tests\/benchmarks\/CMakeFiles\/bench_assembly_levels.dir\/bench_assembly_levels_x0.cpp.o}
ln=${ln//-o tests\/benchmarks\/bench_assembly_levels /-o tests\/benchmarks\/bench_assembly_levels_x0 }
{ echo "$cc"; echo "$ln"; } > $L
env -u EXPORT_REACTANT MFEM3_FALLBACK=$E/results/mfem3_xla_fallback.txt MFEM3_LAUNCH_LOG=$E/logs/mfem3_launch_$b.log \
  REACTANT_LOWER_INVOKE=0 REACTANT_SPECIALIZE_INDEX_STRIDES=1 REACTANT_SPECIALIZE_INT_SCALARS=1 \
  REACTANT_UNROLL_MAX_TRIP_COUNT=0 REACTANT_UNROLL_MAX_NESTED_OPS=4096 bash -c "$cc" >> $L 2>&1 && bash -c "$ln" >> $L 2>&1 && echo X0-BUILT >> $L
tail -1 $L
