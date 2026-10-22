#!/bin/bash
# build_xla2sync.sh: mfem_builds/xla2sync = the xla2 build (same cache, same fallback list) through shim-xla2sync
# (libRaise_best4sync.so: cudaDeviceSynchronize lowered to reactantXLASync; links rt/paper18). Builds mfem + benchmarks only.
# Log logs/mfem3_build2_xla2sync.log, marker MFEM3-BUILD-DONE.
E=/home/wmoses/ejwork; n=xla2sync; j=${JOBS:-24}; L=$E/logs/mfem3_build2_$n.log
mkdir -p $E/mfem_builds/$n
sed "s#/home/wmoses/ejwork/mfem_builds/xla2\b#/home/wmoses/ejwork/mfem_builds/$n#g; s#mfem_builds/shim-xla2\$#mfem_builds/shim-xla2sync#; s#mfem_builds/shim-xla2;#mfem_builds/shim-xla2sync;#" $E/mfem_builds/xla2/CMakeCache.txt > $E/mfem_builds/$n/CMakeCache.txt
cmake -S /mnt3/wmoses/git/mfem -B $E/mfem_builds/$n -DEnzyme_DIR=$E/mfem_builds/shim-xla2sync -DCMAKE_CUDA_COMPILER_LAUNCHER=$E/scripts/mfem3_launch.sh -DCMAKE_CXX_COMPILER_LAUNCHER=$E/scripts/mfem3_launch.sh > $L 2>&1 || { echo "MFEM3-BUILD-DONE $n cmake failed" >> $L; exit 1; }
grep -c best4sync $E/mfem_builds/$n/build.ninja >> $L
start=$(date +%s)
env -u EXPORT_REACTANT MFEM3_FALLBACK=$E/results/mfem3_xla_fallback.txt MFEM3_LAUNCH_LOG=$E/logs/mfem3_launch_$n.log \
  REACTANT_LOWER_INVOKE=0 REACTANT_SPECIALIZE_INDEX_STRIDES=1 REACTANT_SPECIALIZE_INT_SCALARS=1 \
  REACTANT_UNROLL_MAX_TRIP_COUNT=0 REACTANT_UNROLL_MAX_NESTED_OPS=4096 \
  ninja -C $E/mfem_builds/$n -k 0 -j $j mfem benchmarks >> $L 2>&1
rc=$?
echo "MFEM3-BUILD-DONE $n rc=$rc secs=$(( $(date +%s) - start )) failed=$(grep -c '^FAILED:' $L)" >> $L
