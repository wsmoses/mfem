#!/bin/bash
# mfem3_bench_pair.sh <buildA> <buildB> <gpu> <subset-file> <tag>: like mfem3_bench_idle.sh, but runs each "<binary> <regex>"
# line of the subset with build A and then build B back to back, so both columns of a host-bound suite see the same
# machine state (another user's jobs change host-side timings by 1.5x over a day). Output layout as mfem3_bench_idle.sh.
# MIN_TIME=8s makes every case run at least that long: the raised benchmarks never wait for the GPU (cudaDeviceSynchronize
# is erased), so a short run reports the host enqueue rate until the launch queue saturates.
E=/home/wmoses/ejwork; a=$1; b=$2; gpu=$3; sub=$4; tag=$5
for bl in $a $b; do O=$E/results/mfem3/$tag/$bl; mkdir -p $O; : > $O/summary.txt; done
while read -r bin filt; do
  [ -z "$bin" ] && continue
  for bl in $a $b; do
    X=$E/mfem_builds/$bl/tests/benchmarks; O=$E/results/mfem3/$tag/$bl
    if [ ! -x $X/$bin ]; then echo "$bin|missing|0" >> $O/summary.txt; continue; fi
    bash $E/scripts/gpu_wait_idle.sh $gpu >> $O/waits.txt
    # The host-bound cases follow the machine load: wait for a quiet machine before every run (1-min load < MAX_LOAD, default 30).
    until [ $(cut -d. -f1 /proc/loadavg) -lt ${MAX_LOAD:-30} ]; do sleep 60; done; start=$(date +%s)
    (cd $O && CUDA_VISIBLE_DEVICES=$gpu REACTANT_EXEC_UNROLL=8 timeout 5400 $X/$bin --benchmark_context=device=cuda \
       --benchmark_filter="$filt" --benchmark_min_warmup_time=1 ${MIN_TIME:+--benchmark_min_time=$MIN_TIME} --benchmark_repetitions=${REPS:-1} --benchmark_out=$O/$bin.json --benchmark_out_format=json > $O/$bin.txt 2>&1)
    echo "$bin|$?|$(( $(date +%s) - start ))|load $(cut -d' ' -f1 /proc/loadavg)" >> $O/summary.txt
  done
done < $sub
for bl in $a $b; do echo "MFEM3-BENCH-DONE $bl $(date -Is)" >> $E/results/mfem3/$tag/$bl/summary.txt; done
