#!/bin/bash
# bench_paper20pair.sh: the paper pair sweep native vs xla2sync on runtime paper20 (concat barrier round on by default,
# dots off), default length, REPS=3, GPU 0 -> results/mfem3/paper20 (+ the assembly_levels_x0 run), compared with paper18.
E=/home/wmoses/ejwork; cd $E
until grep -q 'PAPER20-BUILT\|PAPER20-BUILD-FAILED' logs/build_paper20.out 2>/dev/null; do sleep 60; done
grep -q PAPER20-BUILT logs/build_paper20.out || { echo build failed; echo PAIR20-DONE; exit 1; }
until [ $(cut -d. -f1 /proc/loadavg) -lt 26 ]; do sleep 120; done; echo "start $(uptime)"
LD_LIBRARY_PATH=$E/rt/paper20 REACTANT_EXEC_DOT_FORMATION=0 REPS=3 bash scripts/mfem3_bench_pair.sh native xla2sync 0 results/mfem3_bench_final.txt bench_paper20pair > logs/bench_paper20pair.log 2>&1
echo "end $(uptime)"
mkdir -p results/mfem3/paper20; ln -sfn ../bench_paper20pair/native results/mfem3/paper20/native; ln -sfn ../bench_paper20pair/xla2sync results/mfem3/paper20/xla2off
python3 scripts/paper_table.py paper20 xla2off
echo "== native paper18 vs paper20"; python3 scripts/cmp_runs.py results/mfem3/paper18/native results/mfem3/bench_paper20pair/native 1.10 | tail -4
echo "== xla paper18 vs paper20"; python3 scripts/cmp_runs.py results/mfem3/paper18/xla2off results/mfem3/bench_paper20pair/xla2sync 1.10 | tail -12
echo PAIR20-DONE
