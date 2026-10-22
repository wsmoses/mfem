#!/bin/bash
# gpu_wait_idle.sh <gpu-index> [max-wait-s]: block until the GPU is free -- under 3 GB used and 0% utilization on 3
# samples 2 s apart (another user's jobs share this machine and hold ~25 GB per GPU; XLA OOMs next to them), up to
# max-wait (default 8 h). Prints how long it waited.
g=$1; max=${2:-28800}; t=0
while :; do
  ok=1
  for i in 1 2 3; do
    read u m < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i $g | tr -d ',')
    { [ "$u" -gt 0 ] || [ "$m" -gt 3000 ]; } && ok=0; sleep 2
  done
  [ $ok = 1 ] && break
  sleep 24; t=$((t+30)); [ $t -ge $max ] && break
done
echo "waited ${t}s for gpu $g; used ${m} MiB $(date -Is)"
