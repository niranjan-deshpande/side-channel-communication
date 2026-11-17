#!/usr/bin/env bash
# Step-size channel characterization (physics.py) on the pod. Each run = a fresh engine start with its own compile caches.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results
one() {  # name inv args...
  local name=$1 inv=$2; shift 2
  test -f "results/$name/complete" && return
  rm -rf "results/$name"; local c=/root/cc/$name; rm -rf $c; mkdir -p $c
  VLLM_BATCH_INVARIANT=$inv VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t \
    python physics.py "$@" > "logs/$name.log" 2>&1 || echo "FAILED $name"
  echo "finished $name"
}
case "${1:-all}" in
  smoke)
    one smoke_transfer 0 transfer smoke_transfer --coarse --seed 5
    one smoke_traffic 0 traffic smoke_traffic --rates 0.25 --receivers 10 --seed 5 ;;
  all)
    one transfer_fit 0 transfer transfer_fit --seed 11
    one transfer_hold 0 transfer transfer_hold --coarse --seed 12
    one traffic_fit 0 traffic traffic_fit --seed 21 --rates ${RATES:-0.1 0.25 0.5 1.0}
    one traffic_hold 0 traffic traffic_hold --seed 22 --rates ${RATES:-0.1 0.25 0.5 1.0}
    one traffic_inv 1 traffic traffic_inv --seed 23 --rates ${INV_RATES:-0.1 0.5} --receivers 60 ;;
esac
echo DONE
