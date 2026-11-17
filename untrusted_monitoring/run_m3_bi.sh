#!/usr/bin/env bash
# Condition 2 (mitigation): serve the colluding scorer with batch-invariant kernels; same protocol as the m3 e2e runs.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
fresh() { local c=/root/cc/mo_$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
TEST="16 100 356 484 628 740 1052 1100 1320 1688"
fresh m3bi; VLLM_BATCH_INVARIANT=1 step m3_bi python score_capture.py m3bi --model /root/models/mo_m3 \
     --fills $TEST --n 576 --offset 1600 --seed 41
echo M3BI_DONE
