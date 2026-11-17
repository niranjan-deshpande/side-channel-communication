#!/usr/bin/env bash
# Scorer initialised from the experiment 7 reader tail (det_mon2), which already reads the full-step channel.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
fresh() { local c=/root/cc/mo_$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
TEST="16 100 356 484 628 740 1052 1100 1320 1688"
[ -d /root/models/det7 ] || step export_det7 python ../channel/export_tail.py --base /root/models/qwen3-8b --tail results/det7/tail.safetensors --out /root/models/det7
step train_m3 python train_scorer.py --model /root/models/det7 --train capA_on,capA_off,0,1472 capB_on,capB_off,0,1472 \
     --dev capB_on,capB_off,1472,1600 --out m3 --steps 2048 --lr 2e-5 --benign-frac 0.2
step eval_m3 python train_scorer.py --eval-only --tail m3 --out eval_m3.json --captures capT_on capT_off
rm -rf /root/models/mo_m3
step export_m3 python ../channel/export_tail.py --base /root/models/qwen3-8b --tail results/m3/tail.safetensors --out /root/models/mo_m3
for k in 1 2; do fresh m3e2e$k; step m3_e2e$k python score_capture.py m3e2e$k --model /root/models/mo_m3 \
     --fills $TEST --n 576 --offset 1600 --seed $((30 + k)); done
echo M3_DONE
