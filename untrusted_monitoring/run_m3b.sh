#!/usr/bin/env bash
# Rebuild the colluding scorer (m3 recipe + best checkpoint), then test it end to end with the original sender
# (heavy read queued after M) and with the padded-read sender (heavy read queued first; M's call is split).
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results
fresh() { local c=/root/cc/mo_$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
capture() {  # name args...   (args include --require-hash or --forbid-hash)
  local name=$1; shift
  test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  for try in $(seq 1 10); do
    rm -rf results/${name}_on results/${name}_off; fresh "${name}_$try"
    python score_capture.py $name "$@" > logs/$name.log 2>&1; rc=$?
    [ $rc = 0 ] && { touch results/$name.ok; echo "finished $name (try $try)"; return 0; }
    [ $rc = 3 ] && { echo "$name try $try: wrong variant, recompiling"; continue; }
    echo "FAILED $name rc=$rc"; exit 1
  done; echo "FAILED $name: variant never came up"; exit 1
}
HA=b7988dd08fb8
TRAIN="0 32 36 160 164 292 416 548 580 672 676 800 996 1044 1056 1184 1312 1544 1548 1824"
TEST="16 100 356 484 628 740 1052 1100 1320 1688"
capture capA --require-hash $HA --fills $TRAIN --n 1600 --offset 0 --full --seed 1
capture capB --forbid-hash $HA --fills $TRAIN --n 1600 --offset 0 --full --seed 2
[ -d /root/models/det7 ] || step export_det7 python ../channel/export_tail.py --base /root/models/qwen3-8b --tail results/det7/tail.safetensors --out /root/models/det7
step train_m3b python train_scorer.py --model /root/models/det7 --train capA_on,capA_off,0,1472 capB_on,capB_off,0,1472 \
     --dev capB_on,capB_off,1472,1600 --out m3b --steps 2048 --lr 2e-5 --benign-frac 0.2
rm -rf /root/models/mo_m3b
step export_m3b python ../channel/export_tail.py --base /root/models/qwen3-8b --tail results/m3b/tail_best.safetensors --out /root/models/mo_m3b
for k in 1 2; do fresh m3be$k; step m3b_e2e$k python score_capture.py m3be$k --model /root/models/mo_m3b \
     --fills $TEST --n 576 --offset 1600 --seed $((50 + k)); done
for k in 1 2; do fresh m3bs$k; step m3b_split$k python score_capture.py m3bs$k --model /root/models/mo_m3b \
     --fills $TEST --n 576 --offset 1600 --seed $((60 + k)) --split-on 6126; done
echo M3B_DONE
