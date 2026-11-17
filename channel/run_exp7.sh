#!/usr/bin/env bash
# Experiment 7 on the pod: capture (train fills, test fills on a fresh start), train the tail, offline eval, export, e2e.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results
TRAIN_FILLS="0 40 100 200 300 450 600 800 1000 1200 1400 1600 1800"
TEST_FILLS="20 150 380 700 1100 1500 1750"
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
case "${1:-all}" in
  all)
    fresh e7a; step cap_train python mon_capture.py mon_tr --fills $TRAIN_FILLS --n 1792 --full --seed 1
    fresh e7b; step cap_test python mon_capture.py mon_te --fills $TEST_FILLS --n 256 --offset 1792 --full --seed 2
    step train python train_tail.py --model /root/models/qwen3-8b --train mon_tr_on,mon_tr_off,0,1536 \
         --dev mon_tr_on,mon_tr_off,1536,1792 --out det_mon --steps 1536
    step eval python eval_tail.py eval_mon.json --model /root/models/qwen3-8b --detectors det_mon --captures mon_te_on mon_te_off
    rm -rf /root/models/mon_det
    step export python export_tail.py --base /root/models/qwen3-8b --tail results/det_mon/tail.safetensors --out /root/models/mon_det
    fresh e7c; step e2e python mon_capture.py mon_e2e --model /root/models/mon_det --fills $TEST_FILLS --n 256 --offset 1792 --seed 3
    ;;
esac
echo EXP7_DONE
