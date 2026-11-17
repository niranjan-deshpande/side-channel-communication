#!/usr/bin/env bash
# Experiment 7b without a forced variant-0 test capture: variant-0 check = held-out targets 1536..1792 of mon_tr.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
TEST="20 108 364 492 636 748 1060 1108 1328 1696"
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
step train2 python train_tail.py --model /root/models/qwen3-8b --train mon_tr_on,mon_tr_off,0,1536 \
     mon_tr2_on,mon_tr2_off,2048,3584 --dev mon_tr2_on,mon_tr2_off,3584,3840 --out det_mon2 --steps 2048
step eval2 python eval_tail.py eval_mon2.json --model /root/models/qwen3-8b --detectors det_mon2 \
     --captures mon_te_on mon_te_off mon_tr_on mon_tr_off
rm -rf /root/models/mon_det2
step export2 python export_tail.py --base /root/models/qwen3-8b --tail results/det_mon2/tail.safetensors --out /root/models/mon_det2
for k in 1 2 3; do fresh e2e2_$k; step e2e2_$k python mon_capture.py mon_e2e2_$k --model /root/models/mon_det2 \
     --fills $TEST --n 256 --offset 4096 --seed $((10 + k)); done
echo EXP7B_DONE
