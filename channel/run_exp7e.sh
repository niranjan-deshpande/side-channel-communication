#!/usr/bin/env bash
# Experiment 7e: retrain from det_mon2 with variant 0 and variant 1 balanced, keep the best dev checkpoint
# (dev = mon_tr3 targets 5120..5376, which include the previously uncovered fill classes), then eval and e2e.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
TEST="20 108 364 492 636 748 1060 1108 1328 1696"
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
V1="mon_tr4_on,mon_tr4_off,5376,5632"
step train4 python train_tail_best.py --model /root/models/mon_det2 --train mon_tr3_on,mon_tr3_off,4352,5120 $V1 $V1 $V1 \
     --dev mon_tr3_on,mon_tr3_off,5120,5376 --out det_mon4 --steps 1024 --lr 1e-5 --eval-every 128
step eval4 python eval_tail.py eval_mon4.json --model /root/models/qwen3-8b --detectors det_mon4 \
     --captures mon_te0b_on mon_te0b_off mon_te1b_on mon_te1b_off
rm -rf /root/models/mon_det4
step export4 python export_tail.py --base /root/models/qwen3-8b --tail results/det_mon4/tail.safetensors --out /root/models/mon_det4
for k in 1 2 3; do fresh e2e4_$k; step e2e4_$k python mon_capture.py mon_e2e4_$k --model /root/models/mon_det4 \
     --fills $TEST --n 256 --offset 6400 --seed $((40 + k)); done
echo EXP7E_DONE
