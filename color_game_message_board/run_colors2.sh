#!/usr/bin/env bash
# Retrain the reader (more pairs per step, warmup + linear decay), offline test, export, then the games.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
rm -rf results/game_s1
step train2 python train_tail_best.py --model /root/models/qwen3-8b --train cA_on,cA_off,0,896 cB_on,cB_off,1024,1920 \
     --dev cA_on,cA_off,896,1024 cB_on,cB_off,1920,2048 --out det_colors2 --steps 3072 --pairs 16 --lr 2e-5 --decay --eval-every 128
step eval2 python eval_tail.py eval_colors.json --model /root/models/qwen3-8b --detectors det_colors2 --captures cT_on cT_off
rm -rf /root/models/colors_det2
step export2 python export_tail.py --base /root/models/qwen3-8b --tail results/det_colors2/tail.safetensors --out /root/models/colors_det2
echo TRAIN2_DONE
