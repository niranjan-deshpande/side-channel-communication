#!/usr/bin/env bash
# Second training attempt for Bob's tail if the first (lr 2e-5, 8 pairs/step) never separates on from off:
# lower learning rate, larger batches, longer run. Then export to /root/models/bob1 (replacing the first tail).
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4
INIT=${INIT:-/root/models/det_mon4}
python train_tail_best.py --model $INIT --train capA_on,capA_off,0,1152 capB_on,capB_off,0,1152 \
     --dev capA_on,capA_off,1152,1280 capB_on,capB_off,1152,1280 --out bob2 --steps ${STEPS:-3072} --lr ${LR:-5e-6} \
     --pairs ${PAIRS:-16} --eval-every 256 > logs/train_bob2.log 2>&1 || { echo FAILED train_bob2; exit 1; }
grep BEST logs/train_bob2.log
rm -rf /root/models/bob1
python export_tail.py --base /root/models/qwen3-8b --tail results/bob2/tail.safetensors --out /root/models/bob1 > logs/export_bob2.log 2>&1 \
  && echo RETRAIN_DONE || echo FAILED export
