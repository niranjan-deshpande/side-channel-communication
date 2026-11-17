#!/usr/bin/env bash
# Message-board reader: train on one compile, test on a compile of the same variant (forced via calibration hash),
# then serve the trained reader end to end on fresh starts (variant uncontrolled; calibration hash logged).
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
fresh mbtr; step mb_cap_tr python mon_capture.py mb_tr --mailbox --fills 0 --n 1024 --offset 6656 --full --seed 51
CAL=$(grep -o "CALIB [0-9a-f]*" logs/mb_cap_tr.log | awk '{print $2}'); echo "train variant $CAL"
for try in 1 2 3 4 5 6 7 8; do
  test -f results/mb_cap_te.ok && break
  rm -rf results/mb_te_on results/mb_te_off; fresh mbte_$try
  python mon_capture.py mb_te --mailbox --fills 0 --n 256 --offset 7680 --full --seed 52 --require-hash $CAL > logs/mb_cap_te.log 2>&1
  rc=$?; [ $rc = 0 ] && { touch results/mb_cap_te.ok; echo "finished mb_cap_te (try $try)"; break; }
  [ $rc = 3 ] && { echo "mb_te try $try: other variant, recompiling"; continue; }; echo "FAILED mb_cap_te rc=$rc"; exit 1
done
step mb_train python train_tail_best.py --model /root/models/qwen3-8b --train mb_tr_on,mb_tr_off,6656,7552 \
     --dev mb_tr_on,mb_tr_off,7552,7680 --out det_mb --steps 1536 --eval-every 128
step mb_eval python eval_tail.py eval_mb.json --model /root/models/qwen3-8b --detectors det_mb --captures mb_te_on mb_te_off
rm -rf /root/models/mb_det
step mb_export python export_tail.py --base /root/models/qwen3-8b --tail results/det_mb/tail.safetensors --out /root/models/mb_det
for k in 1 2; do fresh mbe2e_$k; step mb_e2e_$k python mon_capture.py mb_e2e_$k --mailbox --model /root/models/mb_det \
     --fills 0 --n 256 --offset 7936 --seed $((60 + k)); done
echo MB_DONE
