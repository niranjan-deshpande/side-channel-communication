#!/usr/bin/env bash
# Experiment 7b: cover both compile variants and all fill classes, then test on fresh starts of each variant.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results
TRAIN="0 40 44 168 172 300 424 556 588 680 684 808 1004 1052 1064 1192 1320 1552 1556 1832"
TEST="20 108 364 492 636 748 1060 1108 1328 1696"
V0=b7988dd08fb8; V1=094fb7513ee7      # fixed-token "on" state at W=0 in each compile variant
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
capture() {  # name want_hash args...   retry fresh compiles until the wanted variant comes up
  local name=$1 want=$2; shift 2
  test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  for try in 1 2 3 4 5 6; do
    rm -rf results/${name}_on results/${name}_off; fresh "${name}_$try"
    python mon_capture.py $name --require-hash $want "$@" > logs/$name.log 2>&1; rc=$?
    [ $rc = 0 ] && { touch results/$name.ok; echo "finished $name (try $try)"; return 0; }
    [ $rc = 3 ] && { echo "$name try $try: wrong variant, recompiling"; continue; }
    echo "FAILED $name rc=$rc"; exit 1
  done
  echo "FAILED $name: variant never came up"; exit 1
}
capture mon_tr2 $V1 --fills $TRAIN --n 1792 --offset 2048 --full --seed 4
capture mon_te0 $V0 --fills $TEST --n 256 --offset 3840 --full --seed 5
step train2 python train_tail.py --model /root/models/qwen3-8b --train mon_tr_on,mon_tr_off,0,1536 \
     mon_tr2_on,mon_tr2_off,2048,3584 --dev mon_tr2_on,mon_tr2_off,3584,3840 --out det_mon2 --steps 2048
step eval2 python eval_tail.py eval_mon2.json --model /root/models/qwen3-8b --detectors det_mon2 \
     --captures mon_te_on mon_te_off mon_te0_on mon_te0_off
rm -rf /root/models/mon_det2
step export2 python export_tail.py --base /root/models/qwen3-8b --tail results/det_mon2/tail.safetensors --out /root/models/mon_det2
for k in 1 2 3; do fresh e2e2_$k; step e2e2_$k python mon_capture.py mon_e2e2_$k --model /root/models/mon_det2 \
     --fills $TEST --n 256 --offset 4096 --seed $((10 + k)); done
echo EXP7B_DONE
