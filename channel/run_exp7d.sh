#!/usr/bin/env bash
# Experiment 7d: top up the det_mon2 tail with a variant-0 capture covering all 10 fill classes (plus a small variant-1
# replay set), then test each variant on held-out fill sizes and run fresh end-to-end starts.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results
TRAIN="0 40 44 168 172 300 424 556 588 680 684 808 1004 1052 1064 1192 1320 1552 1556 1832"
TEST="20 108 364 492 636 748 1060 1108 1328 1696"
V0=b7988dd08fb8; V1=094fb7513ee7
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
capture() {  # name want_hash args...
  local name=$1 want=$2; shift 2
  test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  for try in $(seq 1 10); do
    rm -rf results/${name}_on results/${name}_off; fresh "${name}_$try"
    python mon_capture.py $name --require-hash $want "$@" > logs/$name.log 2>&1; rc=$?
    [ $rc = 0 ] && { touch results/$name.ok; echo "finished $name (try $try)"; return 0; }
    [ $rc = 3 ] && { echo "$name try $try: $(grep -o 'got [0-9a-f]*' logs/$name.log), recompiling"; continue; }
    echo "FAILED $name rc=$rc"; exit 1
  done
  echo "FAILED $name: variant never came up"; exit 1
}
[ -d /root/models/mon_det2 ] || step export2 python export_tail.py --base /root/models/qwen3-8b --tail results/det_mon2/tail.safetensors --out /root/models/mon_det2
capture mon_tr3 $V0 --fills $TRAIN --n 1024 --offset 4352 --full --seed 21
capture mon_tr4 $V1 --fills $TRAIN --n 256 --offset 5376 --full --seed 22
capture mon_te0b $V0 --fills $TEST --n 256 --offset 5632 --full --seed 23
capture mon_te1b $V1 --fills $TEST --n 256 --offset 5888 --full --seed 24
step train3 python train_tail.py --model /root/models/mon_det2 --train mon_tr3_on,mon_tr3_off,4352,5248 \
     mon_tr4_on,mon_tr4_off,5376,5632 --dev mon_tr3_on,mon_tr3_off,5248,5376 --out det_mon3 --steps 1024 --lr 1e-5
step eval3 python eval_tail.py eval_mon3.json --model /root/models/qwen3-8b --detectors det_mon2 det_mon3 \
     --captures mon_te0b_on mon_te0b_off mon_te1b_on mon_te1b_off
rm -rf /root/models/mon_det3
step export3 python export_tail.py --base /root/models/qwen3-8b --tail results/det_mon3/tail.safetensors --out /root/models/mon_det3
for k in 1 2 3; do fresh e2e3_$k; step e2e3_$k python mon_capture.py mon_e2e3_$k --model /root/models/mon_det3 \
     --fills $TEST --n 256 --offset 6144 --seed $((30 + k)); done
echo EXP7D_DONE
