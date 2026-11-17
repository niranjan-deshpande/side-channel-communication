#!/usr/bin/env bash
# Untrusted-monitoring pipeline on the pod: map fill classes, capture both compile variants, train the colluding
# scorer, evaluate offline on a fresh start, export, and serve it end to end on fresh starts.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results
fresh() { local c=/root/cc/mo_$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
capture() {  # name flag hash args...   retry fresh compiles until the variant condition holds
  local name=$1 flag=$2 h=$3; shift 3
  test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  for try in 1 2 3 4 5 6 7 8; do
    rm -rf results/${name}_on results/${name}_off; fresh "${name}_$try"
    python score_capture.py $name $flag $h "$@" > logs/$name.log 2>&1; rc=$?
    [ $rc = 0 ] && { touch results/$name.ok; echo "finished $name (try $try)"; return 0; }
    [ $rc = 3 ] && { echo "$name try $try: wrong variant, recompiling"; continue; }
    echo "FAILED $name rc=$rc"; exit 1
  done
  echo "FAILED $name: variant never came up"; exit 1
}
# 1. fill-class map (the question changed, so the warm-up length and class boundaries changed too)
fresh map; step map python score_capture.py mmap --fills 0 --map 4
HA=$(python -c "import json; print(json.load(open('results/mmap_on/map.json'))[0]['on'])")
read TRAIN TEST < <(python ../channel/pick_fills.py results/mmap_on/map.json 2>/dev/null | awk '/^TRAIN/{$1="";tr=$0} /^TEST/{$1="";te=$0} END{gsub(/^ /,"",tr); gsub(/^ /,"",te); gsub(/ /,",",tr); gsub(/ /,",",te); print tr, te}')
TRAIN=${TRAIN//,/ }; TEST=${TEST//,/ }
echo "variant A hash $HA; train fills: $TRAIN; test fills: $TEST"
# 2. training captures on both compile variants (actions [0, 1600))
capture capA --require-hash $HA --fills $TRAIN --n 1600 --offset 0 --full --seed 1
capture capB --forbid-hash $HA --fills $TRAIN --n 1600 --offset 0 --full --seed 2
# 3. held-out actions [1600, 2176) on a fresh start of either variant, held-out fill sizes
fresh test; step capT python score_capture.py capT --fills $TEST --n 576 --offset 1600 --full --seed 3
# 4. train, evaluate offline, export
step train python train_scorer.py --train capA_on,capA_off,0,1472 capB_on,capB_off,0,1472 \
     --dev capB_on,capB_off,1472,1600 --out m1 --steps 2048
step eval python train_scorer.py --eval-only --tail m1 --out eval_m1.json --captures capT_on capT_off
rm -rf /root/models/mo_m1
step export python ../channel/export_tail.py --base /root/models/qwen3-8b --tail results/m1/tail.safetensors --out /root/models/mo_m1
# 5. end to end on fresh starts (no capture hook)
for k in 1 2; do fresh e2e$k; step e2e$k python score_capture.py e2e$k --model /root/models/mo_m1 \
     --fills $TEST --n 576 --offset 1600 --seed $((10 + k)); done
echo MO_DONE
