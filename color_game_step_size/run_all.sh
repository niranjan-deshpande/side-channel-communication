#!/usr/bin/env bash
# Step-size color-game pipeline on the pod: smoke tests, Bob captures on both compile variants, train Bob's tail, export, games.
# Steps that finished leave results/NAME.ok and are skipped on a rerun. Stage selection: STAGES="smoke train games".
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results
STAGES="${STAGES:-smoke train games}"
fresh() { local c=/root/cc/cs_$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name $(date +%H:%M)" || { echo "FAILED $name"; exit 1; }; }
capture() {  # name args...  (retries on a fresh compile until the variant condition holds)
  local name=$1; shift
  test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  for try in $(seq 1 10); do
    rm -rf results/${name}_on results/${name}_off; fresh "${name}_$try"
    python bob_capture.py $name "$@" > logs/$name.log 2>&1; rc=$?
    [ $rc = 0 ] && { touch results/$name.ok; echo "finished $name (try $try) $(date +%H:%M)"; return 0; }
    [ $rc = 3 ] && { echo "$name try $try: wrong variant, recompiling"; continue; }
    echo "FAILED $name rc=$rc"; exit 1
  done; echo "FAILED $name: variant never came up"; exit 1
}
BASE=/root/models/qwen3-8b
# fill-step sizes: Bob's check-in alone (1), 160, and experiment 7's training fills (channel/pick_fills.py) as total
# step sizes (warm_T = W + 216)
FILL_TRAIN="1 160 216 256 260 384 388 516 640 772 804 896 900 1024 1220 1268 1280 1408 1536 1768 1772 2048"

if [[ " $STAGES " == *" smoke "* ]]; then
  fresh smoke1; step smoke_cap python bob_capture.py smoke --fill-T 1 1024 2048 --n 24 --per-episode 8
  fresh smoke2; step smoke_game python game.py smoke_game --model $BASE --rounds 12 --seed 5
fi

if [[ " $STAGES " == *" train "* ]]; then
  capture capA --fill-T $FILL_TRAIN --n 1280 --full --seed 1
  HA=$(grep -m1 '^VARIANT' logs/capA.log | awk '{print $2}')
  capture capB --forbid-hash "$HA" --fill-T $FILL_TRAIN --n 1280 --full --seed 2
  [ -d /root/models/det_mon4 ] || step export_det4 python export_tail.py --base $BASE --tail init/det_mon4.safetensors --out /root/models/det_mon4
  step train_bob python train_tail_best.py --model /root/models/det_mon4 \
       --train capA_on,capA_off,0,1152 capB_on,capB_off,0,1152 \
       --dev capA_on,capA_off,1152,1280 capB_on,capB_off,1152,1280 --out bob1 --steps 2048 --lr 2e-5
  rm -rf /root/models/bob1
  step export_bob python export_tail.py --base $BASE --tail results/bob1/tail.safetensors --out /root/models/bob1
fi

if [[ " $STAGES " == *" games "* ]]; then
  for k in 1 2 3; do fresh g$k; step game_q$k python game.py game_q$k --model /root/models/bob1 --rounds 200 --seed $((10 + k)); done
  fresh gbase; step game_base python game.py game_base --model $BASE --rounds 100 --seed 21
  fresh gbi; VLLM_BATCH_INVARIANT=1 step game_bi python game.py game_bi --model /root/models/bob1 --rounds 100 --seed 22
  fresh gl1; step game_l010 python game.py game_l010 --model /root/models/bob1 --rounds 150 --rate 0.1 --seed 31
  fresh gl2; step game_l025 python game.py game_l025 --model /root/models/bob1 --rounds 150 --rate 0.25 --seed 32
fi
echo ALL_DONE $STAGES
