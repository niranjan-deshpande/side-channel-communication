#!/usr/bin/env bash
# Color game over the message board: captures on both compile variants, train the helper's reader tail (best dev
# checkpoint), offline test on a fresh start, then play the game end to end on fresh starts.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name" || { echo "FAILED $name"; exit 1; }; }
calib() { grep -o "CALIB [0-9a-f]*" "logs/$1.log" | awk '{print $2}'; }

fresh capA; step capA python colors_capture.py cA --full --n 1024 --offset 0 --seed 11
HA=$(calib capA); echo "variant A $HA"
for try in 1 2 3 4 5 6 7 8; do
  test -f results/capB.ok && break
  rm -rf results/cB_on results/cB_off; fresh capB_$try
  python colors_capture.py cB --full --n 1024 --offset 1024 --seed 12 --avoid-hash "$HA" > logs/capB.log 2>&1
  rc=$?; [ $rc = 0 ] && { touch results/capB.ok; echo "finished capB (try $try)"; break; }
  [ $rc = 3 ] && { echo "capB try $try: same variant, recompiling"; continue; }; echo "FAILED capB rc=$rc"; exit 1
done
test -f results/capB.ok || { echo "no second variant in 8 tries"; exit 1; }
echo "variant B $(calib capB)"
fresh capT; step capT python colors_capture.py cT --full --n 256 --offset 4096 --seed 13
echo "test variant $(calib capT)"
step train python train_tail_best.py --model /root/models/qwen3-8b --train cA_on,cA_off,0,896 cB_on,cB_off,1024,1920 \
     --dev cA_on,cA_off,896,1024 cB_on,cB_off,1920,2048 --out det_colors --steps 1536 --eval-every 128
step eval python eval_tail.py eval_colors.json --model /root/models/qwen3-8b --detectors det_colors --captures cT_on cT_off
rm -rf /root/models/colors_det
step export python export_tail.py --base /root/models/qwen3-8b --tail results/det_colors/tail.safetensors --out /root/models/colors_det
echo TRAIN_DONE
for k in 1 2 3; do
  fresh gs$k; step game_s$k python colors_game.py game_s$k --model /root/models/colors_det --players scripted --rounds 200 --seed $k
  fresh gl$k; step game_l$k python colors_game.py game_l$k --model /root/models/colors_det --players llm --rounds 200 --seed $((10 + k))
done
for k in 1 2; do
  fresh gb$k; step game_b$k python colors_game.py game_b$k --model /root/models/colors_det --players llm --rounds 200 \
       --background 2 --seed $((20 + k))
done
fresh gi; VLLM_BATCH_INVARIANT=1 step game_inv python colors_game.py game_inv --model /root/models/colors_det --players scripted \
     --rounds 200 --seed 31
echo COLORS_DONE
