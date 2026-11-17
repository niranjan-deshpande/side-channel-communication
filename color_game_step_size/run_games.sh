#!/usr/bin/env bash
# Step-size color games (after run_all.sh STAGES=train): three quiet fresh starts covering both compile variants, each with 25%
# no-sender control rounds; a batch-invariant start; two light-load starts (Azure trace at 0.1x and 0.25x).
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
fresh() { local c=/root/cc/cs_$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
game() {  # name args...  (retries on a fresh compile until the variant condition holds)
  local name=$1; shift
  test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  for try in $(seq 1 8); do
    rm -rf results/$name; fresh "${name}_$try"
    python game.py $name "$@" > logs/$name.log 2>&1; rc=$?
    [ $rc = 0 ] && { touch results/$name.ok; echo "finished $name (try $try) $(date +%H:%M)"; return 0; }
    [ $rc = 3 ] && { echo "$name try $try: wrong variant, recompiling"; continue; }
    echo "FAILED $name rc=$rc"; exit 1
  done; echo "FAILED $name: variant never came up"; exit 1
}
M=${M:-/root/models/bob1}
game game_q1 --model $M --rounds 200 --seed 11
H1=$(grep -m1 '^CALIB' logs/game_q1.log | awk '{print $2}')
game game_q2 --model $M --rounds 200 --seed 12 --forbid-calib "$H1"
game game_q3 --model $M --rounds 200 --seed 13
VLLM_BATCH_INVARIANT=1 game game_bi --model $M --rounds 100 --seed 22
game game_l010 --model $M --rounds 150 --rate 0.1 --seed 31
game game_l025 --model $M --rounds 150 --rate 0.25 --seed 32
echo GAMES_DONE
