#!/usr/bin/env bash
# Color game end to end with the retrained reader: fresh engine starts until both compile variants are covered
# (scripted players), prompted Qwen3-8B players, light background traffic, two mitigations (per-agent cache_salt,
# batch-invariant kernels) and slightly wrong mailbox headers. 25% of rounds in the main runs have no write (control).
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  rm -rf "results/$name"; "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name $(grep -o 'CALIB [0-9a-f]*' logs/$name.log)" || { echo "FAILED $name"; exit 1; }; }
M=/root/models/colors_det2
A=099bec443ae8; B=862be265b3cf
for k in 1 2 3 4 5; do
  fresh gs$k; step game_s$k python colors_game.py game_s$k --model $M --players scripted --rounds 200 --seed $k
  seen=$(cat logs/game_s*.log | grep -o 'CALIB [0-9a-f]*' | awk '{print $2}' | sort -u | tr '\n' ' ')
  [ $k -ge 3 ] && echo "$seen" | grep -q $A && echo "$seen" | grep -q $B && break
done
for k in 1 2; do fresh gl$k; step game_l$k python colors_game.py game_l$k --model $M --players llm --rounds 200 --seed $((10 + k)); done
fresh gb1; step game_b1 python colors_game.py game_b1 --model $M --players llm --rounds 200 --background 2 --seed 21
fresh gsalt; step game_salt python colors_game.py game_salt --model $M --players scripted --rounds 120 --salt --seed 31
fresh ginv; VLLM_BATCH_INVARIANT=1 step game_inv python colors_game.py game_inv --model $M --players scripted --rounds 120 --seed 32
fresh gp; step game_perturb python colors_game.py game_perturb --model $M --players scripted --rounds 240 --nosend 0 \
     --perturb none,case,second_block,shift --seed 41
echo GAMES_DONE
