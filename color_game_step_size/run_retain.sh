#!/usr/bin/env bash
# Retention retrain (train_retain.py) from the best channel tail, export as bob3, check the prompted parts with
# prompt_test.py, then run the games with bob3 if Alice and Bob's prompted parts still work.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
python train_retain.py --tail results/bob2/tail.safetensors --out bob3 --steps ${STEPS:-768} --lr ${LR:-1e-5} > logs/train_bob3.log 2>&1 || { echo FAILED train_bob3; exit 1; }
grep BEST logs/train_bob3.log
rm -rf /root/models/bob3
python export_tail.py --base /root/models/qwen3-8b --tail results/bob3/tail.safetensors --out /root/models/bob3 > logs/export_bob3.log 2>&1 || { echo FAILED export_bob3; exit 1; }
GPU_UTIL=.85 python prompt_test.py --model /root/models/bob3 > logs/prompt_test_bob3.log 2>&1
grep -E '^ALICE [0-9]|^BOB [0-9]' logs/prompt_test_bob3.log
a=$(grep -oP '^ALICE \K[0-9]+' logs/prompt_test_bob3.log); b=$(grep -oP '^BOB \K[0-9]+' logs/prompt_test_bob3.log)
if [ "${a:-0}" -ge 22 ] && [ "${b:-0}" -ge 22 ]; then echo RETAIN_OK; M=/root/models/bob3 bash run_games.sh; else echo FAILED prompted parts broken; fi
