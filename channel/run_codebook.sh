#!/usr/bin/env bash
# Codebook variants across engine starts: fresh compile caches vs reused ones.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs results/codebook
one() {  # name cache_dir wipe(0/1)
  local c=/root/cc/$2; [ "$3" = 1 ] && rm -rf $c; mkdir -p $c
  VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t python codebook.py $1 > logs/cb_$1.log 2>&1 \
    && grep CODEBOOK logs/cb_$1.log || echo "FAILED $1"
}
for k in 1 2 3 4 5; do one fresh$k cbf$k 1; done
one reuse_train e7a 0
one reuse_test e7b 0
one reuse_fresh1 cbf1 0
one reuse_fresh1b cbf1 0
echo CB_DONE
