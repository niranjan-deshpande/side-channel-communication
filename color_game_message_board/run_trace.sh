#!/usr/bin/env bash
# Color game over the message board under the Azure 2023 trace, matching the step-size runs (0.1x, 0.25x) plus 0.5x
# and 1.0x. Scripted players, 150 rounds per start (25% no-write control), one fresh engine start per speed.
set -u
cd "$(dirname "$0")"
export PATH=/root/venv/bin:$PATH OMP_NUM_THREADS=4 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
fresh() { local c=/root/cc/$1; rm -rf $c; mkdir -p $c; export VLLM_CACHE_ROOT=$c/v TORCHINDUCTOR_CACHE_DIR=$c/i TRITON_CACHE_DIR=$c/t; }
step() { local name=$1; shift; test -f "results/$name.ok" && { echo "skip $name"; return 0; }
  rm -rf "results/$name"; "$@" > "logs/$name.log" 2>&1 && touch "results/$name.ok" && echo "finished $name $(grep -o 'CALIB [0-9a-f]*' logs/$name.log)" || { echo "FAILED $name"; exit 1; }; }
mkdir -p logs results
M=/root/models/colors_det2
step export2 python export_tail.py --base /root/models/qwen3-8b --tail results/det_colors2/tail.safetensors --out $M
for spec in "010 0.1 51" "025 0.25 52" "050 0.5 53" "100 1.0 54"; do
  set -- $spec
  fresh gt$1; step game_t$1 python colors_game.py game_t$1 --model $M --players scripted --rounds 150 --trace $2 --seed $3
done
echo TRACE_DONE
for f in results/game_t*/summary.json; do python - "$f" <<'PY'
import json,sys; s=json.load(open(sys.argv[1]))
print(s["name"], s["calib"], "send", s["send_win_rate"], s["send_win_ci"], s["send_rounds"], "nosend", s["nosend_win_rate"], s["nosend_rounds"],
      "yesW", round(s["read_yes_written"],3), "yesU", round(s["read_yes_unwritten"],4), "cache_ok", round(s["cache_as_designed"],3),
      "writeFull", s["write_full_step"], "writeT", s["write_T_median"], "readT", s["read_T_median"], "alone", round(s["reads_alone"],3),
      "collide", round(s["reads_collision_zone"],3), "bg", s["background_requests"], "sec", s["seconds"])
PY
done
date
