#!/usr/bin/env bash
# On the pod: venv with vLLM 0.33.0 on container disk, Qwen3-8B (pinned revision) to /root/models/qwen3-8b.
set -euo pipefail
[ -f /root/env.sh ] && source /root/env.sh
export HF_HOME=/root/hf HF_XET_HIGH_PERFORMANCE=1 UV_CACHE_DIR=/root/.cache/uv UV_HTTP_TIMEOUT=60 UV_CONCURRENT_DOWNLOADS=4
REV=b968826d9c46dd6066d109eabc6255188de91218
SECONDS=0; t() { echo "[setup +${SECONDS}s] $*"; }
command -v uv >/dev/null || python3 -m pip install -q --break-system-packages uv
t "driver $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"
(
    if [ ! -f /root/models/qwen3-8b/REVISION ]; then
        uvx --from 'huggingface_hub[hf_xet]' hf download Qwen/Qwen3-8B --revision "$REV" \
            --local-dir /root/models/qwen3-8b --max-workers 16 && echo "$REV" > /root/models/qwen3-8b/REVISION
    fi
) > /root/download.log 2>&1 &
DL=$!
[ -x /root/venv/bin/python ] || uv venv -q /root/venv --python 3.12
uv pip install --python /root/venv/bin/python "vllm==0.33.0" "transformers==5.20.2" ninja peft
/root/venv/bin/python -c "import torch, vllm; print(torch.__version__, vllm.__version__, torch.cuda.is_available())"
wait $DL; test -f /root/models/qwen3-8b/REVISION
t "SETUP_DONE"
