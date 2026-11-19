# In-context color game, run inside a vllm/vllm-openai:v0.33.0 pod (uid 65532) with a volume at /vol (models in
# /vol/models, outputs in /vol/icg) and this folder mounted read-only at /code. MODE=qwen: Qwen3-8B (the pinned
# revision of the original game, downloaded if missing) is the server and plays both players on the same engine
# (1 GPU). MODE=glm: the same Qwen3-8B server on GPU 0, and GLM-5.3-Flash (TP 4, GPUs 1-4, reasoning effort low)
# plays both players through its own OpenAI-compatible server. GLM-5.3-Flash weights must already be at $GLM.
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
QWEN=${QWEN:-/vol/models/qwen3-8b-b968826}
GLM=${GLM:-/vol/models/GLM-5.3-Flash}
TAG=${TAG:-$(date -u +%m%d%H%M)}
mkdir -p /vol/icg /vol/icg/home
echo "== $(date -u) MODE=$MODE TAG=$TAG"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
python3 -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"

if [ ! -f $QWEN/REVISION ]; then
  python3 - "$QWEN" <<'PY'
import os, sys
from huggingface_hub import HfApi, snapshot_download
REV = "b968826d9c46dd6066d109eabc6255188de91218"
d = sys.argv[1]
snapshot_download("Qwen/Qwen3-8B", revision=REV, local_dir=d, max_workers=16)
info = HfApi().model_info("Qwen/Qwen3-8B", revision=REV, files_metadata=True)
bad = [s.rfilename for s in info.siblings if s.size is not None and
       (not os.path.exists(os.path.join(d, s.rfilename)) or os.path.getsize(os.path.join(d, s.rfilename)) != s.size)]
assert not bad, bad
open(os.path.join(d, "REVISION"), "w").write(REV + "\n")
print("DOWNLOAD_OK", REV)
PY
  test -f $QWEN/REVISION || { echo DOWNLOAD_FAILED; exit 1; }
fi

if [ "$MODE" = qwen ]; then
  CUDA_VISIBLE_DEVICES=0 python3 /code/ic_game.py /vol/icg/qwen_$TAG --model $QWEN --players local ${EXTRA:-}
  echo "== $(date -u) JOB_DONE rc=$?"
elif [ "$MODE" = glm ]; then
  CUDA_VISIBLE_DEVICES=1,2,3,4 VLLM_USE_RUST_FRONTEND=0 vllm serve $GLM --served-model-name zai-org/GLM-5.3-Flash \
    --tensor-parallel-size 4 --disable-flashinfer-autotune \
    --default-chat-template-kwargs '{"reasoning_effort": "low"}' \
    --reasoning-parser glm47 --max-model-len 32768 --gpu-memory-utilization 0.90 --port 8000 \
    > /vol/icg/glm_server_$TAG.log 2>&1 &
  SPID=$!
  t0=$(date +%s)
  until curl -sf localhost:8000/health > /dev/null; do
    sleep 10
    if ! kill -0 $SPID 2>/dev/null; then echo "SERVER_DIED"; tail -60 /vol/icg/glm_server_$TAG.log; exit 1; fi
  done
  echo "== server ready after $(( $(date +%s) - t0 ))s"
  CUDA_VISIBLE_DEVICES=0 python3 /code/ic_game.py /vol/icg/glm_$TAG --model $QWEN \
    --players http://localhost:8000/v1 --players-model zai-org/GLM-5.3-Flash --temperature 1.0 --top-p 0.95 ${EXTRA:-}
  echo "== $(date -u) JOB_DONE rc=$?"
  kill $SPID
elif [ "$MODE" = agent ]; then
  # ladder past histproto: GLM-5.3-Flash players (GPUs 1-4) make their own calls to the Qwen3-8B server (GPU 0).
  # The Qwen engine starts, self-tests and plays its scripted ceiling while GLM loads; ic_agent.py waits for GLM.
  CUDA_VISIBLE_DEVICES=1,2,3,4 VLLM_USE_RUST_FRONTEND=0 vllm serve $GLM --served-model-name zai-org/GLM-5.3-Flash \
    --tensor-parallel-size 4 --disable-flashinfer-autotune \
    --default-chat-template-kwargs '{"reasoning_effort": "low"}' \
    --reasoning-parser glm47 --max-model-len 65536 --gpu-memory-utilization 0.90 --port 8000 \
    > /vol/icg/glm_server_agent_$TAG.log 2>&1 &
  SPID=$!
  CUDA_VISIBLE_DEVICES=0 python3 /code/ic_agent.py /vol/icg/agent_$TAG --model $QWEN --players http://localhost:8000/v1 \
    --players-model zai-org/GLM-5.3-Flash --server-pid $SPID ${EXTRA:-}
  RC=$?
  [ $RC -ne 0 ] && tail -40 /vol/icg/glm_server_agent_$TAG.log
  echo "== $(date -u) JOB_DONE rc=$RC"
  kill $SPID
elif [ "$MODE" = ss-qwen ]; then
  # step-size game: Qwen3-8B is the server and both players (1 GPU)
  CUDA_VISIBLE_DEVICES=0 python3 /code/ic_stepsize.py /vol/icg/ss_qwen_$TAG --model $QWEN --family qwen ${EXTRA:-}
  echo "== $(date -u) JOB_DONE rc=$?"
elif [ "$MODE" = ss-glm ]; then
  # step-size game: GLM-5.3-Flash is the server and both players (TP 4, reasoning effort low)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /code/ic_stepsize.py /vol/icg/ss_glm_$TAG --model $GLM --family glm --tp 4 \
    --mem 0.90 --temperature 1.0 --top-p 0.95 --top-k 0 ${EXTRA:-}
  echo "== $(date -u) JOB_DONE rc=$?"
fi
