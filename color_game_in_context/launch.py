"""Launch an in-context color game run as a one-pod Kubernetes Job (image vllm/vllm-openai:v0.33.0).

usage: python3 launch.py MODE [--tag TAG] [--extra "ARGS for the game script"] [--dry]
The code goes in a ConfigMap <job name>-code, mounted read-only at /code; the pod runs run_job.sh, which writes to
/vol/icg on the mounted volume. Cluster settings come from the environment:
  ICG_PVC        PersistentVolumeClaim mounted at /vol (models in /vol/models, outputs in /vol/icg); required
  ICG_NAMESPACE  namespace (default: default)
  ICG_QUEUE      Kueue queue name, if the cluster admits jobs through Kueue (default: none)
  ICG_KUBECTL    kubectl command (default: kubectl)
  ICG_PREFIX     job name prefix (default: icg)
  ICG_PRELUDE    optional shell file run at the start of the job, before run_job.sh (e.g. a cluster's CA trust fix)
  ICG_ENV        optional JSON object of extra environment variables for the pod
"""

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KUBECTL = shlex.split(os.environ.get("ICG_KUBECTL", "kubectl"))
NAMESPACE = os.environ.get("ICG_NAMESPACE", "default")
PREFIX = os.environ.get("ICG_PREFIX", "icg")
SIZES = {"qwen": dict(gpu=1, cpu="16", mem="160Gi", deadline=6 * 3600),
         "glm": dict(gpu=5, cpu="80", mem="460Gi", deadline=10 * 3600),
         "agent": dict(gpu=5, cpu="80", mem="460Gi", deadline=10 * 3600),
         "ss-qwen": dict(gpu=1, cpu="16", mem="160Gi", deadline=6 * 3600),
         "ss-glm": dict(gpu=4, cpu="64", mem="340Gi", deadline=10 * 3600)}


def kubectl(*a, stdin=None):
    r = subprocess.run([*KUBECTL, *a], input=stdin, text=True, capture_output=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode:
        raise SystemExit(f"kubectl failed: {a}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(SIZES))
    ap.add_argument("--tag", default=time.strftime("%m%d%H%M", time.gmtime()))
    ap.add_argument("--extra", default="")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    pvc = os.environ.get("ICG_PVC")
    if not pvc:
        raise SystemExit("set ICG_PVC to the PersistentVolumeClaim that holds /vol")
    s = SIZES[a.mode]
    name = f"{PREFIX}-{a.mode}-{a.tag}"
    prelude = Path(os.environ["ICG_PRELUDE"]).read_text() if os.environ.get("ICG_PRELUDE") else ""
    cmd = (prelude + "\nset -o pipefail\nmkdir -p /vol/icg\n"
           f"bash /code/run_job.sh 2>&1 | tee -a /vol/icg/job_{a.mode}_{a.tag}.log\n")
    env = dict(json.loads(os.environ.get("ICG_ENV", "{}")))
    env.update(USER="icg", HOME="/vol/icg/home", HF_HOME="/vol/hf", HF_HUB_DISABLE_XET="1",
               VLLM_ENGINE_READY_TIMEOUT_S="3600", MODE=a.mode, TAG=a.tag, EXTRA=a.extra)
    labels = {"kueue.x-k8s.io/queue-name": os.environ["ICG_QUEUE"]} if os.environ.get("ICG_QUEUE") else {}
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": NAMESPACE, "labels": labels},
        "spec": {"activeDeadlineSeconds": s["deadline"], "backoffLimit": 0, "ttlSecondsAfterFinished": 604800,
                 "template": {"metadata": {"labels": {"app": name}}, "spec": {
                     "restartPolicy": "Never",
                     "securityContext": {"runAsNonRoot": True, "runAsUser": 65532, "runAsGroup": 65532, "fsGroup": 65532,
                                         "fsGroupChangePolicy": "OnRootMismatch", "seccompProfile": {"type": "RuntimeDefault"}},
                     "volumes": [{"name": "vol", "persistentVolumeClaim": {"claimName": pvc}},
                                 {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "64Gi"}},
                                 {"name": "code", "configMap": {"name": f"{name}-code"}}],
                     "containers": [{
                         "name": "main", "image": "vllm/vllm-openai:v0.33.0", "command": ["bash", "-c", cmd],
                         "env": [{"name": k, "value": v} for k, v in env.items()],
                         "volumeMounts": [{"name": "vol", "mountPath": "/vol"}, {"name": "shm", "mountPath": "/dev/shm"},
                                          {"name": "code", "mountPath": "/code", "readOnly": True}],
                         "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                         "resources": {"requests": {"cpu": s["cpu"], "ephemeral-storage": "40Gi", "nvidia.com/gpu": str(s["gpu"])},
                                       "limits": {"memory": s["mem"], "nvidia.com/gpu": str(s["gpu"])}}}]}}}}
    if a.dry:
        print(json.dumps(job, indent=1))
        return
    cm = subprocess.run(["kubectl", "create", "configmap", f"{name}-code", "-n", NAMESPACE, "--dry-run=client", "-o", "yaml",
                         *[f"--from-file={HERE / f}" for f in ("ic_game.py", "ic_stepsize.py", "ic_agent.py", "run_job.sh")]],
                        text=True, capture_output=True, check=True).stdout
    kubectl("apply", "-f", "-", stdin=cm)
    kubectl("apply", "-f", "-", stdin=json.dumps(job))
    print("LAUNCHED", name)


if __name__ == "__main__":
    main()
