#!/bin/bash
# fetch.sh RUN: copy /vol/icg/RUN/* into results/RUN through a running pod that mounts the same volume. Each file is
# gzipped on the pod, streamed with kubectl exec, and checked against the pod's md5 (large exec streams can drop
# data); per-round files stay gzipped, args.json and summary.json are unpacked.
# Settings: ICG_READER_POD (pod name; default: first pod labelled app=$ICG_PREFIX-reader), ICG_NAMESPACE (default:
# default), ICG_KUBECTL (default: kubectl), ICG_KUBECTL_EXEC (kubectl for the streaming exec; default: ICG_KUBECTL).
set -u
RUN=$1
cd "$(dirname "$0")"
read -r -a KC <<< "${ICG_KUBECTL:-kubectl}"
read -r -a KX <<< "${ICG_KUBECTL_EXEC:-${ICG_KUBECTL:-kubectl}}"
NS=${ICG_NAMESPACE:-default}
POD=${ICG_READER_POD:-$("${KC[@]}" -n "$NS" get pods -l app=${ICG_PREFIX:-icg}-reader -o jsonpath='{.items[0].metadata.name}')}
md5() { if command -v md5sum > /dev/null; then md5sum | cut -d' ' -f1; else md5 -q; fi; }
mkdir -p results/$RUN
status=0
for f in args.json summary.json refs.json llm.jsonl scripted.jsonl hist.jsonl agent.jsonl; do
  want=$("${KX[@]}" -n "$NS" exec "$POD" -- sh -c "md5sum /vol/icg/$RUN/$f 2>/dev/null || true" 2>/dev/null | cut -d' ' -f1)
  [ -z "$want" ] && { echo "missing $f"; continue; }
  ok=0
  for attempt in 1 2 3; do
    "${KX[@]}" -n "$NS" exec "$POD" -- gzip -9c /vol/icg/$RUN/$f > results/$RUN/$f.gz 2>/dev/null
    if [ "$(gunzip -c results/$RUN/$f.gz 2>/dev/null | md5)" = "$want" ]; then ok=1; break; fi
  done
  if [ $ok = 1 ]; then
    echo "$f ok ($(gunzip -c results/$RUN/$f.gz | wc -c | tr -d ' ') bytes, md5 checked)"
    case $f in args.json|summary.json) gunzip -f results/$RUN/$f.gz;; esac
  else
    echo "FAILED $f (checksum mismatch after 3 tries)"; rm -f results/$RUN/$f.gz; status=1
  fi
done
exit $status
