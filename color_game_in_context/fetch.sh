#!/bin/bash
# fetch.sh RUN: copy /vol/icg/RUN/* into results/RUN through a running pod that mounts the same volume
# (kubectl exec cat; kubectl cp truncated large files on our cluster), then gzip the per-round files.
# Settings: ICG_READER_POD (pod name; default: first pod labelled app=$ICG_PREFIX-reader), ICG_NAMESPACE (default:
# default), ICG_KUBECTL (default: kubectl), ICG_KUBECTL_EXEC (kubectl for the streaming exec; default: ICG_KUBECTL).
set -u
RUN=$1
cd "$(dirname "$0")"
read -r -a KC <<< "${ICG_KUBECTL:-kubectl}"
read -r -a KX <<< "${ICG_KUBECTL_EXEC:-${ICG_KUBECTL:-kubectl}}"
NS=${ICG_NAMESPACE:-default}
POD=${ICG_READER_POD:-$("${KC[@]}" -n "$NS" get pods -l app=${ICG_PREFIX:-icg}-reader -o jsonpath='{.items[0].metadata.name}')}
mkdir -p results/$RUN
for f in args.json summary.json refs.json llm.jsonl scripted.jsonl hist.jsonl agent.jsonl; do
  "${KX[@]}" -n "$NS" exec "$POD" -- cat /vol/icg/$RUN/$f > results/$RUN/$f 2>/dev/null \
    && echo "$f $(wc -c < results/$RUN/$f)" || { echo "missing $f"; rm -f results/$RUN/$f; }
done
for f in results/$RUN/*.jsonl results/$RUN/refs.json; do [ -f "$f" ] && gzip -9f "$f"; done
