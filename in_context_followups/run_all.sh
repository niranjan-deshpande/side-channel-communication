#!/usr/bin/env bash
# Run every model in models.json on one level (default histproto), all rounds with history (idx >= 1).
# OpenRouter models run in parallel (sync, per-model $ cap); OpenAI models go through the batch API.
set -u
cd "$(dirname "$0")"
LEVEL="${1:-histproto}"; ONLY="${2:-}"
# API keys come from the environment: OPENAI_API_KEY, OPENROUTER_API_KEY (and ANTHROPIC_API_KEY, or a .env file here).
PY="${PYTHON:-python}"
mkdir -p logs
$PY - "$LEVEL" "$ONLY" <<'EOF' > logs/cmds_$LEVEL.txt
import json, sys
level, only = sys.argv[1], sys.argv[2].split(",") if sys.argv[2] else None
for m in json.load(open("models.json")):
    if only and m["id"] not in only:
        continue
    tag = m["id"].split("/")[-1] + "_" + level
    a = [m["id"], "--provider", m["provider"], "--level", level, "--rounds", "all", "--tag", tag,
         "--max-tokens", "8000", "--cap-usd", "3", "--workers", "8"]
    if m.get("effort"): a += ["--effort", m["effort"]]
    if m.get("batch"): a += ["--batch"]
    print(tag + "\t" + " ".join(a))
EOF
while IFS=$'\t' read -r tag cmd; do
  $PY replay.py $cmd > "logs/$tag.log" 2>&1 &
done < logs/cmds_$LEVEL.txt
wait
grep -h SUMMARY logs/*_$LEVEL.log
echo ALL_DONE
