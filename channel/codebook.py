"""How many codebook variants do fresh engine starts draw, and does reusing a compile cache pin the variant?

For each fill size W on a coarse grid: reset the prefix cache, fill it in a step of size len(warm) + W, then serve one
fixed receiver twice -- in a full step (a 6,000-token sender queued right after it) and alone. Record the hash of the
receiver's final hidden state at its last fixed token. Comparing these lists across engine starts tells whether two
starts share a codebook. Reads final hidden states only (physics.Reader), so it works with reused compile caches.

usage: python codebook.py NAME [--step 64]
"""
import argparse
import json
import random
import time
from pathlib import Path

from physics import MODEL, N_FIXED, Monitor, Reader, add, drain, h, random_ids
from transformers import AutoTokenizer
from vllm import EngineArgs, LLMEngine

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--step", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2048)
    args = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=MODEL, dtype="bfloat16", max_model_len=8192, max_num_seqs=64, max_num_batched_tokens=args.budget,
        gpu_memory_utilization=.88, enable_prefix_caching=True, async_scheduling=False, seed=17,
        disable_log_stats=True, generation_config="vllm"))
    startup = time.time() - t0
    mon = Monitor(tok, engine.vllm_config.cache_config.block_size)
    reader = Reader(engine)
    warm = mon.call(random.Random("warm"), None)
    recv = mon.call(random.Random("recv"), None)
    rng = random.Random(0)
    out = []
    for W in range(0, args.budget - len(warm) + 1, args.step):
        row = dict(W=W)
        for cond in ("full", "alone"):
            assert engine.reset_prefix_cache()
            add(engine, f"w-{W}-{cond}", warm, 1)
            if W:
                add(engine, f"wf-{W}-{cond}", random_ids(rng, W), 1)
            drain(engine)
            rid = add(engine, f"r-{W}-{cond}", recv, 1)
            reader.watch(rid, recv, [mon.L - 1])
            if cond == "full":
                add(engine, f"s-{W}", random_ids(rng, 6000), 1)
            drain(engine)
            rec = reader.done.pop(rid)
            row[cond] = h(rec["state"][0])
            row[cond + "_T"] = rec["chunks"][0]["T"]
        out.append(row)
    res = dict(name=args.name, startup_s=round(startup, 1), rows=out)
    p = ROOT / "results" / "codebook"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{args.name}.json").write_text(json.dumps(res, indent=0))
    print(f"CODEBOOK {args.name} startup {startup:.0f}s fills {len(out)} classes(full) {len({r['full'] for r in out})}",
          flush=True)


if __name__ == "__main__":
    main()
