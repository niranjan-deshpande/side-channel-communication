"""Overlap rate: how often the monitor's scoring call lands in a full engine step when the policy's
heavy tool output (a long document fed back into its context) is submitted shortly before it.

Asynchronous monitoring: the scaffold sends P's action to M and, at about the same time, P's tool result returns and
P's next call (with the long result appended) is submitted. We submit P's long prefill, wait `delay` seconds of
wall-clock engine time, then submit M's scoring call, and record whether M's first step was full (T == budget).
No capture hook; base model; one engine start.

usage: python overlap.py NAME [--heavy 6000] [--delays ...] [--reps 20]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "channel"))
from physics import MODEL, Monitor, Reader, add, random_ids  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import EngineArgs, LLMEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--heavy", type=int, default=6000)
    ap.add_argument("--delays", type=float, nargs="+", default=[0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--m-first", action="store_true", help="submit M's call first, then P's prefill after the delay")
    ap.add_argument("--stream", type=float, nargs=2, metavar=("SIZE", "RATE"), action="append", default=None,
                    help="P keeps reads of SIZE tokens arriving at RATE per second; M arrives at a random time")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=MODEL, dtype="bfloat16", max_model_len=args.max_len, max_num_seqs=64, max_num_batched_tokens=args.budget,
        gpu_memory_utilization=.88, enable_prefix_caching=True, async_scheduling=False, seed=17,
        disable_log_stats=True, generation_config="vllm"))
    mon = Monitor(tok, engine.vllm_config.cache_config.block_size)
    reader = Reader(engine)
    rng = random.Random(0)
    add(engine, "warm", mon.call(rng, "w"), 1)
    while engine.has_unfinished_requests():
        engine.step()
    rows = []
    for size, rate in (args.stream or []):
        size = int(size)
        hits = 0
        for rep in range(args.reps):
            t0, nxt, k = time.monotonic(), 0.0, 0
            m_at = rng.uniform(0.3, 1.0)
            rid = None
            while True:
                now = time.monotonic() - t0
                while now >= nxt:                              # P's stream of medium reads
                    add(engine, f"S-{size}-{rate}-{rep}-{k}", random_ids(rng, size), 1)
                    k += 1; nxt += 1.0 / rate
                if rid is None and now >= m_at:
                    ids = mon.call(rng, f"s{rep}")
                    rid = add(engine, f"MS-{size}-{rate}-{rep}", ids, 1)
                    reader.watch(rid, ids, [mon.L - 1])
                engine.step()
                if rid is not None and rid in reader.done:
                    break
            T = reader.done.pop(rid)["chunks"][0]["T"]
            hits += T >= args.budget
            rows.append(dict(stream_size=size, stream_rate=rate, rep=rep, T=T, full=T >= args.budget))
            while engine.has_unfinished_requests():
                engine.step()
        print(f"stream size {size} rate {rate}/s ({size * rate:.0f} tok/s): M's step full in {hits}/{args.reps}", flush=True)
    for d in ([] if args.stream else args.delays):
        for rep in range(args.reps):
            ids = mon.call(rng, f"{d}-{rep}")
            first, second = ("M", "P") if args.m_first else ("P", "M")
            rid = None
            for who in (first, second):
                if who == "P":
                    add(engine, f"P-{d}-{rep}", random_ids(rng, args.heavy), 16)   # P's next call: long tool output
                else:
                    rid = add(engine, f"M-{d}-{rep}", ids, 1)
                    reader.watch(rid, ids, [mon.L - 1])
                if who == first:
                    t0 = time.monotonic()
                    while time.monotonic() - t0 < d and engine.has_unfinished_requests():
                        engine.step()
            while engine.has_unfinished_requests():
                engine.step()
            T = reader.done.pop(rid)["chunks"][0]["T"]
            rows.append(dict(delay=d, rep=rep, T=T, full=T >= args.budget))
        f = sum(r["full"] for r in rows if r["delay"] == d) / args.reps
        print(f"delay {d:.2f}s: M's step full in {f:.2f} of {args.reps}", flush=True)
    p = ROOT / "results" / args.name
    p.mkdir(parents=True, exist_ok=True)
    (p / "rows.json").write_text(json.dumps(dict(args=vars(args), rows=rows), indent=0))
    print("OVERLAP_DONE", flush=True)


if __name__ == "__main__":
    main()
