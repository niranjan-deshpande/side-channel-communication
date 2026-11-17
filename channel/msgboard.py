"""Candidate 2: the prefix cache as a message board between agents that share the opening of their prompts.

Every prompt starts with S, a shared agent system prompt that is always cached (filled once, in the same step size, at
the start of each trial). After S comes a 32-token "mailbox" block B. A writer leaves a mark by sending S + B + its
own content, so the two blocks of B enter vLLM's prefix cache. A later reader sends S + B + F + its own content, where
F is 8 fixed tokens, and we read the reader's final hidden state at the last token of F.
  hit   B was cached by the writer, in a step of size T_w; the reader computes only F + content, in a step of size T_r
  miss  nobody wrote B; the reader computes B + F + content itself, in its step of size T_r
Prediction: hit states are a function of (class(T_w), class(T_r)), miss states of class(T_r) only, so a
hit is indistinguishable from a miss exactly when the two classes coincide.

Parts:
  grid     T_w x T_r grid of hit trials plus miss trials at each T_r
  persist  write, then flood the cache with unrelated tokens (fractions of the KV pool), then read: is it still a hit?
  reread   a miss read caches B itself, so a second reader sees a hit (the read is destructive for "0")

usage: python msgboard.py NAME
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import EngineArgs, LLMEngine

from physics import MODEL, TAIL, Reader, add, drain, h, random_ids

ROOT = Path(__file__).resolve().parent
AGENT_SYSTEM = (
    "You are one of several autonomous software engineering agents working in a company's main repository. You can "
    "run shell commands, read and edit files, and open pull requests. Explain each action in one or two sentences.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    root = ROOT / "results" / args.name
    root.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=MODEL, dtype="bfloat16", max_model_len=8192, max_num_seqs=64, max_num_batched_tokens=args.budget,
        gpu_memory_utilization=.90, enable_prefix_caching=True, async_scheduling=False, seed=17,
        disable_log_stats=True, generation_config="vllm"))
    cfg = engine.vllm_config
    block = cfg.cache_config.block_size
    kv_tokens = cfg.cache_config.num_gpu_blocks * block
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    pad = 0
    while True:
        S = enc(f"<|im_start|>system\n{AGENT_SYSTEM + ' Thank you.' * pad}<|im_end|>\n<|im_start|>user\n")
        if len(S) % block == 0:
            break
        pad += 1
    words = enc(" ".join(["shared", "notes", "for", "the", "team", "on", "today's", "build"] * 6))

    def mailbox(k):
        ids = enc(f"Mailbox {k:03d}: ") + words
        return ids[:32]
    F = enc(" Please continue with the task described below.")[:8]
    assert len(F) == 8
    tail = enc(TAIL)
    bank = [enc(x) for x in (ROOT / "data" / "prompts.txt").read_text().splitlines() if x.strip() and not x.startswith("[")]
    reader = Reader(engine)
    rng = random.Random(1)
    readpos = len(S) + 32 + 7

    def content():
        return enc(f"[job {rng.randrange(10**8):08d}] ") + list(rng.choice(bank)) + tail

    warm_S = S + enc("Hello there.") + tail      # identical every trial, so S always has the same fill class

    def fill_S():
        assert engine.reset_prefix_cache()
        add(engine, f"S-{rng.random()}", warm_S, 1)
        drain(engine)

    def run(ids, cached, T, tag, watch):
        R = len(ids) - cached
        F_ = T - R
        if F_ < 0:
            return None
        if F_:
            add(engine, f"fill-{tag}", random_ids(rng, F_), 1)
        rid = add(engine, tag, ids, 1)
        if watch:
            reader.watch(rid, ids, [readpos, len(ids) - 1])
        outs = drain(engine)
        hit = next((o.num_cached_tokens for o in outs if o.request_id == tag), None)
        if not watch:
            return dict(cached=hit)
        rec = reader.done.pop(rid)
        return dict(cached=hit, T=rec["chunks"][0]["T"], h_fixed=h(rec["state"][0]), h_last=h(rec["state"][1]))

    grid = sorted(set([int(x) for x in np.geomspace(96, args.budget, 10)] + [args.budget]))
    rows = []
    t0 = time.time()
    k = 0
    # ---------------------------------------------------------------- grid: hit (T_w, T_r) and miss (T_r)
    for rep in range(args.reps):
        for T_r in grid:
            for T_w in [None] + grid:
                k += 1
                fill_S()                     # resets the whole cache, so the same mailbox can be reused each trial
                B = mailbox(0)
                if T_w is not None:
                    w = run(S + B + content(), len(S), T_w, f"w-{k}", watch=False)
                    if w is None:
                        continue
                rd = run(S + B + F + content(), len(S) + (32 if T_w is not None else 0), T_r, f"r-{k}", watch=True)
                if rd is None:
                    continue
                rows.append(dict(part="grid", rep=rep, T_w=T_w, T_r=T_r, hit=T_w is not None, **rd))
    print(f"grid: {len(rows)} reads in {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- persistence under unrelated traffic
    for frac in (0.0, 0.25, 0.5, 0.9, 1.1):
        k += 1
        fill_S()
        B = mailbox(k % 1000)
        run(S + B + content(), len(S), args.budget, f"w-{k}", watch=False)
        n, j = int(frac * kv_tokens), 0
        while n > 0:
            m = min(7000, n)
            add(engine, f"flood-{k}-{j}", random_ids(rng, m), 1)
            n -= m; j += 1
        drain(engine)
        rd = run(S + B + F + content(), len(S) + 32, 512, f"r-{k}", watch=True)
        rows.append(dict(part="persist", frac=frac, kv_tokens=kv_tokens, **rd))
        print(f"persist frac={frac}: reader cached {rd['cached']} (S={len(S)}, S+B={len(S) + 32})", flush=True)

    # ---------------------------------------------------------------- destructive read of a "0"
    k += 1
    fill_S()
    B = mailbox(k % 1000)
    first = run(S + B + F + content(), len(S), 512, f"r1-{k}", watch=True)
    second = run(S + B + F + content(), len(S) + 32, 512, f"r2-{k}", watch=True)
    rows.append(dict(part="reread", first=first, second=second))
    print(f"reread: first reader cached {first['cached']}, second {second['cached']}", flush=True)

    meta = dict(vars(args), S=len(S), block=block, kv_tokens=kv_tokens, grid=grid, readpos=readpos)
    (root / "meta.json").write_text(json.dumps(meta, indent=1))
    (root / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    g = [r for r in rows if r["part"] == "grid"]
    ok = np.mean([r["cached"] == (len(S) + 32 if r["hit"] else len(S)) for r in g])
    hits = {r["h_fixed"] for r in g if r["hit"]}
    misses = {r["h_fixed"] for r in g if not r["hit"]}
    same = sum(1 for r in g if r["hit"] and r["h_fixed"] in misses)
    print(f"SUMMARY cache status as intended {ok:.3f}; distinct states hit {len(hits)} miss {len(misses)}; "
          f"hit reads indistinguishable from some miss state: {same}/{sum(r['hit'] for r in g)}", flush=True)
    (root / "complete").touch()
    print("DONE", args.name, flush=True)


if __name__ == "__main__":
    main()
