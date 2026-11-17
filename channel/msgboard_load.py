"""Does the message board survive the load that closes the step-size channel?

The step-size channel fails under load because every step fills whether or not the sender is there. A message-board
mark is written ahead of time, so the writer can pick a quiet moment and a small step. The collision rule says a
hit differs from a miss whenever the writer's step class differs from the reader's, so a reader whose step is full
under load should still tell a mark written in a small step from no mark, unless load evicts the mark.

Per load level:
  write   (quiet) reset the cache, fill S, then for each mailbox k with label 1 the writer sends S + B_k + text alone
  read    (Azure trace at this speed) one reader per mailbox, about once a second: S + B_k + F + text
  refer   (quiet) for every reader, reset, fill S, and replay the same reader ids with filler so its step has the same
          T it had under load, with no mark: the miss reference. Decode "written" iff the read state != the reference.
Records whether S and B_k were still cached when read (num_cached), since load can evict them.

usage: python msgboard_load.py NAME [--rates R...] [--readers N]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import EngineArgs, LLMEngine

from msgboard import AGENT_SYSTEM
from mon_traffic import load_trace
from physics import MODEL, TAIL, Reader, add, drain, h, random_ids

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--readers", type=int, default=60)
    ap.add_argument("--budget", type=int, default=2048)
    args = ap.parse_args()
    root = ROOT / "results" / args.name
    root.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=MODEL, dtype="bfloat16", max_model_len=8192, max_num_seqs=128, max_num_batched_tokens=args.budget,
        gpu_memory_utilization=.88, enable_prefix_caching=True, async_scheduling=False, seed=17,
        disable_log_stats=True, generation_config="vllm"))
    block = engine.vllm_config.cache_config.block_size
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    pad = 0
    while True:
        S = enc(f"<|im_start|>system\n{AGENT_SYSTEM + ' Thank you.' * pad}<|im_end|>\n<|im_start|>user\n")
        if len(S) % block == 0:
            break
        pad += 1
    F = enc(" Please continue with the task described below.")[:8]
    tail = enc(TAIL)
    bank = [enc(x) for x in (ROOT / "data" / "prompts.txt").read_text().splitlines() if x.strip() and not x.startswith("[")]
    words = enc(" ".join(["shared", "notes", "for", "the", "team", "on", "today's", "build"] * 6))
    reader = Reader(engine)
    rng = random.Random(3)
    trace = load_trace()
    warm_S = S + enc("Hello there.") + tail
    readpos = len(S) + 32 + 7

    def mailbox(k):
        return (enc(f"Mailbox {k:03d}: ") + words)[:32]

    def fill_S():
        assert engine.reset_prefix_cache()
        add(engine, f"S-{rng.random()}", warm_S, 1)
        drain(engine)

    rows, summary, kk = [], [], 0
    for rate in args.rates:
        n = args.readers
        ks = list(range(kk, kk + n)); kk += n
        labels = [1] * (n // 2) + [0] * (n - n // 2)
        rng.shuffle(labels)
        texts = [enc(f"[job {rng.randrange(10**8):08d}] ") + list(rng.choice(bank)) + tail for _ in range(n)]
        # write, quietly, each mark alone in a small step
        fill_S()
        for k, lab in zip(ks, labels):
            if lab:
                add(engine, f"w-{k}", S + mailbox(k) + enc(" noted.") + tail, 1)
                drain(engine)
        # read under load
        off = rng.randrange(len(trace) // 2)
        base = trace[off][0]
        times, t = [], 3.0
        for _ in range(n):
            times.append(t); t += rng.uniform(0.7, 1.3)
        bg = [((tt - base) / rate, max(16, min(c, 4000)), max(1, min(g, 512))) for tt, c, g in trace[off:]]
        bg = [x for x in bg if x[0] < t + 2]
        meta, live, b, j, t0 = {}, {}, 0, 0, time.monotonic()
        while True:
            now = time.monotonic() - t0
            while b < len(bg) and bg[b][0] <= now:
                ext = f"bg{rate}-{b}"
                live[ext] = add(engine, ext, random_ids(rng, bg[b][1]), bg[b][2]); b += 1
            while j < n and times[j] <= now:
                ids = S + mailbox(ks[j]) + F + texts[j]
                ext = f"r{rate}-{j}"
                rid = add(engine, ext, ids, 1)
                reader.watch(rid, ids, [readpos])
                meta[ext] = dict(rid=rid, j=j, ids=ids); live[ext] = rid; j += 1
            for o in engine.step():
                if o.finished:
                    live.pop(o.request_id, None)
                    if o.request_id in meta:
                        m = meta.pop(o.request_id)
                        rec = reader.done.pop(m["rid"])
                        c = next(c for c in rec["chunks"] if c["done"] <= readpos < c["done"] + c["n"])
                        rows.append(dict(rate=rate, j=m["j"], k=ks[m["j"]], label=labels[m["j"]], cached=o.num_cached_tokens,
                                         T=c["T"], split=len(rec["chunks"]) > 1, h=h(rec["state"][0]), ids=m["ids"]))
            if j == n and not meta:
                break
        try:
            engine.abort_request(list(live.values()))
        except Exception as e:  # noqa: BLE001
            print("abort failed", e, flush=True)
        drain(engine)
        # miss references, quietly, at the same T
        for r in [r for r in rows if r["rate"] == rate]:
            fill_S()
            ids = r["ids"]
            R = len(ids) - len(S)
            if r["T"] - R >= 1:
                add(engine, f"rf-{rate}-{r['j']}", random_ids(rng, r["T"] - R), 1)
            rid = add(engine, f"ref-{rate}-{r['j']}", ids, 1)
            reader.watch(rid, ids, [readpos])
            drain(engine)
            rec = reader.done.pop(rid)
            r["ref_T"], r["ref_h"] = rec["chunks"][0]["T"], h(rec["state"][0])
        rr = [r for r in rows if r["rate"] == rate]
        for r in rr:
            r["decoded"] = int(r["h"] != r["ref_h"])
            r["mark_cached"] = r["cached"] >= len(S) + 32
            r["S_cached"] = r["cached"] >= len(S)
        on = [r for r in rr if r["label"]]; offr = [r for r in rr if not r["label"]]
        s = dict(rate=rate, n=len(rr), bg=b, hit_decoded=float(np.mean([r["decoded"] for r in on])),
                 miss_decoded=float(np.mean([r["decoded"] for r in offr])),
                 ber=float((1 - np.mean([r["decoded"] for r in on]) + np.mean([r["decoded"] for r in offr])) / 2),
                 mark_survived=float(np.mean([r["mark_cached"] for r in on])),
                 S_cached=float(np.mean([r["S_cached"] for r in rr])),
                 reader_full=float(np.mean([r["T"] == args.budget for r in rr])),
                 ref_T_matched=float(np.mean([r["ref_T"] == r["T"] for r in rr])),
                 reads_split=float(np.mean([r["split"] for r in rr])))
        summary.append(s)
        print("RATE", json.dumps(s), flush=True)
    for r in rows:
        r.pop("ids")
    (root / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "summary.json").write_text(json.dumps(summary, indent=1))
    print("MBLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
