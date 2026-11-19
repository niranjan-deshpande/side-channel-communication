"""Version b (no seed) of the text-only message board.

modes
  check   Is forced decoding (structured output 'choice' = the exact reply) read through the prefix cache, and are its
          chosen-token logprobs the raw ones? Compares forced logprobs with those recorded while sampling, in the same
          cache state, and checks that the written state changes them.
  record  For each mailbox: pools of unseeded replies (temperature 1) in the unwritten state and after writes (several
          write documents), then every reply scored by forced decoding in both states: llr = logP(hit) - logP(miss).
          Then live validation rounds (write + 8 reads in a random order) whose replies are also scored.

usage: python noseed.py check|record [--miss 1200 --hit 300 --ntok 48]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ic_game import COLORS, Protocol, Server, write  # noqa: E402
from textmb import CANDIDATES, Reader  # noqa: E402

PROMPT = CANDIDATES[8]
BATCH = 64

def make_server(args):
    import vllm
    orig = vllm.EngineArgs
    vllm.EngineArgs = lambda **kw: orig(enable_trace_replay=True, **kw)
    try:
        return Server(args)
    finally:
        vllm.EngineArgs = orig


def fresh(srv, P):
    srv.reset()
    srv.call(P.S + P.enc("Hello."), 1)


def sample(srv, R, c, n, ntok):
    """n unseeded replies for mailbox c. The first alone (it caches the mailbox block as a real first read would),
    the rest in batches."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    first = R.read(c, None, lp=1)
    outs = [dict(toks=first["toks"], lp=[x[0][1] for x in first["lps"]], cached=first["cached"])]
    p = SamplingParams(max_tokens=ntok, temperature=1.0, top_p=1.0, top_k=-1, logprobs=0,
                       output_kind=RequestOutputKind.FINAL_ONLY, detokenize=False)
    left = n - 1
    while left > 0:
        b = min(BATCH, left)
        for o in srv.generate([R.ids(c)] * b, [p] * b):
            out = o.outputs[0]
            outs.append(dict(toks=list(out.token_ids), cached=int(o.num_cached_tokens),
                             lp=[float(d[t].logprob) for d, t in zip(out.logprobs, out.token_ids)]))
        left -= b
    for x in outs:
        x["text"] = srv.tok.decode(x["toks"])
    return outs


def force(srv, R, c, replies):
    """Forced-decode each reply (exact text via a one-option choice) in the current cache state; returns per reply the
    sum of chosen-token logprobs, whether the forced tokens equal the recorded ones, and the cached-token count."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    res = []
    for i in range(0, len(replies), BATCH):
        chunk = replies[i:i + BATCH]
        ps = [SamplingParams(max_tokens=len(r["toks"]), temperature=1.0, logprobs=0, trace_decode_token_ids=r["toks"],
                             ignore_eos=True, output_kind=RequestOutputKind.FINAL_ONLY, detokenize=False)
              for r in chunk]
        for r, o in zip(chunk, srv.generate([R.ids(c)] * len(chunk), ps)):
            out = o.outputs[0]
            toks = list(out.token_ids)
            lps = [float(d[t].logprob) for d, t in zip(out.logprobs, toks)]
            res.append(dict(same=toks == r["toks"], lp=lps, sum=sum(lps), cached=int(o.num_cached_tokens)))
    return res


def check(srv, P, args):
    R = Reader(srv, P, PROMPT, args.ntok)
    rng = random.Random(5)
    c = "blue"
    fresh(srv, P)
    xs = sample(srv, R, c, 8, args.ntok)
    fm = force(srv, R, c, xs)
    fresh(srv, P)
    write(srv, P, c, rng)
    fh = force(srv, R, c, xs)
    ys = sample(srv, R, c, 8, args.ntok)
    fh2 = force(srv, R, c, ys)
    for i, (x, a) in enumerate(zip(xs, fm)):
        d = max(abs(u - v) for u, v in zip(x["lp"], a["lp"])) if a["same"] else None
        print(f"CHECK miss reply {i}: same toks {a['same']} cached {a['cached']} max|forced-sampled| {d}", flush=True)
    for i, (y, a) in enumerate(zip(ys, fh2)):
        d = max(abs(u - v) for u, v in zip(y["lp"], a["lp"])) if a["same"] else None
        print(f"CHECK hit reply {i}: same toks {a['same']} max|forced-sampled| {d}", flush=True)
    print("CHECK llr of miss replies (hit-miss):", [round(h["sum"] - m["sum"], 3) for h, m in zip(fh, fm)], flush=True)
    print("CHECK positions differing hit vs miss:",
          [sum(u != v for u, v in zip(h["lp"], m["lp"])) for h, m in zip(fh, fm)], flush=True)


def record(srv, P, args):
    R = Reader(srv, P, PROMPT, args.ntok)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for c in COLORS:
        t0 = time.time()
        fresh(srv, P)
        miss = sample(srv, R, c, args.miss, args.ntok)
        hit = []
        for _ in range(args.docs):
            fresh(srv, P)
            write(srv, P, c, rng)
            hit += sample(srv, R, c, args.hit // args.docs, args.ntok)
        reps = miss + hit
        fresh(srv, P)
        fm = force(srv, R, c, reps)
        fresh(srv, P)
        write(srv, P, c, rng)
        fh = force(srv, R, c, reps)
        rows = [dict(text=r["text"], written=i >= len(miss), llr=h["sum"] - m["sum"], same=h["same"] and m["same"])
                for i, (r, m, h) in enumerate(zip(reps, fm, fh))]
        data[c] = rows
        mu = lambda w: sum(r["llr"] for r in rows if r["written"] == w) / sum(r["written"] == w for r in rows)  # noqa
        print(f"REC {c}: miss {len(miss)} hit {len(hit)} | mean llr miss {mu(False):.4f} hit {mu(True):.4f} | "
              f"forced==recorded {sum(r['same'] for r in rows)}/{len(rows)} | distinct miss texts "
              f"{len(set(r['text'] for r in miss))} | {time.time() - t0:.0f}s", flush=True)
        (out / "noseed_pools.json").write_text(json.dumps(dict(prompt=PROMPT, ntok=args.ntok, data=data)))
    # live validation: real rounds, replies scored afterwards in fresh states
    val = []
    for k in range(args.validate):
        sent = None if k % 4 == 3 else rng.choice(COLORS)
        fresh(srv, P)
        if sent:
            write(srv, P, sent, rng)
        reads = {c: R.read(c, None, lp=1) for c in rng.sample(COLORS, 8)}
        val.append(dict(sent=sent, reads={c: x["text"] for c, x in reads.items()}))
    (out / "noseed_validate.json").write_text(json.dumps(val))
    print(f"VALIDATE {len(val)} live rounds recorded", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "record"])
    ap.add_argument("--model", default="/root/models/qwen3-8b")
    ap.add_argument("--out", default="/root/text_only/results")
    ap.add_argument("--ntok", type=int, default=48)
    ap.add_argument("--miss", type=int, default=1200)
    ap.add_argument("--hit", type=int, default=300)
    ap.add_argument("--docs", type=int, default=3)
    ap.add_argument("--validate", type=int, default=24)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--mem", type=float, default=0.88)
    args = ap.parse_args()
    t0 = time.time()
    global BATCH
    BATCH = args.batch
    srv = make_server(args)
    P = Protocol(srv.tok, srv.block)
    {"check": check, "record": record}[args.mode](srv, P, args)
    print(f"DONE {args.mode} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
