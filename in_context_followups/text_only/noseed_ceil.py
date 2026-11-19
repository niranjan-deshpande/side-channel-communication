"""Ceilings for the no-seed text-only message board, from results/noseed_pools.json.

(a) model-based: in each round, every mailbox shows k unseeded replies (drawn from its written pool if it is the
    written mailbox, else from its unwritten pool); the oracle guesses the mailbox with the largest summed
    log-likelihood ratio log P(reply | written) - log P(reply | unwritten), scored by the engine itself.
(b) text-only: an oracle Bob could run. For each mailbox, a smoothed word-unigram model is fit to everything else Bob
    has seen (this round's other mailboxes plus all earlier rounds); the guess is the mailbox whose k replies are least
    likely under it (most unusual words).
Rounds follow David's glm_h1 histproto sessions (8 x 16, same written mailboxes and no-write rounds). Accuracy is
over send rounds with idx >= 1, averaged over R resamples of the pools.

usage: python noseed_ceil.py [--ks 1 3 6 10 20 50] [--reps 50]
"""
import argparse
import gzip
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "brown", "gray"]


def words(t):
    return re.findall(r"[a-z']+", t.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(HERE / "results/noseed_pools.json"))
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 3, 6, 10, 20, 50])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--sessions", type=int, default=8)
    args = ap.parse_args()
    D = json.load(open(args.pools))["data"]
    pool = {c: {w: [r for r in D[c] if r["written"] == w] for w in (False, True)} for c in D}
    with gzip.open(HERE.parent.parent / "color_game_in_context/results/glm_h1/hist.jsonl.gz", "rt") as f:
        david = sorted((r for r in map(json.loads, f) if r["level"] == "histproto"),
                       key=lambda r: (r["session"], r["idx"]))
    david = [r for r in david if r["session"] < args.sessions]
    print("pools:", {c: (len(pool[c][False]), len(pool[c][True])) for c in pool})
    for c in pool:
        m = [r["llr"] for r in pool[c][False]]
        h = [r["llr"] for r in pool[c][True]]
        sd = (sum((x - sum(m) / len(m)) ** 2 for x in m) / len(m)) ** .5
        print(f"  {c:7s} mean llr unwritten {sum(m) / len(m):+.3f} written {sum(h) / len(h):+.3f} sd {sd:.3f} "
              f"d' {(sum(h) / len(h) - sum(m) / len(m)) / sd:.2f}")
    out = {}
    for k in args.ks:
        rng = random.Random(k)
        acc_a = acc_b = n = 0
        ctrl_a = 0
        nctrl = 0
        for _ in range(args.reps):
            hist = Counter()
            last_session = None
            for r in david:
                if r["session"] != last_session:
                    hist, last_session = Counter(), r["session"]
                shown = {c: rng.sample(pool[c][c == r["sent"]], k) for c in COLORS}
                sa = {c: sum(x["llr"] for x in shown[c]) for c in COLORS}
                ga = max(COLORS, key=lambda c: sa[c])
                wc = {c: Counter(w for x in shown[c] for w in words(x["text"])) for c in COLORS}
                allnow = sum(wc.values(), Counter())
                sb = {}
                for c in COLORS:
                    ref = hist + allnow - wc[c]
                    tot, V = sum(ref.values()), len(ref) + 1000
                    sb[c] = -sum(v * math.log((ref[w] + 0.5) / (tot + 0.5 * V)) for w, v in wc[c].items()) / \
                        max(1, sum(wc[c].values()))
                gb = max(COLORS, key=lambda c: sb[c])
                if r["idx"] >= 1:
                    if r["nosend"]:
                        nctrl += 1
                        ctrl_a += ga == r["secret"]
                    else:
                        n += 1
                        acc_a += ga == r["sent"]
                        acc_b += gb == r["sent"]
                hist += allnow
        out[k] = dict(a=acc_a / n, b=acc_b / n, n_send=n // args.reps)
        print(f"k={k:3d}: ceiling (a) model LLR {acc_a / n:.3f} | (b) text unigram {acc_b / n:.3f} | chance 0.125 | "
              f"send rounds {n // args.reps} x {args.reps} resamples", flush=True)
    (HERE / "results/noseed_ceilings.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
