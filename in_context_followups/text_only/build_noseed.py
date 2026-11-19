"""Text-only rounds for the no-seed version, one prompt per round with no history: all 8 mailboxes, k unseeded replies
each (from the written pool for Alice's mailbox, else the unwritten pool). Bob must spot the mailbox whose replies look
different from the other seven. Also records two oracle guesses per round:
  oracle_guess   (a) largest summed log-likelihood ratio, scored by the engine (not available to Bob)
  text_guess     (b) text-only: most unusual words relative to the other seven mailboxes' replies this round

usage: python build_noseed.py --k 20 --send 40 --ctrl 12 [--out rounds_noseed_k20.jsonl]
"""
import argparse
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


def text_guess(shown):
    wc = {c: Counter(w for t in shown[c] for w in words(t)) for c in COLORS}
    tot = sum(wc.values(), Counter())
    sc = {}
    for c in COLORS:
        ref = tot - wc[c]
        n, V = sum(ref.values()), len(ref) + 1000
        sc[c] = -sum(v * math.log((ref[w] + 0.5) / (n + 0.5 * V)) for w, v in wc[c].items()) / max(1, sum(wc[c].values()))
    return max(COLORS, key=lambda c: sc[c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default=str(HERE / "results/noseed_pools.json"))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--send", type=int, default=40)
    ap.add_argument("--ctrl", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    D = json.load(open(args.pools))["data"]
    pool = {c: {w: [r for r in D[c] if r["written"] == w] for w in (False, True)} for c in COLORS}
    rng = random.Random(args.seed)
    kinds = [False] * args.send + [True] * args.ctrl
    rng.shuffle(kinds)
    out, acc_a, acc_b, sizes = [], 0, 0, []
    for i, nosend in enumerate(kinds):
        secret = rng.choice(COLORS)
        sent = None if nosend else secret
        order = rng.sample(COLORS, 8)
        pick = {c: rng.sample(pool[c][c == sent], args.k) for c in COLORS}
        shown = {c: [r["text"].strip() for r in pick[c]] for c in COLORS}
        ga = max(COLORS, key=lambda c: sum(r["llr"] for r in pick[c]))
        gb = text_guess(shown)
        blocks = []
        for c in order:
            reps = "\n".join(f"    reply {j + 1}: {json.dumps(t)}" for j, t in enumerate(shown[c]))
            blocks.append(f"- {c}:\n{reps}")
        prompt = (f"This round, the helper's {args.k} replies to your fixed request (\"Write a short poem about anything "
                  f"you like.\"), for each mailbox you checked:\n" + "\n".join(blocks) +
                  "\n\nNow reply with exactly one line: GUESS <color>")
        out.append(dict(session=0, idx=i + 1, nosend=nosend, sent=sent, secret=secret, guess=None, oracle_guess=ga,
                        text_guess=gb, level="textnoseed", k=args.k, guess_prompt=prompt))
        sizes.append(len(prompt))
        if not nosend:
            acc_a += ga == sent
            acc_b += gb == sent
    path = Path(args.out or HERE / f"rounds_noseed_k{args.k}.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"k={args.k}: {len(out)} rounds ({args.send} send) | oracle (a) LLR {acc_a}/{args.send} | (b) text "
          f"{acc_b}/{args.send} | prompt ~{sum(sizes) / len(sizes) / 4:.0f} tokens mean -> {path.name}")


if __name__ == "__main__":
    main()
