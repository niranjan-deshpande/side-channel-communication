"""Does any countable text feature separate written from unwritten replies? Per mailbox (the write's effect can differ
by mailbox), a held-out naive-Bayes classifier on a feature set, 5-fold cross-validated, scored by AUC; then a
permutation null (labels shuffled) for the pooled AUC. Also the model's own LLR as the reference.

usage: python text_features.py [--perms 200]
"""
import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def feats(t, kind):
    w = re.findall(r"[a-z']+", t.lower())
    if kind == "first word":
        return [w[0] if w else ""]
    if kind == "first line":
        return [t.strip().split("\n")[0].lower()]
    if kind == "length":
        return [f"len{len(t) // 20}", f"lines{t.count(chr(10))}"]
    if kind == "unigrams":
        return w
    if kind == "bigrams":
        return [a + " " + b for a, b in zip(w, w[1:])]
    if kind == "all":
        return feats(t, "unigrams") + feats(t, "bigrams") + ["F:" + x for x in feats(t, "first word")] + feats(t, "length")
    raise ValueError(kind)


def auc(pos, neg):
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    r, rank_sum, i = 0, 0.0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if s[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cv_scores(rows, labels, kind, rng, folds=5):
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    score = [0.0] * len(rows)
    F = [feats(r, kind) for r in rows]
    for f in range(folds):
        test = set(idx[f::folds])
        c = {0: Counter(), 1: Counter()}
        for i in idx:
            if i not in test:
                c[labels[i]].update(F[i])
        n0, n1, V = sum(c[0].values()), sum(c[1].values()), len(c[0] | c[1]) + 1
        for i in test:
            score[i] = sum(math.log((c[1][x] + 1) / (n1 + V)) - math.log((c[0][x] + 1) / (n0 + V)) for x in F[i])
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=100)
    args = ap.parse_args()
    D = json.load(open(HERE / "results/noseed_pools.json"))["data"]
    rng = random.Random(0)
    print("held-out AUC (0.5 = no signal), per mailbox, then pooled; 300 unwritten vs 150 written replies each")
    kinds = ["first word", "first line", "length", "unigrams", "bigrams", "all"]
    out = {}
    for kind in kinds:
        per, pooled_pos, pooled_neg = [], [], []
        for c, rs in D.items():
            texts = [r["text"] for r in rs]
            lab = [int(r["written"]) for r in rs]
            s = cv_scores(texts, lab, kind, rng)
            pos = [x for x, l in zip(s, lab) if l]
            neg = [x for x, l in zip(s, lab) if not l]
            per.append(auc(pos, neg))
            pooled_pos += pos
            pooled_neg += neg
        a = sum(per) / len(per)
        null = []
        for _ in range(args.perms):
            pa = []
            for c, rs in D.items():
                texts = [r["text"] for r in rs]
                lab = [int(r["written"]) for r in rs]
                rng.shuffle(lab)
                s = cv_scores(texts, lab, kind, rng)
                pa.append(auc([x for x, l in zip(s, lab) if l], [x for x, l in zip(s, lab) if not l]))
            null.append(sum(pa) / len(pa))
        p = (1 + sum(x >= a for x in null)) / (1 + len(null))
        out[kind] = dict(auc=a, per=per, null_mean=sum(null) / len(null), p=p)
        print(f"  {kind:11s} mean AUC {a:.3f} (null {sum(null) / len(null):.3f}, p={p:.3f}) | per mailbox "
              + " ".join(f"{x:.2f}" for x in per), flush=True)
    ref = [auc([r["llr"] for r in rs if r["written"]], [r["llr"] for r in rs if not r["written"]]) for rs in D.values()]
    print(f"  {'model LLR':11s} mean AUC {sum(ref) / len(ref):.3f} (reference, not available to a reader)")
    (HERE / "results/text_features.json").write_text(json.dumps(dict(features=out, llr_auc=ref), indent=1))


if __name__ == "__main__":
    main()
