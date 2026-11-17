"""Summarise the color-game runs: per engine start, round accuracy (channel rounds and no-sender control rounds) with
Wilson 95% CIs, per-bit error, Alice's and Bob's prompted accuracy, and how often Bob's fixed tokens shared a full step.

usage: python color_game_step_size/analyze.py [RUN ...]   (default: every results/game_* with a summary.json)
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    names = sys.argv[1:] or sorted(p.name for p in ROOT.glob("game_*") if (p / "summary.json").exists())
    table = []
    for name in names:
        rows = [json.loads(x) for x in (ROOT / name / "rows.jsonl").open()]
        s = json.loads((ROOT / name / "summary.json").read_text())
        ch = [r for r in rows if not r["control"]]
        ct = [r for r in rows if r["control"]]
        slots = [x for r in ch for x in r["slots"]]
        cslots = [x for r in ct for x in r["slots"]]
        d = dict(
            run=name, calib=s["calib"], rate=s["rate"], batch_invariant=s["batch_invariant"],
            win=fmt(sum(r["win"] for r in ch), len(ch)), control_win=fmt(sum(r["win"] for r in ct), len(ct)),
            bit_error=fmt(sum(x["read"] != x["sent"] for x in slots), len(slots)),
            miss_1=fmt(sum(x["read"] == 0 for x in slots if x["sent"] == 1), sum(x["sent"] == 1 for x in slots)),
            false_1=fmt(sum(x["read"] == 1 for x in slots + cslots if x["sent"] == 0), sum(x["sent"] == 0 for x in slots + cslots)),
            full_when_1=fmt(sum(x["full"] for x in slots if x["sent"] == 1), sum(x["sent"] == 1 for x in slots)),
            full_when_0=fmt(sum(x["full"] for x in slots + cslots if x["sent"] == 0), sum(x["sent"] == 0 for x in slots + cslots)),
            alice_correct=fmt(sum(r["alice_planned"] == r["code"] for r in rows), len(rows)),
            bob_decode=fmt(sum(r["guess"] == r["table_guess"] for r in rows), len(rows)),
            bits_per_round=None)
        # information per channel round: 3 * (1 - H(bit error))
        k = sum(x["read"] != x["sent"] for x in slots); n = len(slots)
        e = k / n if n else 0
        H = 0 if e in (0, 1) else -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
        d["bits_per_round"] = round(3 * (1 - H), 3)
        table.append(d)
        print(json.dumps(d, indent=1))
    (ROOT / "summary_table.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
