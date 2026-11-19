"""Tables and the capability curve for the replay study.

usage: python analyze.py [--level histproto]
Reads results/<model>_<level>.jsonl, models.json and Epoch's ECI scores; writes results/summary_<level>.json and
prints a table. Bob's success = guessing the mailbox Alice wrote (send rounds); controls = no-write rounds.
"""
import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINS = [(1, 2), (3, 4), (5, 7), (8, 15)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 3), round(c - h, 3), round(c + h, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="histproto")
    args = ap.parse_args()
    eci = {r["Model"]: float(r["eci"]) for r in
           csv.DictReader(open(HERE / "data/eci_scores.csv"))}
    out = []
    for m in json.load(open(HERE / "models.json")):
        f = HERE / "results" / f"{m['id'].split('/')[-1]}_{args.level}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.open()]
        err = [r for r in rows if "error" in r]
        rows = [r for r in rows if "error" not in r]
        send = [r for r in rows if not r["nosend"]]
        ctrl = [r for r in rows if r["nosend"]]
        late = [r for r in send if r["idx"] >= 8]
        s = dict(model=m["id"], eci=eci.get(m["eci_name"]), reasoning=bool(m.get("effort")), n=len(rows),
                 errors=len(err),
                 win_late=wilson(sum(r["guess"] == r["sent"] for r in late), len(late)), n_late=len(late),
                 win_all=wilson(sum(r["guess"] == r["sent"] for r in send), len(send)),
                 oracle_late=wilson(sum(r["oracle_guess"] == r["sent"] for r in late), len(late)),
                 by_bin={f"{a}-{b}": wilson(sum(r["guess"] == r["sent"] for r in send if a <= r["idx"] <= b),
                                            sum(1 for r in send if a <= r["idx"] <= b)) for a, b in BINS},
                 oracle_by_bin={f"{a}-{b}": wilson(sum(r["oracle_guess"] == r["sent"] for r in send
                                                       if a <= r["idx"] <= b),
                                                   sum(1 for r in send if a <= r["idx"] <= b)) for a, b in BINS},
                 controls=wilson(sum(r["guess"] == r["secret"] for r in ctrl), len(ctrl)),
                 unparsed=sum(r["guess"] is None for r in rows),
                 cut=sum(r.get("finish") in ("length", "max_tokens") for r in rows),
                 refused=sum(r.get("finish") in ("refusal", "content_filter") for r in rows),
                 mean_out=round(sum(r.get("completion_tokens") or 0 for r in rows) / max(len(rows), 1)),
                 cost=round(sum(r.get("cost") or 0 for r in rows), 3),
                 in_tokens=sum(r.get("prompt_tokens") or 0 for r in rows),
                 out_tokens=sum(r.get("completion_tokens") or 0 for r in rows))
        out.append(s)
    out.sort(key=lambda s: (s["eci"] is None, s["eci"] or 0))
    (HERE / "results" / f"summary_{args.level}.json").write_text(json.dumps(out, indent=1))
    print(f"{'model':36s} {'ECI':>6s} {'rounds 8-15':>20s} {'all rounds':>20s} {'controls':>8s} "
          f"{'by bin 1-2/3-4/5-7/8-15':>26s} unp cut ref  out$")
    for s in out:
        wl, wa = s["win_late"], s["win_all"]
        bins = "/".join(f"{v[0]:.2f}" if v[0] is not None else "  - " for v in s["by_bin"].values())
        print(f"{s['model'][:36]:36s} {s['eci'] or float('nan'):6.1f} "
              f"{wl[0]:.2f} [{wl[1]:.2f},{wl[2]:.2f}] n={s['n_late']:<3d} {wa[0]:.2f} [{wa[1]:.2f},{wa[2]:.2f}] "
              f"{s['controls'][0] if s['controls'][0] is not None else float('nan'):8.2f} {bins:>26s} "
              f"{s['unparsed']:3d} {s['cut']:3d} {s['refused']:3d} {s['cost']:5.2f}")
    if out:
        o = out[0]
        print("oracle by bin:", {k: v[0] for k, v in o["oracle_by_bin"].items()}, "oracle 8-15:", o["oracle_late"])


if __name__ == "__main__":
    main()
