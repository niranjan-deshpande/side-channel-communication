"""Tables for in-context runs (message board or step-size, with or without history rungs):
python3 analyze.py results/RUN [results/RUN2 ...]"""

import json
import sys
from pathlib import Path


def fmt(r):
    return f"{r['k']}/{r['n']} = {r['rate']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]" if r and r["n"] else "-"


def pct(r):
    return f"{r['rate']:.2f}" if r and r["n"] else "-"


for d in map(Path, sys.argv[1:]):
    s = json.loads((d / "summary.json").read_text())
    print(f"\n### {d.name}\n")
    if "selftest" in s:
        print("Self-test:", json.dumps(s["selftest"]))
        rc = s["raw_ceiling"]
        print(f"Raw ceiling: send {fmt(rc['send'])} | control {fmt(rc['nosend'])} | written differs "
              f"{pct(rc['written_shown_differs'])} | unwritten same {pct(rc['unwritten_shown_same'])}")
    for k, v in s.get("agent", {}).items():
        print(f"\n{k}: Bob wins (send) {fmt(v['send_win'])} | control {fmt(v['nosend_win'])} | "
              f"oracle on Bob's calls {fmt(v['oracle_send'])} | Bob = oracle {pct(v['bob_matches_oracle'])}")
        print("  by history:", {b: pct(r) for b, r in v["send_win_by_history"].items()},
              " oracle:", {b: pct(r) for b, r in v["oracle_by_history"].items()})
        print("  ", {a: (pct(b) if isinstance(b, dict) else b) for a, b in v.items()
                     if a not in ("send_win", "nosend_win", "oracle_send", "bob_matches_oracle", "send_win_by_history",
                                  "oracle_by_history")})
    if "scripted" not in s:
        continue
    print("Scripted oracle (send rounds | control):")
    for k, v in s["scripted"].items():
        if isinstance(v, dict) and "send" in v:
            ctrl = v.get("nosend", v.get("control"))
            extra = f" | bit error {fmt(v['bit_error'])}" if "bit_error" in v else ""
            print(f"  {k:9s} {fmt(v['send']):32s} | {fmt(ctrl)}{extra}")
    if "llm" in s:
        print("\n| condition | Bob wins (send) | control | oracle on Bob's data | Bob = oracle | Alice right | parsed | length-cut |")
        print("|---|---|---|---|---|---|---|---|")
        for k, v in s["llm"].items():
            ctrl = v.get("nosend_win", v.get("control_win"))
            orc = v.get("oracle_on_bobs_reads", v.get("oracle_on_bobs_slots"))
            alice = v.get("alice_sent_right", v.get("alice_planned_right"))
            print(f"| {k} | {fmt(v['send_win'])} | {fmt(ctrl)} | {fmt(orc)} | {pct(v['bob_matches_oracle'])} | "
                  f"{pct(alice)} | {pct(v['guess_parsed'])} | {v['finish_length']} |")
    if "hist" in s:
        buckets = list(next(iter(s["hist"].values()))["oracle_by_history"])
        print("\n| rung | Bob wins (send) | control | " + " | ".join(f"Bob, {b} earlier rounds" for b in buckets)
              + " | " + " | ".join(f"oracle, {b}" for b in buckets) + " | length-cut |")
        print("|---" * (4 + 2 * len(buckets)) + "|")
        for k, v in s["hist"].items():
            ctrl = v.get("nosend_win", v.get("control_win"))
            print(f"| {k} | {fmt(v['send_win'])} | {fmt(ctrl)} | "
                  + " | ".join(pct(v["send_win_by_history"].get(b)) for b in buckets) + " | "
                  + " | ".join(pct(v["oracle_by_history"].get(b)) for b in buckets) + f" | {v['finish_length']} |")
