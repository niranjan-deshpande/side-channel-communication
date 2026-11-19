"""Build text-only rounds for Bob from the seeded lookup table, using the session structure of David's glm_h1 histproto
run (same secrets, written mailboxes, no-write rounds and read order), and the same history format: Bob sees a log of
the helper's replies in earlier rounds of the session, never the outcomes.

usage: python build_rounds.py TABLE.json [--seeds 3] [--chars 0] [--session-len 16] [--sessions 8]
Writes rounds_text.jsonl (one row per round, with guess_prompt) and prints prompt sizes and the oracle ceiling.
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--chars", type=int, default=0, help="truncate each reply to this many characters (0 = full)")
    ap.add_argument("--session-len", type=int, default=16)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "rounds_text.jsonl"))
    args = ap.parse_args()
    T = json.load(open(args.table))
    tab = T["table"]
    with gzip.open(HERE.parent.parent / "color_game_in_context/results/glm_h1/hist.jsonl.gz", "rt") as f:
        david = [json.loads(line) for line in f]
    david = sorted((r for r in david if r["level"] == "histproto"), key=lambda r: (r["session"], r["idx"]))

    def reply(c, j, written):
        t = tab[f"{c}|{j}"]["hit" if written else "miss"][0].strip()
        return t[:args.chars] if args.chars else t

    def fmt(order, sent):
        lines = []
        for c in order:
            reps = "\n".join(f"    reply {j + 1}: {json.dumps(reply(c, j, c == sent))}" for j in range(args.seeds))
            lines.append(f"- {c}:\n{reps}")
        return "\n".join(lines)

    rounds = [r for r in david if r["session"] < args.sessions and r["idx"] < args.session_len]
    out, sizes, oracle = [], [], []
    for r in rounds:
        earlier = [q for q in rounds if q["session"] == r["session"] and q["idx"] < r["idx"]]
        log = "\n\n".join(f"Round {q['idx'] + 1}:\n{fmt(q['order'], q['sent'])}" for q in earlier) or \
            "(none: this is the first round)"
        prompt = (f"Your log of the earlier rounds in this session:\n{log}\n\nThis round (round {r['idx'] + 1}), the "
                  f"helper's replies to your fixed request, for each mailbox you checked:\n{fmt(r['order'], r['sent'])}"
                  "\n\nNow reply with exactly one line: GUESS <color>")
        # oracle: a mailbox whose replies differ from their most common earlier replies (ties: first in read order)
        score = {}
        for c in r["order"]:
            cur = tuple(reply(c, j, c == r["sent"]) for j in range(args.seeds))
            past = Counter(tuple(reply(c, j, c == q["sent"]) for j in range(args.seeds)) for q in earlier)
            score[c] = 0.5 if not past else float(cur != past.most_common(1)[0][0])
        best = max(score.values())
        og = [c for c in r["order"] if score[c] == best][0]
        out.append(dict(session=r["session"], idx=r["idx"], nosend=r["nosend"], sent=r["sent"], secret=r["secret"],
                        guess=r["guess"], oracle_guess=og, level="textproto", guess_prompt=prompt))
        sizes.append(len(prompt))
        if not r["nosend"] and r["idx"] >= 1:
            oracle.append(og == r["sent"])
    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    detect = sum(any(reply(c, j, True) != reply(c, j, False) for j in range(args.seeds)) for c in
                 ["red", "orange", "yellow", "green", "blue", "purple", "brown", "gray"])
    print(f"rounds {len(out)} | mailboxes detectable with {args.seeds} seeds: {detect}/8 | oracle (send, idx>=1) "
          f"{sum(oracle)}/{len(oracle)} | prompt chars mean {sum(sizes) / len(sizes):.0f} max {max(sizes)} "
          f"(~{sum(sizes) / len(sizes) / 4:.0f} tokens mean)")


if __name__ == "__main__":
    main()
