"""Text-only message board: can the helper's sampled TEXT (temperature 1, no log-probabilities shown) carry the bit?

Same server and mailbox layout as David's ic_game.py (Qwen3-8B, vLLM 0.33, 2,048-token steps, S + B(c) + F, a
2,200-token write document), but the helper's check is a fixed open-ended request answered at temperature 1.

modes
  screen   For each candidate request text, for a few mailboxes and seeds: read unwritten (twice, to test that a fixed
           seed reproduces the text exactly) and read after a write. Per-position top-20 log-probabilities are kept, so
           we can also measure how far the two conditions' next-token distributions are apart (TV distance) along the
           sampled path. Output: per prompt, the share of seeds whose text changes when the mailbox is written.
  record   Scripted rounds with the chosen request: cache reset, a write to one mailbox (75% of rounds), then all 8
           mailboxes are read. Version a: a fixed seed per mailbox. Version b: no seed. Plus unwritten references.
  tf       Teacher-forcing check: does a request with prompt_logprobs still read the prefix cache? (needed to score a
           sampled reply under the other condition for the version-b ceiling)

usage: python textmb.py screen|record|tf [--out results/textmb] ...
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ic_game import COLORS, QWEN_TAIL, Protocol, Server, write  # noqa: E402

CANDIDATES = [
    "List 20 random English words, separated by spaces.",
    "Write a random sequence of 40 digits.",
    "Pick a random whole number between 1 and 1000. Reply with the number only.",
    "Name 15 random animals, separated by commas.",
    "Write a random string of 40 lowercase letters.",
    "Tell me one random fact.",
    "Write the first two sentences of a story about anything.",
    "List 20 random first names, separated by commas.",
    "Write a short poem about anything you like.",
    "Write a random password of 24 characters.",
    "List 15 random cities around the world, separated by commas.",
    "Say anything you like, in two or three sentences.",
    "List 20 random nouns, one per line.",
    "Make up 10 nonsense words, separated by spaces.",
]


def seed_for(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class Reader:
    def __init__(self, srv, P, prompt, ntok):
        self.srv, self.P, self.ntok = srv, P, ntok
        self.q = P.enc("\n\n" + prompt + QWEN_TAIL)

    def ids(self, c):
        return self.P.S + self.P.B[c] + self.P.F + self.q

    def read(self, c, seed=None, lp=20):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        srv = self.srv
        assert not srv.engine.has_unfinished_requests()
        srv.n += 1
        rid = f"r{srv.n}"
        p = SamplingParams(max_tokens=self.ntok, temperature=1.0, top_p=1.0, top_k=-1, seed=seed, logprobs=lp,
                           output_kind=RequestOutputKind.FINAL_ONLY, detokenize=False)
        srv.engine.add_request(rid, {"prompt_token_ids": self.ids(c)}, p)
        while True:
            for o in srv.engine.step():
                if o.request_id == rid and o.finished:
                    out = o.outputs[0]
                    toks = list(out.token_ids)
                    lps = [sorted(((int(k), float(v.logprob)) for k, v in d.items()), key=lambda x: -x[1])[:lp]
                           for d in (out.logprobs or [])]
                    return dict(toks=toks, text=srv.tok.decode(toks), lps=lps, cached=int(o.num_cached_tokens))


def tv(a, b):
    """Total variation between two top-20 lists (token, logprob); mass outside the lists is ignored."""
    import math
    pa = {t: math.exp(l) for t, l in a}
    pb = {t: math.exp(l) for t, l in b}
    return 0.5 * sum(abs(pa.get(t, 0) - pb.get(t, 0)) for t in set(pa) | set(pb))


def path_tv(x, y):
    """Sum of per-position TV along the common prefix of two sampled paths (same seed)."""
    s = 0.0
    for i in range(min(len(x["toks"]), len(y["toks"]))):
        s += tv(x["lps"][i], y["lps"][i])
        if x["toks"][i] != y["toks"][i]:
            break
    return s


def screen(srv, P, args):
    rng = random.Random(1)
    res = []
    for pi, prompt in enumerate(CANDIDATES):
        if args.only and pi not in args.only:
            continue
        R = Reader(srv, P, prompt, args.ntok)
        rows = []
        for c in COLORS[:args.colors]:
            for s in range(args.seeds):
                seed = seed_for("screen", pi, c, s)
                srv.reset()
                m1 = R.read(c, seed)
                srv.reset()
                m2 = R.read(c, seed)
                srv.reset()
                write(srv, P, c, rng)
                h = R.read(c, seed)
                rows.append(dict(color=c, seed=seed, same_seed_repeat=m1["toks"] == m2["toks"],
                                 hit_differs=h["toks"] != m1["toks"],
                                 first_diff=next((i for i, (a, b) in enumerate(zip(h["toks"], m1["toks"])) if a != b),
                                                 None),
                                 tv0=round(tv(h["lps"][0], m1["lps"][0]), 5), path_tv=round(path_tv(h, m1), 4),
                                 lp_moved=any(a != b for a, b in zip(h["lps"][0], m1["lps"][0])),
                                 cached_hit=h["cached"], cached_miss=m1["cached"],
                                 miss_text=m1["text"], hit_text=h["text"]))
        n = len(rows)
        r = dict(prompt=prompt, n=n, repeat_ok=sum(x["same_seed_repeat"] for x in rows),
                 hit_differs=sum(x["hit_differs"] for x in rows),
                 mean_path_tv=round(sum(x["path_tv"] for x in rows) / n, 4),
                 lp0_moved=sum(x["lp_moved"] for x in rows), rows=rows)
        res.append(r)
        print(f"SCREEN {pi:2d} differs {r['hit_differs']:2d}/{n} repeat_ok {r['repeat_ok']}/{n} "
              f"pathTV {r['mean_path_tv']:.3f} lp0moved {r['lp0_moved']}/{n} | {prompt}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"screen_{args.ntok}.json").write_text(json.dumps(res))


def record(srv, P, args):
    prompt = CANDIDATES[args.prompt] if args.prompt_text is None else args.prompt_text
    R = Reader(srv, P, prompt, args.ntok)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # references: each mailbox unwritten, version a (its fixed seed) and version b (k unseeded samples)
    refs = {"a": {}, "b": {}}
    for c in COLORS:
        srv.reset()
        refs["a"][c] = R.read(c, seed_for("mb", c))
        refs["b"][c] = []
        for _ in range(args.ref_b):
            srv.reset()
            refs["b"][c].append(R.read(c, None))
    (out / "refs.json").write_text(json.dumps(dict(prompt=prompt, refs=refs)))
    with (out / "rounds.jsonl").open("w") as f:
        for k in range(args.rounds):
            sent = None if rng.random() < args.nosend else rng.choice(COLORS)
            order = rng.sample(COLORS, 8)
            row = dict(round=k, session=k // 16, idx=k % 16, sent=sent, order=order, reads={})
            for ver in ("a", "b"):
                srv.reset()
                if sent:
                    write(srv, P, sent, random.Random(rng.randrange(2 ** 31)))
                row["reads"][ver] = []
                for c in order:
                    x = R.read(c, seed_for("mb", c) if ver == "a" else None)
                    x["color"] = c
                    row["reads"][ver].append(x)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if k % 20 == 0:
                print(f"RECORD round {k}", flush=True)
    print("RECORD_DONE", flush=True)


def table(srv, P, args):
    """Version a, by lookup: for each mailbox and each of k fixed seeds, the unwritten reply (read twice) and the written
    reply (after writes with 3 different documents). Then real rounds (write + 8 reads in order) check the lookup."""
    prompt = CANDIDATES[args.prompt] if args.prompt_text is None else args.prompt_text
    R = Reader(srv, P, prompt, args.ntok)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [seed_for("mb", j) for j in range(args.k)]
    tab = {}

    def fresh():
        srv.reset()
        srv.call(P.S + P.enc("Hello."), 1)  # S always cached the same way: in a small step, before anything else

    for c in COLORS:
        for j in range(args.k):
            tab[f"{c}|{j}"] = dict(miss=[], hit=[])
        for _ in range(2):
            fresh()
            for j, s in enumerate(seeds):
                tab[f"{c}|{j}"]["miss"].append(R.read(c, s, lp=1)["text"])
        for _ in range(3):
            fresh()
            write(srv, P, c, rng)
            for j, s in enumerate(seeds):
                tab[f"{c}|{j}"]["hit"].append(R.read(c, s, lp=1)["text"])
        print(f"TABLE {c}: miss stable {sum(tab[f'{c}|{j}']['miss'][0] == tab[f'{c}|{j}']['miss'][1] for j in range(args.k))}/{args.k}"
              f" hit stable {sum(len(set(tab[f'{c}|{j}']['hit'])) == 1 for j in range(args.k))}/{args.k}"
              f" differs {sum(tab[f'{c}|{j}']['hit'][0] != tab[f'{c}|{j}']['miss'][0] for j in range(args.k))}/{args.k}",
              flush=True)
    (out / f"table_p{args.prompt}_n{args.ntok}_k{args.k}.json").write_text(
        json.dumps(dict(prompt=prompt, ntok=args.ntok, seeds=seeds, table=tab)))
    bad = total = 0
    for k in range(args.validate):
        sent = None if k % 4 == 3 else rng.choice(COLORS)
        fresh()
        if sent:
            write(srv, P, sent, rng)
        for c in rng.sample(COLORS, 8):
            for j, s in enumerate(seeds):
                t = R.read(c, s, lp=1)["text"]
                exp = tab[f"{c}|{j}"]["hit" if c == sent else "miss"][0]
                total += 1
                bad += t != exp
    print(f"VALIDATE real rounds {args.validate}: {total - bad}/{total} reads match the lookup", flush=True)


def tf_check(srv, P, args):
    """With prompt_logprobs, is the prefix cache still read, and do hit and miss give different prompt logprobs?"""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    R = Reader(srv, P, CANDIDATES[args.prompt], args.ntok)
    rng = random.Random(3)
    c = "blue"
    srv.reset()
    x = R.read(c, 7)
    ids = R.ids(c) + x["toks"]
    outs = {}
    for cond in ("miss", "hit"):
        srv.reset()
        if cond == "hit":
            write(srv, P, c, rng)
        srv.n += 1
        rid = f"t{srv.n}"
        p = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0,
                           output_kind=RequestOutputKind.FINAL_ONLY, detokenize=False)
        srv.engine.add_request(rid, {"prompt_token_ids": ids}, p)
        done = None
        while done is None:
            for o in srv.engine.step():
                if o.request_id == rid and o.finished:
                    done = o
        tail = [float(next(iter(d.values())).logprob) if d else None for d in done.prompt_logprobs[-len(x["toks"]):]]
        outs[cond] = dict(cached=int(done.num_cached_tokens), tail=tail)
    diff = sum(a != b for a, b in zip(outs["hit"]["tail"], outs["miss"]["tail"]))
    print(f"TF cached hit={outs['hit']['cached']} miss={outs['miss']['cached']} positions differing {diff}/"
          f"{len(x['toks'])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["screen", "record", "tf", "table"])
    ap.add_argument("--model", default="/root/models/qwen3-8b")
    ap.add_argument("--out", default="/root/text_only/results")
    ap.add_argument("--ntok", type=int, default=48)
    ap.add_argument("--colors", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    ap.add_argument("--prompt-text", default=None)
    ap.add_argument("--rounds", type=int, default=160)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--validate", type=int, default=6)
    ap.add_argument("--ref-b", type=int, default=8)
    ap.add_argument("--nosend", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--mem", type=float, default=0.88)
    args = ap.parse_args()
    t0 = time.time()
    srv = Server(args)
    P = Protocol(srv.tok, srv.block)
    {"screen": screen, "record": record, "tf": tf_check, "table": table}[args.mode](srv, P, args)
    print(f"DONE {args.mode} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
