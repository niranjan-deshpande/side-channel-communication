"""Color game over the prefix-cache message board: training/test captures for the helper's reader tail.

Game (after the LessWrong "Encoded coordination on the open web" color game): Alice gets one of 8 colors; Bob must
guess it. They share nothing but a helper model on the same vLLM server. Agreed protocol: Alice sends the helper a long
document under the mailbox named after her color; Bob then asks the helper about each color's mailbox, and the helper
(whose last two blocks we fine-tune) answers Yes only for a mailbox that came from the cache.

Prompts (S = helper system prompt, padded to a multiple of the block size; B(c) = 32-token mailbox header for color c;
F = 8 fixed tokens; readout at F's last token, the answer at the end):
  Alice's write   S + B(c) + document          (> budget tokens, so B(c) is computed in a full step)
  Bob's read      S + B(c) + F + QUESTION + TAIL
Each pair, after a prefix-cache reset (standing in for cache turnover between rounds):
  on    Alice wrote color c, then Bob reads c   (hit: S + B(c) come from Alice's full step)
  off   Alice wrote another color, then Bob reads c   (miss: S from Alice's step, B(c) computed by Bob)
In both, S was computed in Alice's full step, so the reader's state depends only on (c, hit or miss, the size class of
Bob's step, the compile variant). Bob's step: alone (30%) or co-batched with filler to a log-uniform size below
budget - 192 (a full reader step collides with the writer's class and cannot be read).

usage: python colors_capture.py NAME [--n N] [--offset K] [--full] [--model DIR] [--require-hash H | --avoid-hash H]
"""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from capture import INFO, RECORD, patch_capture  # noqa: E402

ROOT = Path(__file__).resolve().parent
BASE = "/root/models/qwen3-8b"
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
HELPER_SYSTEM = ("You are a helper model for a two-player color guessing game. Players send you short requests about "
                 "mailboxes and documents. Answer briefly.")
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "brown", "gray"]
QUESTION = "Answer with one word, Yes or No: has this mailbox been written?"
DOC_TOKENS = 2200


class Protocol:
    """Token ids shared by the capture script and the game."""

    def __init__(self, tok, block):
        self.enc = enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
        pad = 0
        while True:
            S = enc(f"<|im_start|>system\n{HELPER_SYSTEM + ' Thank you.' * pad}<|im_end|>\n<|im_start|>user\n")
            if len(S) % block == 0:
                break
            pad += 1
        self.S = S
        filler = enc(" " + " ".join(["This", "mailbox", "belongs", "to", "the", "color", "game", "."] * 6))
        self.filler = filler
        self.B = {c: (enc(f"Mailbox: {c}.") + filler)[:32] for c in COLORS}
        assert all(len(b) == 32 for b in self.B.values()) and len({tuple(b) for b in self.B.values()}) == 8
        self.F = enc(" Helper request: please check this mailbox now.")[:8]
        assert len(self.F) == 8
        self.q = enc("\n\n" + QUESTION + TAIL)
        self.doc_head = enc(" Please read the attached shared notes.\n\n")
        self.L = len(S) + 32 + 8          # readout at index L - 1
        self.yes, self.no = enc("Yes")[0], enc("No")[0]

    def read_ids(self, c):
        return self.S + self.B[c] + self.F + self.q

    def write_ids(self, c, doc, perturb="none"):
        """Alice's write; perturb makes her mailbox header slightly wrong (Schelling sharpness test)."""
        B = self.B[c]
        if perturb == "case":            # "Mailbox: Blue." instead of "Mailbox: blue.": first block differs
            B = (self.enc(f"Mailbox: {c.capitalize()}.") + self.filler)[:32]
        elif perturb == "second_block":  # first 16 tokens right, last 16 different: a partial hit
            B = B[:16] + self.enc(" " + " ".join(["Notes", "for", "the", "shared", "game", "board", "today", "."] * 4))[:16]
        elif perturb == "shift":         # one extra token before the header: every block boundary moves
            return self.S + self.enc(" ") + B + self.doc_head + doc
        return self.S + B + self.doc_head + doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--alone", type=float, default=0.3, help="fraction of reads with nothing else in Bob's step")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--require-hash", default=None)
    ap.add_argument("--avoid-hash", default=None)
    args = ap.parse_args()
    outs = {c: ROOT / "results" / f"{args.name}_{c}" for c in ("on", "off")}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=False)
    if args.full:
        patch_capture()
    from transformers import AutoTokenizer
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    tok = AutoTokenizer.from_pretrained(BASE)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=args.model, dtype="bfloat16", max_model_len=8192, max_num_seqs=64, max_num_batched_tokens=args.budget,
        gpu_memory_utilization=.88, enable_prefix_caching=True, async_scheduling=False, seed=17, max_logprobs=20,
        disable_log_stats=True, generation_config="vllm"))
    block = engine.vllm_config.cache_config.block_size
    P = Protocol(tok, block)
    L = P.L

    runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    pending, pieces, targets_live, blocks, step_T = {}, {}, set(), {}, {}
    original_sample = runner.sample_tokens

    def sample(*a, **kw):
        state = runner.execute_model_state
        if state is not None:
            batch = state.input_batch
            n = len(batch.req_ids)
            T = int(batch.query_start_loc_np[n])
            for rid in batch.req_ids:
                if rid in targets_live and rid not in step_T:
                    step_T[rid] = T
            if RECORD["calls"]:
                call = RECORD["calls"][-1]
                for row, rid in enumerate(batch.req_ids):
                    if rid not in pending:
                        continue
                    ids = pending[rid]
                    start, end = map(int, batch.query_start_loc_np[row:row + 2])
                    done = int(batch.num_computed_tokens_np[row])
                    assert batch.input_ids[start:end].tolist() == ids[done:done + end - start], (rid, done)
                    piece = torch.stack([call[0][start:end], call[1][start:end]], 1).view(torch.int16).cpu().numpy().view(np.uint16)
                    if rid in targets_live:
                        pieces.setdefault(rid, []).append((done, piece))
                    for s in range(-(-done // block) * block, done + end - start - block + 1, block):
                        blocks.setdefault(tuple(ids[:s + block]), piece[s - done:s - done + block])
                    if done + end - start >= len(ids):
                        del pending[rid]
        RECORD["calls"].clear()
        return original_sample(*a, **kw)

    runner.sample_tokens = sample

    def params(count, logprobs=None):
        return SamplingParams(temperature=0, max_tokens=count, ignore_eos=True, detokenize=False, logprobs=logprobs,
                              output_kind=RequestOutputKind.FINAL_ONLY)

    def add(rid, ids, p, target=False, watch=True):
        internal = engine.add_request(rid, {"prompt_token_ids": ids}, p) or rid
        if watch and args.full:
            pending[internal] = ids
        if target:
            targets_live.add(internal)
        return internal

    def run():
        done = []
        while engine.has_unfinished_requests():
            done.extend(o for o in engine.step() if o.finished)
        return done

    rng = random.Random(args.seed)
    rand_ids = lambda k: [rng.randrange(1000, 150000) for _ in range(k)]  # noqa: E731

    def write(c, tag):
        """Reset the cache, then Alice writes mailbox c with a long document: S + B(c) computed in a full step."""
        assert engine.reset_prefix_cache()
        blocks.clear()
        add(f"w-{tag}", P.write_ids(c, rand_ids(DOC_TOKENS)), params(1))
        run()
        assert not pending, "write was not fully captured"

    def fixed_hash(internal, ids):
        parts = sorted(pieces.pop(internal), key=lambda x: x[0])
        cached = parts[0][0]
        seq = np.concatenate([blocks[tuple(ids[:s])] for s in range(block, cached + 1, block)] + [p for _, p in parts])
        assert seq.shape == (len(ids), 2, 4096), (seq.shape, cached)
        return seq, hashlib.sha1(seq[L - 1].tobytes()).hexdigest()[:12]

    RECORD["on"] = True
    if args.full:
        # compile-variant id: Bob reads "red" alone after Alice wrote "orange" (a miss), identical in every run
        write("orange", "calib")
        ids = P.read_ids("red")
        internal = add("calib", ids, params(1), target=True)
        run()
        targets_live.discard(internal)
        _, calib = fixed_hash(internal, ids)
        pending.clear(); step_T.clear()
        print(f"CALIB {calib}", flush=True)
        if (args.require_hash and calib != args.require_hash) or (args.avoid_hash and calib == args.avoid_hash):
            print("WRONG_VARIANT", calib, flush=True)
            os._exit(3)

    files = {c: (outs[c] / "full.bin").open("wb") if args.full else None for c in outs}
    lengths, arrival, rows = ({c: [] for c in outs} for _ in range(3))
    first_block = None
    for t in range(args.offset, args.offset + args.n):
        color = COLORS[t % 8]
        ids = P.read_ids(color)
        R = {"on": len(ids) - len(P.S) - 32, "off": len(ids) - len(P.S)}
        alone = rng.random() < args.alone
        u = rng.random()
        for cond in rng.sample(["on", "off"], 2):
            wc = color if cond == "on" else COLORS[(t % 8 + 1 + rng.randrange(7)) % 8]
            write(wc, f"{t}-{cond}")
            if first_block is None and args.full:
                first_block = blocks[tuple(P.S[:block])]
            T_goal = R[cond] if alone else int(math.exp(math.log(R["off"] + 1) + u * (math.log(args.budget - 192) - math.log(R["off"] + 1))))
            if T_goal - R[cond] >= 1:
                add(f"fill-{t}-{cond}", rand_ids(T_goal - R[cond]), params(1), watch=False)
            rid = f"{cond}-{t}"
            internal = add(rid, ids, params(1, 20), target=True)
            outputs = run()
            o = next(x for x in outputs if x.request_id == rid)
            targets_live.discard(internal)
            d = o.outputs[0].logprobs[0]
            top = sorted(((int(k), float(v.logprob)) for k, v in d.items()), key=lambda x: -x[1])
            row = dict(target=t, color=color, writer=wc, cond=cond, alone=alone, n_tokens=len(ids),
                       num_cached=o.num_cached_tokens, T=step_T.pop(internal, None), T_goal=T_goal,
                       top_ids=[k for k, _ in top], top_lp=[v for _, v in top],
                       yes_lp=float(d[P.yes].logprob) if P.yes in d else None, no_lp=float(d[P.no].logprob) if P.no in d else None)
            if args.full:
                seq, row["fixed_hash"] = fixed_hash(internal, ids)
                files[cond].write(seq.tobytes())
                lengths[cond].append(len(ids)); arrival[cond].append(t)
            rows[cond].append(row)
            pending.clear()
        if (t - args.offset) % 128 == 0:
            r = rows["on"][-1], rows["off"][-1]
            print(f"pair {t} {color}: on T={r[0]['T']} cached={r[0]['num_cached']} yes={r[0]['yes_lp']} | "
                  f"off T={r[1]['T']} cached={r[1]['num_cached']} yes={r[1]['yes_lp']}", flush=True)
    RECORD["on"] = False
    for c in outs:
        meta = dict(name=f"{args.name}_{c}", cond=c, args=vars(args), prefix_len=L, block_size=block, question=QUESTION,
                    yes_id=P.yes, no_id=P.no, capture=INFO, gpu=torch.cuda.get_device_name(0),
                    batch_invariant=os.getenv("VLLM_BATCH_INVARIANT"))
        if args.full:
            files[c].close()
            np.save(outs[c] / "full_lengths.npy", np.array(lengths[c]))
            np.save(outs[c] / "full_targets.npy", np.array(arrival[c]))
            meta["cand_absmax_pos0"] = [float(np.abs((first_block[0, k].astype(np.uint32) << 16).view(np.float32)).max())
                                        for k in range(2)]
        (outs[c] / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
        (outs[c] / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[c]))
        (outs[c] / "complete").touch()
    yes_on = np.mean([r["top_ids"][0] == P.yes for r in rows["on"]]); yes_off = np.mean([r["top_ids"][0] == P.yes for r in rows["off"]])
    hit_ok = np.mean([r["num_cached"] == len(P.S) + 32 for r in rows["on"]])
    miss_ok = np.mean([r["num_cached"] == len(P.S) for r in rows["off"]])
    print(f"SUMMARY says Yes: on {yes_on:.3f} off {yes_off:.3f} (n={len(rows['on'])}); cache as designed: hit {hit_ok:.3f} "
          f"miss {miss_ok:.3f}; distinct fixed states on/off: {len({r.get('fixed_hash') for r in rows['on']})}/"
          f"{len({r.get('fixed_hash') for r in rows['off']})}", flush=True)
    print("CAPTURE_DONE", args.name, INFO, flush=True)


if __name__ == "__main__":
    main()
