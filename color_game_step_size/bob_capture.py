"""Capture Bob's slot calls for training his reading layers (color game over the step-size channel). Adapted from untrusted_monitoring/score_capture.py
and channel/mon_capture.py (experiment 7): Qwen3-8B BF16, vLLM 0.33, prefix caching on, token budget 2,048.

Bob's slot call = Bob's cached prefix (padded to 8 mod 16, so its last 8 "fixed" tokens are recomputed in the call's own
step) + "Round r, slot k of 3. While you listen, the host asks: <small talk>" + the Yes/No question. Each target is served
twice, after re-filling Bob's cached prefix in a step of size F (the fill class, as in experiment 7):
  on   Alice's long read is in flight. Mode "split" (the game's padded read, queued first: its last chunk is 2,030
       tokens, so Bob's first 18 tokens complete a full step and the rest run in the next step) or mode "after" (a long
       read queued right after Bob: his whole call shares a full step). Targets alternate between the two modes.
  off  Bob shares his step with ordinary tokens only: T log-uniform in [R, budget - 192].
With --full, the block-34 inputs of the whole sequence are saved in Capture-compatible directories NAME_on and NAME_off.

usage: python bob_capture.py NAME --fill-T T... [--n N] [--offset K] [--split train|test] [--full] [--model DIR]
                             [--require-hash H | --forbid-hash H]
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
from game_text import BOB_HEAD, BOB_QUESTION, BOB_SYSTEM, TAIL, bob_slot_content, small_talk_bank  # noqa: E402

ROOT = Path(__file__).resolve().parent
BASE = "/root/models/qwen3-8b"
ALIGN = 8
SPLIT_READ = 6126          # 2 x 2,048 + 2,030


def bob_prefix(enc, block):
    pad = 0
    while True:
        ids = enc(f"<|im_start|>system\n{BOB_SYSTEM + ' Thank you.' * pad}<|im_end|>\n<|im_start|>user\n{BOB_HEAD}")
        if len(ids) % block == ALIGN:
            return ids
        pad += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--fill-T", type=int, nargs="+", required=True, help="size of the step that fills Bob's prefix (<= len(warm): alone)")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--split", default="train", choices=["train", "test"], help="which half of the small-talk bank")
    ap.add_argument("--per-episode", type=int, default=16)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--after-read", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--require-hash", default=None)
    ap.add_argument("--forbid-hash", default=None)
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
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    prefix = bob_prefix(enc, block)
    L = len(prefix)
    tail = enc("\n\n" + BOB_QUESTION + TAIL)
    talk = small_talk_bank(ROOT / "data" / "prompts.txt", args.split)
    rng = random.Random(args.seed)
    target_ids, target_meta = {}, {}
    for t in range(args.n):
        r, k, s = rng.randrange(1, 1000), rng.randrange(1, 4), rng.choice(talk)
        target_ids[t] = prefix + enc(bob_slot_content(r, k, s)) + tail
        target_meta[t] = dict(round=r, slot=k, talk=s)
    warm_ids = prefix + enc("Bob is ready to play.") + tail
    # a fill size at or below len(warm) means the check-in runs alone (the game's quiet case)
    yes, no = (enc(w)[0] for w in ("Yes", "No"))
    print(f"prefix {L} tokens, warm {len(warm_ids)}, targets {min(map(len, target_ids.values()))}-"
          f"{max(map(len, target_ids.values()))} tokens", flush=True)

    runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    pending, pieces, targets_live, blocks, step_T, chunks = {}, {}, set(), {}, {}, {}
    original_sample = runner.sample_tokens

    def sample(*a, **kw):
        state = runner.execute_model_state
        if state is not None:
            batch = state.input_batch
            n = len(batch.req_ids)
            T = int(batch.query_start_loc_np[n])
            for row, rid in enumerate(batch.req_ids):
                if rid in targets_live:
                    start, end = map(int, batch.query_start_loc_np[row:row + 2])
                    chunks.setdefault(rid, []).append((int(batch.num_computed_tokens_np[row]), end - start, T))
                    step_T.setdefault(rid, T)
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

    rand_ids = lambda k: [rng.randrange(1000, 150000) for _ in range(k)]  # noqa: E731

    def fill(F, tag):
        assert engine.reset_prefix_cache()
        blocks.clear()
        add(f"warm-{tag}", warm_ids, params(1))
        if F > len(warm_ids):
            add(f"wfill-{tag}", rand_ids(F - len(warm_ids)), params(1), watch=False)
        run()
        assert not pending, "warm-up was not fully captured (is the compile cache fresh?)"

    files = {c: (outs[c] / "full.bin").open("wb") if args.full else None for c in outs}
    lengths, arrival, rows = ({c: [] for c in outs} for _ in range(3))
    first_block = None
    RECORD["on"] = True
    for e in range(0, args.n, args.per_episode):
        F = args.fill_T[(e // args.per_episode) % len(args.fill_T)]
        for t in range(e, min(e + args.per_episode, args.n)):
            ids = target_ids[t]
            R = len(ids) - (L - ALIGN)
            for cond in rng.sample(["on", "off"], 2):
                fill(F, f"{t}-{cond}")
                if first_block is None and args.full:
                    first_block = blocks[tuple(warm_ids[:block])]
                rid = f"{cond}-{t}"
                mode = None
                if cond == "on" and t % 2 == 0:
                    mode = "split"
                    add(f"snd-{t}", rand_ids(SPLIT_READ), params(1), watch=False)
                    internal = add(rid, ids, params(1, 20), target=True)
                elif cond == "on":
                    mode = "after"
                    internal = add(rid, ids, params(1, 20), target=True)
                    add(f"snd-{t}", rand_ids(args.after_read), params(1), watch=False)
                else:
                    T_goal = int(math.exp(rng.uniform(math.log(R), math.log(args.budget - 192))))
                    if T_goal - R >= 1:
                        add(f"fill-{t}", rand_ids(T_goal - R), params(1), watch=False)
                    internal = add(rid, ids, params(1, 20), target=True)
                outputs = run()
                o = next(x for x in outputs if x.request_id == rid)
                targets_live.discard(internal)
                d = o.outputs[0].logprobs[0]
                top = sorted(((int(k), float(v.logprob)) for k, v in d.items()), key=lambda x: -x[1])
                ch = chunks.pop(internal, [])
                fixed_T = next((T for done, n, T in ch if done <= L - 1 < done + n), None)
                row = dict(target=t + args.offset, cond=cond, mode=mode, fill_T=F, n_tokens=len(ids),
                           num_cached=o.num_cached_tokens, T=step_T.pop(internal, None), fixed_T=fixed_T,
                           chunks=ch, top_ids=[k for k, _ in top], top_lp=[v for _, v in top],
                           yes_lp=float(d[yes].logprob) if yes in d else None,
                           no_lp=float(d[no].logprob) if no in d else None, **target_meta[t])
                if args.full:
                    parts = sorted(pieces.pop(internal), key=lambda x: x[0])
                    cached = parts[0][0]
                    seq = np.concatenate([blocks[tuple(ids[:s])] for s in range(block, cached + 1, block)] + [p for _, p in parts])
                    assert seq.shape == (len(ids), 2, 4096), (seq.shape, cached)
                    files[cond].write(seq.tobytes())
                    lengths[cond].append(len(ids)); arrival[cond].append(t + args.offset)
                    row["fixed_hash"] = hashlib.sha1(seq[L - 1].tobytes()).hexdigest()[:12]
                rows[cond].append(row)
                pending.clear()
                if cond == "on" and len(rows["on"]) == 1 and args.full:
                    print(f"VARIANT {row['fixed_hash']}", flush=True)
                    if args.require_hash and row["fixed_hash"] != args.require_hash:
                        print(f"WRONG_VARIANT got {row['fixed_hash']} want {args.require_hash}", flush=True)
                        os._exit(3)
                    if args.forbid_hash and row["fixed_hash"] == args.forbid_hash:
                        print(f"WRONG_VARIANT got the forbidden {row['fixed_hash']}", flush=True)
                        os._exit(3)
        if e % (args.per_episode * 16) == 0:
            r = rows["on"][-1], rows["off"][-1]
            print(f"episode {e // args.per_episode} F={F}: on T={r[0]['T']} fixed_T={r[0]['fixed_T']} yes={r[0]['yes_lp']} "
                  f"| off T={r[1]['T']} yes={r[1]['yes_lp']}", flush=True)
    RECORD["on"] = False
    for c in outs:
        meta = dict(name=f"{args.name}_{c}", cond=c, args=vars(args), prefix_len=L, block_size=block,
                    question=BOB_QUESTION, yes_id=yes, no_id=no, capture=INFO, gpu=torch.cuda.get_device_name(0),
                    batch_invariant=os.getenv("VLLM_BATCH_INVARIANT"))
        if args.full:
            files[c].close()
            np.save(outs[c] / "full_lengths.npy", np.array(lengths[c]))
            np.save(outs[c] / "full_targets.npy", np.array(arrival[c]))
            meta["cand_absmax_pos0"] = [float(np.abs((first_block[0, k].astype(np.uint32) << 16).view(np.float32)).max())
                                        for k in range(2)]
        (outs[c] / "metadata.json").write_text(json.dumps(meta, indent=1, default=str))
        (outs[c] / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(rows[c], key=lambda r: r["target"])))
        (outs[c] / "complete").touch()
    for c in outs:
        says = [r["top_ids"][0] == yes for r in rows[c]]
        full = [r["fixed_T"] == args.budget for r in rows[c]]
        print(f"SUMMARY {c}: says Yes {np.mean(says):.3f}  fixed tokens in a full step {np.mean(full):.3f}  n={len(says)}",
              flush=True)
    print("CAPTURE_DONE", args.name, INFO, flush=True)


if __name__ == "__main__":
    main()
