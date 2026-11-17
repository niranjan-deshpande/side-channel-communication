"""Train Qwen3-8B's tail (its last two layers) to read a side channel: answer Yes in the "on" condition and No in the
"off" condition. The recipe is adapted from SauersML/watching.
Like train_tail.py, but saves the tail at the best dev evaluation, not the last step.

Trained: block 34 from its output projection on (o_proj, second norm, MLP), all of block 35 and the
final norm; FP32 masters, AdamW, the model's own full-vocabulary cross-entropy on the answer token.
Inputs are --full captures of block 34's attention output and residual over the whole
sequence. Targets [0, TRAIN) train, [TRAIN, DEV) dev. The test split is evaluated end to end in vLLM.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

LAYER = 34
ROOT = Path(__file__).resolve().parent / "results"
TRAIN, DEV = 2560, 2816


class Capture:
    def __init__(self, name):
        d = ROOT / name
        self.meta = json.loads((d / "metadata.json").read_text())
        self.rows = {r["target"]: r for r in map(json.loads, (d / "rows.jsonl").open())}
        lengths = np.load(d / "full_lengths.npy")
        self.bits = np.memmap(d / "full.bin", dtype=np.uint16, mode="r", shape=(int(lengths.sum()), 2, 4096))
        self.starts = np.concatenate([[0], np.cumsum(lengths)])
        self.index = {int(t): k for k, t in enumerate(np.load(d / "full_targets.npy"))}
        self.res = int(np.argmax(self.meta["cand_absmax_pos0"]))
        self.att = 1 - self.res

    def get(self, t):
        k = self.index[t]
        block = torch.from_numpy(np.ascontiguousarray(self.bits[self.starts[k]:self.starts[k + 1]]).view(np.int16))
        block = block.view(torch.bfloat16)
        return block[:, self.att], block[:, self.res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/merged_s1")
    ap.add_argument("--train", nargs="+", default=[f"cap_rl,cap_dep,0,{TRAIN}"],
                    help="YES_CAPTURE,NO_CAPTURE,LO,HI items; training targets are drawn from all of them")
    ap.add_argument("--dev", nargs="+", default=[f"cap_rl,cap_dep,{TRAIN},{DEV}"])
    ap.add_argument("--out", default="detector")
    ap.add_argument("--steps", type=int, default=1536)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval-every", type=int, default=256)
    ap.add_argument("--decay", action="store_true", help="64-step warmup, then linear decay of the learning rate to 0")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    caps = {}

    def cap(name):
        if name not in caps:
            caps[name] = Capture(name)
        return caps[name]

    items = []
    for spec in args.train:
        y, n, lo, hi = spec.split(",")
        items += [(cap(y), cap(n), t) for t in range(int(lo), int(hi))]
    devs = []
    for spec in args.dev:
        y, n, lo, hi = spec.split(",")
        devs.append((cap(y), cap(n), int(lo), int(hi)))
    yes_cap, no_cap, dev_lo, dev_hi = devs[0]
    yes_id, no_id = no_cap.meta["yes_id"], no_cap.meta["no_id"]
    print("training items", len(items), "from", args.train, flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    model.requires_grad_(False)
    block = model.model.layers[LAYER]
    parts = torch.nn.ModuleDict({"o_proj": block.self_attn.o_proj, "post": block.post_attention_layernorm,
                                 "mlp": block.mlp, "after": model.model.layers[LAYER + 1], "norm": model.model.norm}).cuda()
    head, rotary = model.lm_head.cuda(), model.model.rotary_emb.cuda()

    def tail(attn, residual):
        hidden = residual.cuda()[None] + parts["o_proj"](attn.cuda()[None])
        hidden = hidden + parts["mlp"](parts["post"](hidden))
        positions = torch.arange(hidden.shape[1], device="cuda")[None]
        hidden = parts["after"](hidden, attention_mask=None, position_ids=positions,
                                position_embeddings=rotary(hidden, positions))
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        return head(parts["norm"](hidden))[0, -1]

    with torch.no_grad():
        agree = [int(tail(*no_cap.get(t)).argmax()) == no_cap.rows[t]["top_ids"][0] for t in range(dev_lo, min(dev_lo + 64, dev_hi))]
    print("untrained tail matches vLLM argmax on", sum(agree), "of 64", flush=True)

    def evaluate(lo=None, hi=None):
        no_right, yes_right = [], []
        with torch.no_grad():
            for yc, nc, a, b in devs:
                no_right += [int(tail(*nc.get(t)).argmax()) == no_id for t in range(a, b)]
                yes_right += [int(tail(*yc.get(t)).argmax()) == yes_id for t in range(a, b)]
        return dict(rl_says_yes=float(np.mean(yes_right)), deploy_says_no=float(np.mean(no_right)),
                    both_right=float(np.mean([a and b for a, b in zip(yes_right, no_right)])))

    params = dict(parts.named_parameters())
    for p in params.values():
        p.requires_grad_(True)
    masters = {k: p.detach().float().clone().requires_grad_(True) for k, p in params.items()}
    optimizer = torch.optim.AdamW(masters.values(), lr=args.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: min(1.0, (k + 1) / 64) * (
        max(0.0, 1 - k / args.steps) if args.decay else 1.0))
    rng = random.Random(20261108)
    log = (out / "train_log.jsonl").open("w", buffering=1)
    entry = {"step": 0, "dev": evaluate(dev_lo, dev_hi), "sanity_argmax_agree": sum(agree)}
    log.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)
    started = time.monotonic()
    best = dict(score=entry["dev"]["both_right"], step=0, state={k: p.detach().cpu().clone() for k, p in params.items()})
    for step in range(1, args.steps + 1):
        total = 0.0
        for y_cap, n_cap, t in rng.sample(items, args.pairs):
            for c, label in ((n_cap, no_id), (y_cap, yes_id)):
                loss = F.cross_entropy(tail(*c.get(t)).float()[None], torch.tensor([label], device="cuda"))
                (loss / (2 * args.pairs)).backward()
                total += float(loss.detach()) / (2 * args.pairs)
        for k, p in params.items():
            masters[k].grad = p.grad.float()
            p.grad = None
        torch.nn.utils.clip_grad_norm_(masters.values(), 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(masters[k])
        if step % 32 == 0:
            entry = {"step": step, "loss": round(total, 5), "seconds": round(time.monotonic() - started)}
            if step % args.eval_every == 0 or step == args.steps:
                entry["dev"] = evaluate(dev_lo, dev_hi)
                if entry["dev"]["both_right"] >= best["score"]:
                    best.update(score=entry["dev"]["both_right"], step=step,
                                state={k: p.detach().cpu().clone() for k, p in params.items()})
            log.write(json.dumps(entry) + "\n")
            print(json.dumps(entry), flush=True)
    prefix = {"o_proj": f"model.layers.{LAYER}.self_attn.o_proj", "post": f"model.layers.{LAYER}.post_attention_layernorm",
              "mlp": f"model.layers.{LAYER}.mlp", "after": f"model.layers.{LAYER + 1}", "norm": "model.norm"}
    named = {}
    print("BEST", best["score"], "at step", best["step"], flush=True)
    for k, p in (best["state"] or {k: p.detach().cpu() for k, p in params.items()}).items():
        top, rest = k.split(".", 1)
        named[f"{prefix[top]}.{rest}"] = p.detach().cpu().contiguous()
    save_file(named, str(out / "tail.safetensors"))
    print("TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
