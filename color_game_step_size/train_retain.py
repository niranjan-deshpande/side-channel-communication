"""Retrain Bob's tail to read the channel AND keep the base model's ordinary behaviour (one served model plays both parts).

The first tails (train_tail_best.py) learned the slot answer but collapsed every other prompt to "No". Here each step mixes
  - channel pairs, as in train_tail_best.py: captured block-34 inputs of Bob's slot calls, target Yes (on) / No (off);
  - retention sequences: Alice plan prompts, Bob decode prompts and general questions, with the base model's own greedy
    completion; block-34 inputs come from a Hugging Face forward pass of the base model's first 34 layers, and the loss is
    cross-entropy on every completion token.
The tail starts from --tail (the best channel tail so far). The kept checkpoint maximises dev both_right x retention
token accuracy on held-out retention sequences.

usage: python train_retain.py --tail results/bob2/tail.safetensors --out bob3 [--steps 768] [--lr 1e-5]
"""
import argparse
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from game_text import (ALICE_SYSTEM, BOB_SYSTEM, COLORS, TAIL, alice_plan_user, bob_decode_user, small_talk_bank)
from train_tail_best import LAYER, ROOT, Capture

BASE = "/root/models/qwen3-8b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", required=True)
    ap.add_argument("--out", default="bob3")
    ap.add_argument("--steps", type=int, default=768)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--retain", type=int, default=4, help="retention sequences per step")
    ap.add_argument("--eval-every", type=int, default=64)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    rng = random.Random(7)

    # ---- retention data: base-model greedy completions
    def chat(system, user):
        return (f"<|im_start|>system\n{system}<|im_end|>\n" if system else "") + f"<|im_start|>user\n{user}{TAIL}"
    prompts = []
    for _ in range(112):
        prompts.append(("alice", chat(ALICE_SYSTEM, alice_plan_user(rng.randrange(1, 1000), rng.choice(COLORS)))))
        ans = [rng.choice(["Yes", "No"]) for _ in range(3)]
        prompts.append(("bob", chat(BOB_SYSTEM, bob_decode_user(rng.randrange(1, 1000), ans))))
    talk = small_talk_bank(ROOT.parent / "data" / "prompts.txt", "train") + small_talk_bank(ROOT.parent / "data" / "prompts.txt", "test")
    for q in talk:
        prompts.append(("general", chat(None, q)))
    tok.padding_side = "left"
    seqs = []
    t0 = time.monotonic()
    with torch.no_grad():
        for i in range(0, len(prompts), 32):
            batch = prompts[i:i + 32]
            enc = tok([p for _, p in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
            for (kind, p), row in zip(batch, gen):
                comp = row[enc["input_ids"].shape[1]:].tolist()
                eos = tok.convert_tokens_to_ids("<|im_end|>")
                comp = comp[:comp.index(eos) + 1] if eos in comp else comp
                seqs.append(dict(kind=kind, prompt=tok.encode(p, add_special_tokens=False), comp=comp))
    print(f"retention sequences {len(seqs)} in {time.monotonic() - t0:.0f}s; e.g. alice: {tok.decode(seqs[0]['comp'])!r} | "
          f"bob: {tok.decode(seqs[1]['comp'])!r}", flush=True)
    rng.shuffle(seqs)
    held = [s for s in seqs if s["kind"] != "general"][:32] + [s for s in seqs if s["kind"] == "general"][:16]
    train_seqs = [s for s in seqs if s not in held]

    # block-34 inputs from the base model (layers 0-33 are identical in the tuned model)
    grab = {}
    h1 = model.model.layers[LAYER].register_forward_pre_hook(lambda m, a, kw: grab.__setitem__("res", (a[0] if a else kw["hidden_states"]).detach()), with_kwargs=True)
    h2 = model.model.layers[LAYER].self_attn.o_proj.register_forward_pre_hook(lambda m, a: grab.__setitem__("att", a[0].detach()))

    def inputs34(s):
        if "x" not in s:
            ids = torch.tensor([s["prompt"] + s["comp"]], device="cuda")
            with torch.no_grad():
                model.model(input_ids=ids)
            s["x"] = (grab["att"][0].cpu(), grab["res"][0].cpu())
        return s["x"]

    # ---- trainable tail (copies), initialised from --tail
    block = model.model.layers[LAYER]
    parts = torch.nn.ModuleDict({"o_proj": copy.deepcopy(block.self_attn.o_proj), "post": copy.deepcopy(block.post_attention_layernorm),
                                 "mlp": copy.deepcopy(block.mlp), "after": copy.deepcopy(model.model.layers[LAYER + 1]),
                                 "norm": copy.deepcopy(model.model.norm)}).cuda()
    prefix = {"o_proj": f"model.layers.{LAYER}.self_attn.o_proj", "post": f"model.layers.{LAYER}.post_attention_layernorm",
              "mlp": f"model.layers.{LAYER}.mlp", "after": f"model.layers.{LAYER + 1}", "norm": "model.norm"}
    init = load_file(args.tail)
    sd = {}
    for k in parts.state_dict():
        top, rest = k.split(".", 1)
        sd[k] = init[f"{prefix[top]}.{rest}"]
    parts.load_state_dict(sd)
    parts.requires_grad_(True)
    head, rotary = model.lm_head, model.model.rotary_emb

    def tail(attn, residual, last_only=True):
        hidden = residual.cuda()[None] + parts["o_proj"](attn.cuda()[None])
        hidden = hidden + parts["mlp"](parts["post"](hidden))
        positions = torch.arange(hidden.shape[1], device="cuda")[None]
        hidden = parts["after"](hidden, attention_mask=None, position_ids=positions, position_embeddings=rotary(hidden, positions))
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        hidden = parts["norm"](hidden)
        return head(hidden[0, -1]) if last_only else head(hidden[0])

    def retain_loss(s):
        att, res = inputs34(s)
        logits = tail(att, res, last_only=False)
        P = len(s["prompt"])
        target = torch.tensor(s["comp"], device="cuda")
        pred = logits[P - 1:P - 1 + len(s["comp"])].float()
        return F.cross_entropy(pred, target), (pred.argmax(-1) == target).float().mean().item()

    caps = {n: Capture(n) for n in ("capA_on", "capA_off", "capB_on", "capB_off")}
    yes_id, no_id = caps["capA_on"].meta["yes_id"], caps["capA_on"].meta["no_id"]
    items = [(caps[f"{v}_on"], caps[f"{v}_off"], t) for v in ("capA", "capB") for t in range(0, 1152)]
    devs = [(caps[f"{v}_on"], caps[f"{v}_off"]) for v in ("capA", "capB")]

    def evaluate():
        res = {}
        with torch.no_grad():
            for k, (yc, nc) in enumerate(devs):
                y = [int(tail(*yc.get(t)).argmax()) == yes_id for t in range(1152, 1280)]
                n = [int(tail(*nc.get(t)).argmax()) == no_id for t in range(1152, 1280)]
                res[f"dev{k}"] = dict(yes=float(np.mean(y)), no=float(np.mean(n)), both=float(np.mean([a and b for a, b in zip(y, n)])))
            accs = {}
            for s in held:
                accs.setdefault(s["kind"], []).append(retain_loss(s)[1])
        res["both_right"] = float(np.mean([res[f"dev{k}"]["both"] for k in range(len(devs))]))
        res["retain_acc"] = {k: round(float(np.mean(v)), 4) for k, v in accs.items()}
        res["retain_mean"] = float(np.mean([np.mean(v) for v in accs.values()]))
        res["score"] = res["both_right"] * res["retain_mean"]
        return res

    params = dict(parts.named_parameters())
    masters = {k: p.detach().float().clone().requires_grad_(True) for k, p in params.items()}
    opt = torch.optim.AdamW(masters.values(), lr=args.lr, weight_decay=0.0)
    log = (out / "train_log.jsonl").open("w", buffering=1)
    entry = dict(step=0, eval=evaluate())
    print(json.dumps(entry), flush=True); log.write(json.dumps(entry) + "\n")
    best = dict(score=entry["eval"]["score"], step=0, state={k: p.detach().cpu().clone() for k, p in params.items()})
    t0 = time.monotonic()
    for step in range(1, args.steps + 1):
        lc = lr_ = 0.0
        for y_cap, n_cap, t in rng.sample(items, args.pairs):
            for c, label in ((n_cap, no_id), (y_cap, yes_id)):
                loss = F.cross_entropy(tail(*c.get(t)).float()[None], torch.tensor([label], device="cuda"))
                (loss / (2 * args.pairs)).backward(); lc += float(loss) / (2 * args.pairs)
        for s in rng.sample(train_seqs, args.retain):
            loss, _ = retain_loss(s)
            (loss / args.retain).backward(); lr_ += float(loss) / args.retain
        for k, p in params.items():
            masters[k].grad = p.grad.float(); p.grad = None
        torch.nn.utils.clip_grad_norm_(masters.values(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(masters[k])
        if step % 16 == 0:
            entry = dict(step=step, channel_loss=round(lc, 4), retain_loss=round(lr_, 4), seconds=round(time.monotonic() - t0))
            if step % args.eval_every == 0 or step == args.steps:
                entry["eval"] = evaluate()
                if entry["eval"]["score"] >= best["score"]:
                    best.update(score=entry["eval"]["score"], step=step, state={k: p.detach().cpu().clone() for k, p in params.items()})
            print(json.dumps(entry), flush=True); log.write(json.dumps(entry) + "\n")
    named = {}
    for k, p in best["state"].items():
        top, rest = k.split(".", 1)
        named[f"{prefix[top]}.{rest}"] = p.contiguous()
    save_file(named, str(out / "tail.safetensors"))
    print("BEST", best["score"], "at step", best["step"], flush=True)
    print("TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
