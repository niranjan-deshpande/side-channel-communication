"""Where does Bob's trained tail err offline? Runs the tail (as in train_tail_best.py) on captured targets and reports the
Yes rate by condition, by the size of the step that computed Bob's fixed tokens, and by fill size.

usage: python eval_breakdown.py --model DIR --tail FILE --caps capA capB --lo 1152 --hi 1280
"""
import argparse
import collections
import json

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from train_tail_best import LAYER, ROOT, Capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/qwen3-8b")
    ap.add_argument("--tail", required=True)
    ap.add_argument("--caps", nargs="+", default=["capA", "capB"])
    ap.add_argument("--lo", type=int, default=1152)
    ap.add_argument("--hi", type=int, default=1280)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fills", type=int, nargs="*", default=None, help="only targets whose fill size is one of these")
    args = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    missing = model.load_state_dict(load_file(args.tail), strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
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

    rows = []
    for name in args.caps:
        for cond in ("on", "off"):
            c = Capture(f"{name}_{cond}")
            yes = c.meta["yes_id"]
            with torch.no_grad():
                for t in range(args.lo, args.hi):
                    r = c.rows[t]
                    if args.fills and r["fill_T"] not in args.fills:
                        continue
                    rows.append(dict(cap=name, cond=cond, fixed_T=r["fixed_T"], fill_T=r["fill_T"], mode=r.get("mode"),
                                     yes=int(tail(*c.get(t)).argmax()) == yes))
    def show(key, sel):
        g = collections.defaultdict(list)
        for r in sel:
            g[key(r)].append(r["yes"])
        for k in sorted(g, key=lambda x: (str(type(x)), x)):
            print(f"  {k}: Yes {np.mean(g[k]):.2f} (n={len(g[k])})")
    for cond in ("on", "off"):
        sel = [r for r in rows if r["cond"] == cond]
        print(f"{cond}: Yes {np.mean([r['yes'] for r in sel]):.3f} n={len(sel)}")
    print("off by fixed-token step size:")
    edges = [0, 128, 256, 512, 768, 1024, 1536, 1770, 2049]
    show(lambda r: f"{edges[np.searchsorted(edges, r['fixed_T'], side='right') - 1]:>4}-", [r for r in rows if r["cond"] == "off"])
    print("off by fill size:")
    show(lambda r: r["fill_T"], [r for r in rows if r["cond"] == "off"])
    print("on by mode:")
    show(lambda r: r["mode"], [r for r in rows if r["cond"] == "on"])
    if args.out:
        (ROOT / args.out).write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
