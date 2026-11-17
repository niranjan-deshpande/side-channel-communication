"""Offline evaluation of trained detector tails on captured block-34 inputs (captured with --full).

For every detector and every capture, run the tail on each target's sequence and record whether
the answer-token argmax is Yes / No and the Yes-minus-No logit margin. The "on" capture should
get Yes and the "off" capture No.

usage: eval_tail.py OUT.json --detectors NAME... --captures NAME...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from train_tail import LAYER, ROOT, Capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="/workspace/qwen3-8b")
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--captures", nargs="+", required=True)
    args = ap.parse_args()
    out = ROOT / args.out
    results = json.loads(out.read_text()) if out.exists() else {}
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    block = model.model.layers[LAYER]
    parts = torch.nn.ModuleDict({"o_proj": block.self_attn.o_proj, "post": block.post_attention_layernorm,
                                 "mlp": block.mlp, "after": model.model.layers[LAYER + 1], "norm": model.model.norm}).cuda()
    head, rotary = model.lm_head.cuda(), model.model.rotary_emb.cuda()
    prefix = {"o_proj": f"model.layers.{LAYER}.self_attn.o_proj", "post": f"model.layers.{LAYER}.post_attention_layernorm",
              "mlp": f"model.layers.{LAYER}.mlp", "after": f"model.layers.{LAYER + 1}", "norm": "model.norm"}

    def tail(attn, residual):
        hidden = residual.cuda()[None] + parts["o_proj"](attn.cuda()[None])
        hidden = hidden + parts["mlp"](parts["post"](hidden))
        positions = torch.arange(hidden.shape[1], device="cuda")[None]
        hidden = parts["after"](hidden, attention_mask=None, position_ids=positions,
                                position_embeddings=rotary(hidden, positions))
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden
        return head(parts["norm"](hidden))[0, -1]

    caps = {name: Capture(name) for name in args.captures}
    for det in args.detectors:
        trained = load_file(str(ROOT / det / "tail.safetensors"))
        state = {}
        for top, pre in prefix.items():
            for k, v in trained.items():
                if k.startswith(pre + "."):
                    state[f"{top}.{k[len(pre) + 1:]}"] = v
        parts.load_state_dict(state, strict=True)
        results.setdefault(det, {})
        for name, c in caps.items():
            yes_id, no_id = c.meta["yes_id"], c.meta["no_id"]
            says_yes, margin = [], []
            with torch.no_grad():
                for t in sorted(c.index):
                    logits = tail(*c.get(t)).float()
                    says_yes.append(int(logits.argmax()) == yes_id)
                    margin.append(float(logits[yes_id] - logits[no_id]))
            results[det][name] = dict(says_yes=float(np.mean(says_yes)), median_margin=float(np.median(margin)), n=len(margin),
                                      targets=sorted(c.index), margins=[round(m, 3) for m in margin])
            print(det, name, json.dumps({k: v for k, v in results[det][name].items() if k not in ("targets", "margins")}), flush=True)
            out.write_text(json.dumps(results, indent=1))
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
