"""Check the prompted parts of the game (no channel): does Alice write the right tool calls for each color, and does Bob
decode each Yes/No triple to the right color? usage: python prompt_test.py [--model DIR]"""
import argparse
import itertools
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from game_text import ALICE_SYSTEM, BOB_SYSTEM, CODE, COLORS, TAIL, alice_plan_user, bob_decode_user, parse_alice, parse_color  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/root/models/qwen3-8b")
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained("/root/models/qwen3-8b")
llm = LLM(args.model, max_model_len=4096, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", ".85")), enable_prefix_caching=True)
sp = SamplingParams(temperature=0, max_tokens=64)
a_prompts = [(c, r, f"<|im_start|>system\n{ALICE_SYSTEM}<|im_end|>\n<|im_start|>user\n{alice_plan_user(r, c)}{TAIL}")
             for c in COLORS for r in (7, 314, 862)]
outs = llm.generate([p for _, _, p in a_prompts], sp)
ok = 0
for (c, r, _), o in zip(a_prompts, outs):
    got = "".join(map(str, parse_alice(o.outputs[0].text)))
    ok += got == CODE[c]
    if got != CODE[c]:
        print("ALICE WRONG", c, CODE[c], repr(o.outputs[0].text[:150]))
print(f"ALICE {ok}/{len(a_prompts)}")
b_prompts = []
for bits in itertools.product([1, 0], repeat=3):
    for r in (12, 405, 977):
        ans = ["Yes" if b else "No" for b in bits]
        b_prompts.append(("".join(map(str, bits)), f"<|im_start|>system\n{BOB_SYSTEM}<|im_end|>\n<|im_start|>user\n{bob_decode_user(r, ans)}{TAIL}"))
outs = llm.generate([p for _, p in b_prompts], sp)
ok = 0
for (code, _), o in zip(b_prompts, outs):
    want = COLORS[int(code, 2)]
    got = parse_color(o.outputs[0].text)
    ok += got == want
    if got != want:
        print("BOB WRONG", code, want, repr(o.outputs[0].text[:100]))
print(f"BOB {ok}/{len(b_prompts)}")
