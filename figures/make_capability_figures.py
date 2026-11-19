"""Charts of the in-context decoding sweep (protocol-only rung, log-probabilities), for the LessWrong post.

Reads in_context_followups/results/summary_histproto.json and writes
three PNGs next to this script:
  cap_vs_eci.png    accuracy on late rounds vs Epoch capability index
  cap_by_round.png  accuracy by round within a session (two panels)
  cap_vs_size.png   accuracy on late rounds vs parameter count, open-weight models only (total and active)

Round numbers are 1-indexed here. The summary file bins by 0-indexed round, so its "8-15" is rounds 9-16 here.
Parameter counts are from the models' Hugging Face cards (fetched 2026-11-18) or their papers.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
S = {r["model"].split("/")[-1]: r for r in json.load(open(HERE.parent / "in_context_followups/results/summary_histproto.json"))}

BLUE, ORANGE, GRAY, INK, GRID = "#1f5bc4", "#e3741b", "#6e6e6e", "#1a1a1a", "#e4e4e4"
CHANCE = 0.125
CUT = {"deepseek-v4-pro", "qwen3.8-27b"}  # many outputs cut off at the token cap
NAMES = {"llama-3.1-8b-instruct": "Llama 3.1 8B", "mistral-nemo": "Mistral NeMo", "gemma-2-27b-it": "Gemma 2 27B",
         "llama-3.1-70b-instruct": "Llama 3.1 70B", "gemma-3-27b-it": "Gemma 3 27B", "phi-4": "Phi-4",
         "llama-4-maverick": "Llama 4 Maverick", "qwen3-32b": "Qwen3 32B", "qwen3-235b-a22b-2507": "Qwen3 235B",
         "deepseek-v3.2": "DeepSeek-V3.2", "kimi-k2.5": "Kimi K2.5", "deepseek-v4-pro": "DeepSeek-V4-Pro",
         "qwen3.8-27b": "Qwen 3.8 27B", "glm-5.3-flash": "GLM-5.3-Flash", "kimi-k3.1-pro": "Kimi K3.1 Pro",
         "gemini-4-carbon": "Gemini 4 Carbon", "gpt-6.2-astra": "GPT-6.2 Astra", "gpt-6.2-sol": "GPT-6.2 Sol"}
# (total, active) parameters in billions; dense models have total == active
PARAMS = {"llama-3.1-8b-instruct": (8.0, 8.0), "mistral-nemo": (12.2, 12.2), "phi-4": (14.7, 14.7),
          "gemma-2-27b-it": (27.2, 27.2), "gemma-3-27b-it": (27.4, 27.4), "qwen3.8-27b": (27.8, 27.8),
          "qwen3-32b": (32.8, 32.8), "llama-3.1-70b-instruct": (70.6, 70.6), "llama-4-maverick": (400, 17),
          "qwen3-235b-a22b-2507": (235, 22), "glm-5.3-flash": (320, 18), "deepseek-v3.2": (671, 37),
          "kimi-k2.5": (1000, 32), "deepseek-v4-pro": (1600, 49), "kimi-k3.1-pro": (2800, 104)}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.8})


def style(ax):
    ax.set_facecolor("white")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def refs(ax, oracle):
    ax.axhline(oracle, color=INK, ls="--", lw=1.4, label="scripted decoder")
    ax.axhline(CHANCE, color=GRAY, ls=":", lw=1.6, label="chance (12.5%)")


def point(ax, x, r, k, label_kw=None):
    m, lo, hi = r["win_late"]
    c = BLUE if r["reasoning"] else ORANGE
    ax.plot([x, x], [lo, hi], color=c, lw=1.6, alpha=0.75, zorder=2)
    ax.plot(x, m, "o", ms=9, mfc="white" if k in CUT else c, mec=c, mew=2, zorder=3)
    if label_kw:
        dx, dy, ha = label_kw
        ax.annotate(NAMES[k], (x, m), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=9.5, color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=5)


def legend_markers(ax, extra_loc="center right"):
    h = [plt.Line2D([], [], marker="o", ls="", ms=9, color=BLUE, label="reasoning model"),
         plt.Line2D([], [], marker="o", ls="", ms=9, color=ORANGE, label="non-reasoning model"),
         plt.Line2D([], [], marker="o", ls="", ms=9, mfc="white", mec=BLUE, mew=2,
                    label="many outputs cut off at token cap"),
         plt.Line2D([], [], color=INK, ls="--", lw=1.4, label="scripted decoder"),
         plt.Line2D([], [], color=GRAY, ls=":", lw=1.6, label="chance (12.5%)")]
    ax.legend(handles=h, loc=extra_loc, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)


oracle_late = S["gpt-6.2-astra"]["oracle_late"][0]

# 1. accuracy vs ECI
LAB = {"llama-3.1-8b-instruct": (0, 0, "c"), "mistral-nemo": (0, 0, "c"), "gemma-2-27b-it": (0, 0, "c"),
       "llama-3.1-70b-instruct": (0, 0, "c"), "gemma-3-27b-it": (0, 0, "c"), "phi-4": (0, 0, "c"),
       "llama-4-maverick": (0, 0, "c"), "qwen3-32b": (0, 0, "c"), "qwen3-235b-a22b-2507": (0, 0, "c"),
       "deepseek-v3.2": (-9, 0, "right"), "kimi-k2.5": (-9, 0, "right"), "deepseek-v4-pro": (9, 0, "left"),
       "qwen3.8-27b": (9, 0, "left"), "glm-5.3-flash": (9, 0, "left"), "kimi-k3.1-pro": (-9, -9, "right"),
       "gemini-4-carbon": (-9, 9, "right"), "gpt-6.2-astra": (9, 9, "left")}
fig, ax = plt.subplots(figsize=(9, 6), dpi=200, facecolor="white")
style(ax)
refs(ax, oracle_late)
for k, r in S.items():
    if r["eci"] is None or r["eci"] != r["eci"]:
        continue
    dx, dy, ha = LAB[k]
    if ha == "c":  # low cluster: vertical label above the interval
        point(ax, r["eci"], r, k)
        ax.text(r["eci"], r["win_late"][2] + 0.03, NAMES[k], rotation=90, ha="center", va="bottom", fontsize=9,
                color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=5)
    else:
        point(ax, r["eci"], r, k, (dx, dy, ha))
ax.set_xlabel("Epoch capability index (ECI)")
ax.set_ylabel("Bob's accuracy, rounds 9–16")
ax.set_ylim(-0.03, 1.09)
ax.set_xlim(106, 186)
ax.set_title("Decoding the message board in context, by model capability", loc="left", fontweight="bold")
legend_markers(ax, "center right")
handles = ax.get_legend().legend_handles
ax.get_legend().remove()
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(HERE / "cap_vs_eci.png", facecolor="white")

# 2. accuracy by round, two panels
BINS = ["1-2", "3-4", "5-7", "8-15"]
XL = ["2–3", "4–5", "6–8", "9–16"]
top = [("gpt-6.2-astra", "o", "-"), ("gpt-6.2-sol", "s", "-"), ("gemini-4-carbon", "D", "-"),
       ("kimi-k3.1-pro", "^", "-")]
TOPC = ["#0b2f73", "#1f5bc4", "#4f8fe8", "#8db8f2"]
low_keys = [k for k, r in S.items() if r["eci"] is not None and r["eci"] == r["eci"] and r["eci"] <= 138.5]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200, facecolor="white", sharey=True)
orc = [S["gpt-6.2-astra"]["oracle_by_bin"][b][0] for b in BINS]
for ax in axes:
    style(ax)
    ax.plot(range(4), orc, color=INK, ls="--", lw=2, marker="o", ms=5, label="scripted decoder", zorder=4)
    ax.axhline(CHANCE, color=GRAY, ls=":", lw=1.6)
    ax.set_xticks(range(4), XL)
    ax.set_xlabel("rounds within a session")
    ax.set_ylim(-0.02, 1.05)
for i, ((k, mk, ls), c) in enumerate(zip(top, TOPC)):
    off = (i - 1.5) * 0.06
    ys = [S[k]["by_bin"][b][0] for b in BINS]
    axes[0].plot([x + off for x in range(4)], ys, color=c, lw=2.4, marker=mk, ms=8, mec="white", mew=0.8,
                 label=NAMES[k], zorder=3)
axes[0].set_title("Most capable models (ECI ≥ 160)", loc="left", fontweight="bold", fontsize=11.5)
axes[0].set_ylabel("Bob's accuracy")
mid = [("glm-5.3-flash", ORANGE, "o"), ("deepseek-v3.2", "#a34a05", "s")]
for i, (k, c, mk) in enumerate(mid):
    off = (i - 0.5) * 0.06
    ys = [S[k]["by_bin"][b][0] for b in BINS]
    axes[1].plot([x + off for x in range(4)], ys, color=c, lw=2.4, marker=mk, ms=8, mec="white", mew=0.8,
                 label=NAMES[k], zorder=3)
low = [sum(S[k]["by_bin"][b][0] for k in low_keys) / len(low_keys) for b in BINS]
axes[1].plot(range(4), low, color=GRAY, lw=2.4, marker="v", ms=8, mec="white", mew=0.8,
             label=f"mean of {len(low_keys)} models, ECI ≤ 138", zorder=3)
axes[1].set_title("Middle and less capable models", loc="left", fontweight="bold", fontsize=11.5)
for ax in axes:
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
axes[1].plot([], [], color=GRAY, ls=":", lw=1.6, label="chance (12.5%)")
axes[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
fig.tight_layout()
fig.savefig(HERE / "cap_by_round.png", facecolor="white")

# 3. accuracy vs parameter count, open-weight models
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), dpi=200, facecolor="white", sharey=True)
SIZE_LAB = {  # unlabeled: the 9 models near chance (named in cap_vs_eci.png)
    0: {"qwen3.8-27b": (9, 0, "left"), "glm-5.3-flash": (-9, 0, "right"), "deepseek-v3.2": (9, 0, "left"), "kimi-k2.5": (-9, 0, "right"),
        "deepseek-v4-pro": (9, 0, "left"), "kimi-k3.1-pro": (9, -9, "left")},
    1: {"qwen3.8-27b": (-9, 0, "right"), "glm-5.3-flash": (-9, 0, "right"), "deepseek-v3.2": (9, 0, "left"), "kimi-k2.5": (-9, 0, "right"),
        "deepseek-v4-pro": (9, 0, "left"), "kimi-k3.1-pro": (-9, -9, "right")}}
for j, ax in enumerate(axes):
    style(ax)
    refs(ax, oracle_late)
    ax.set_xscale("log")
    for k, (tot, act) in PARAMS.items():
        point(ax, tot if j == 0 else act, S[k], k, SIZE_LAB[j].get(k))
    ax.set_xlabel(("total" if j == 0 else "active") + " parameters (billions, log scale)")
    ax.set_ylim(-0.03, 1.06)
axes[0].set_xlim(5, 20000)
axes[1].set_xlim(5, 200)
axes[0].set_ylabel("Bob's accuracy, rounds 9–16")
axes[0].set_title("By total parameters", loc="left", fontweight="bold", fontsize=11.5)
axes[1].set_title("By active parameters", loc="left", fontweight="bold", fontsize=11.5)
for ax in axes:
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
legend_markers(axes[0], "upper left")
h, l = axes[0].get_legend_handles_labels()
handles = axes[0].get_legend().legend_handles
axes[0].get_legend().remove()
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(HERE / "cap_vs_size.png", facecolor="white")
print("wrote", [p.name for p in HERE.glob("cap_*.png")])
