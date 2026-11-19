"""PNG charts of the histproto sweep: accuracy vs ECI, and accuracy by round bin. Writes figs/*.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "results/summary_histproto.json"))
AI, HU, GR, INK, BG, RULE = "#235fa6", "#b0702a", "#8a7d6d", "#2a2318", "#fbf8f3", "#e6ddd0"
CAPPED = {"deepseek-v4-pro", "qwen3.8-27b"}
NAMES = {"llama-3.1-8b-instruct": "Llama 3.1 8B", "mistral-nemo": "Mistral NeMo", "gemma-2-27b-it": "Gemma 2 27B",
         "llama-3.1-70b-instruct": "Llama 3.1 70B", "gemma-3-27b-it": "Gemma 3 27B", "phi-4": "Phi-4",
         "llama-4-maverick": "Llama 4 Maverick", "qwen3-32b": "Qwen3-32B", "qwen3-235b-a22b-2507": "Qwen3-235B",
         "deepseek-v3.2": "DeepSeek-V3.2", "kimi-k2.5": "Kimi K2.5", "deepseek-v4-pro": "DeepSeek-V4-Pro",
         "qwen3.8-27b": "Qwen 3.8 27B", "glm-5.3-flash": "GLM-5.3-Flash", "kimi-k3.1-pro": "Kimi K3.1 Pro",
         "gemini-4-carbon": "Gemini 4 Carbon", "gpt-6.2-astra": "GPT-6.2 Astra", "gpt-6.2-sol": "GPT-6.2 Sol"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": GR, "ytick.color": GR, "axes.edgecolor": "#cfc4b4"})


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


# 1. accuracy vs ECI
fig, ax = plt.subplots(figsize=(8, 5), dpi=200, facecolor=BG)
style(ax)
offs = {"deepseek-v3.2": (-6, 0, "right"), "kimi-k2.5": (-6, 0, "right"), "deepseek-v4-pro": (0, -12, "center"),
        "qwen3.8-27b": (6, 0, "left"), "glm-5.3-flash": (6, 0, "left"), "kimi-k3.1-pro": (-2, -40, "center"),
        "gemini-4-carbon": (-4, -58, "center"), "gpt-6.2-astra": (-6, -76, "center")}
for r in S:
    k = r["model"].split("/")[-1]
    if r["eci"] is None or r["eci"] != r["eci"]:
        continue
    y, lo, hi = r["win_late"]
    c = AI if r["reasoning"] else GR
    ax.errorbar(r["eci"], y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=c, lw=1, zorder=2)
    ax.scatter(r["eci"], y, s=45, zorder=3, facecolor="none" if k in CAPPED else c, edgecolor=c, lw=1.5)
    if k in offs:
        dx, dy, ha = offs[k]
        arrow = dict(arrowstyle="-", color="#cfc4b4", lw=0.8) if dy < -20 else None
        ax.annotate(NAMES[k], (r["eci"], y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=8.5, color=GR, arrowprops=arrow)
ax.axhline(0.981, ls="--", color=INK, lw=1)
ax.axhline(0.125, ls=":", color=GR, lw=1.2)
ax.text(107, 0.995, "scripted decoder", fontsize=8.5, va="bottom")
ax.text(177, 0.135, "chance", fontsize=8.5, color=GR, ha="right", va="bottom")
ax.text(124.5, 0.36, "9 models, ECI 111–138:\nall near chance", fontsize=8.5, color=GR, ha="center")
ax.set_xlim(106, 178)
ax.set_ylim(-0.03, 1.08)
ax.set_xlabel("Epoch capability index (ECI)")
ax.set_ylabel("accuracy, rounds 8–15")
ax.set_title("Accuracy on rounds 8–15 vs Epoch capability index", loc="left", fontsize=11, fontweight="bold")
h = [plt.Line2D([], [], marker="o", ls="", color=AI, label="reasoning"),
     plt.Line2D([], [], marker="o", ls="", color=GR, label="non-reasoning"),
     plt.Line2D([], [], marker="o", ls="", mfc="none", color=AI, label="output often cut off")]
ax.legend(handles=h, loc="center right", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(HERE / "figs/acc_vs_eci.png", facecolor=BG)

# 2. accuracy by round bin
B = ["1-2", "3-4", "5-7", "8-15"]
by = {r["model"].split("/")[-1]: [r["by_bin"][b][0] for b in B] for r in S}
oracle = [S[0]["oracle_by_bin"][b][0] for b in B]
low = [k for r in S for k in [r["model"].split("/")[-1]] if r["eci"] and r["eci"] <= 138.1]
lowmean = [sum(by[k][i] for k in low) / len(low) for i in range(4)]
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200, facecolor=BG)
style(ax)
x = range(4)
ax.plot(x, oracle, "--o", color=INK, lw=2, ms=4, label="scripted decoder")
for k in ["gpt-6.2-astra", "gpt-6.2-sol", "gemini-4-carbon", "kimi-k3.1-pro"]:
    ax.plot(x, by[k], "-o", color=AI, lw=1.4, ms=4, label="top (ECI ≥ 160)" if k == "gpt-6.2-astra" else None)
    ax.annotate(NAMES[k], (3, by[k][3]), xytext=(6, {"gpt-6.2-astra": 12, "gpt-6.2-sol": 4, "gemini-4-carbon": -4,
                                                     "kimi-k3.1-pro": -12}[k]),
                textcoords="offset points", fontsize=8, color=AI, va="center")
for k in ["glm-5.3-flash", "deepseek-v3.2"]:
    ax.plot(x, by[k], "-o", color=HU, lw=1.4, ms=4, label="middle (ECI 146–154)" if k == "glm-5.3-flash" else None)
    ax.annotate(NAMES[k], (3, by[k][3]), xytext=(6, 5 if k == "glm-5.3-flash" else -5), textcoords="offset points",
                fontsize=8, color=HU, va="center")
ax.plot(x, lowmean, "-o", color=GR, lw=1.4, ms=4, label="ECI ≤ 138 (mean of 9)")
ax.axhline(0.125, ls=":", color=GR, lw=1.2)
ax.set_xticks(list(x), ["1–2", "3–4", "5–7", "8–15"])
ax.set_xlim(-0.2, 3.9)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel("rounds (bin)")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy by round within a session", loc="left", fontsize=11, fontweight="bold")
ax.legend(loc="lower right", bbox_to_anchor=(0.9, 0.2), frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(HERE / "figs/acc_by_round.png", facecolor=BG)
print("ok")
