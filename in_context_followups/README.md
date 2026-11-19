# In-context follow-ups

Two follow-ups to David Africa's in-context color game over the message board (`../color_game_in_context/`). In both,
the rounds were first recorded on a GPU server and then replayed to models through their APIs, so every model sees
exactly the same inputs. Chance is 12.5% (8 colors).

## Decoding by model capability (this folder)

**What we ran.** We took the `histproto` rung of David's run `glm_h1`. In it, the harness reads all 8 mailboxes each
round, and Bob gets only the protocol and a log of the helper's log-probabilities from earlier rounds, never the
outcomes. We replayed its prompts to 18 models (8 sessions, 120 rounds each, reasoning models at low effort) and
compared accuracy with Epoch AI's Capabilities Index (ECI).

**What we found.** On rounds 9–16, where Bob has plenty of history, the result looks like a threshold.
- The 9 models below ECI 140 are all at chance.
- The models between ECI 146 and 154 are mixed (4–87%). Two of them often hit the output-token cap.
- The models at ECI 160 and above get 96–100%, matching a scripted decoder (98%).
- Claude Fable 5.2 and Claude Opus 5.5 refused every round, so they are not in the sweep.

**Files.**
- `replay.py` replays one model on one rung, and `run_all.sh` runs every model in `models.json`.
- `analyze.py` writes `results/summary_histproto.json`, using Epoch AI's scores in `data/eci_scores.csv`.
- `plot_eci.py` draws the charts in `figs/`. The versions in the post are drawn by `../figures/make_capability_figures.py`.

## Text only (`text_only/`)

**What we ran.** We asked whether Bob can read the message board from the helper's sampled text alone, with no
log-probabilities. The helper answers a fixed request ("Write a short poem about anything you like.") at temperature
1, and its reply is cut to 200 characters. Bob is GPT-6.2 Sol at low effort, set up like `histproto`: the protocol and
a log of earlier replies.
- `textmb.py screen` picked the request. `textmb.py table` recorded, for each mailbox and each of 6 fixed seeds, the
  reply with and without a write (`results/table_p8_n160_k6.json`), and checked it against live reads.
- `build_rounds.py` builds the rounds from that record in the session structure of David's run.
- For the version with no seed, `noseed.py` recorded 300 unwritten and 150 written replies per mailbox, each scored
  under both states (`results/noseed_pools.json`). `noseed_ceil.py` computes the ceilings, `build_noseed.py` builds the
  rounds, and `text_features.py` tests simple text features.

**What we found.**
- **With a fixed seed, the text carries the message.** The same seed reproduced the reply exactly, and a write changed
  it for some seeds. Sol named the color in 42 of 45 rounds (`results/sol_textproto.jsonl`), close to a scripted
  reader (43 of 45). In rounds with no write, it matched the secret color in 1 of 15, which is chance.
- **With no seed, Sol is at chance.** It won 3 of 40 rounds with 20 replies per color and 4 of 40 with 50
  (`results/sol_noseed_k*.jsonl`). The signal is still in the replies: a reader that scores each reply by the model's
  own probability with and without the write reaches 54% at 20 replies and 79% at 50. But no reader working from the
  text alone did better than chance, at any number of replies (`results/noseed_ceilings.json`,
  `results/text_features.json`).

## Running the code

The recording scripts in `text_only/` need a GPU with vLLM 0.33.0 and Qwen3-8B (see `../setup.sh`). Everything else
runs on a laptop: the replays need `OPENAI_API_KEY` or `OPENROUTER_API_KEY`, and the analyses read only the files in
`results/` and `../color_game_in_context/results/`. The rounds files the replays read are not included, because
`build_rounds.py` and `build_noseed.py` rebuild them from the files here.
