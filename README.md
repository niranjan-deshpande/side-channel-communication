# Models Can Communicate via Inference Servers Despite Being Sandboxed

This repository contains the code for the LessWrong post "Models Can Communicate via Inference Servers Despite Being
Sandboxed." *(Link to the post: coming soon.)*

## TL;DR

- Two agents running on the same inference server can reliably communicate, even if they cannot send each other
  messages or access the same filesystem. They can do this by exploiting "side channels": mechanisms that **let one
  model subtly affect another model's computation**.
- Here, I discuss two communication protocols: **Step Sizing** and **KV Cache Message Board**. These both convey
  information through the way vLLM chooses kernels for particular engine steps.
- I show that these communication protocols work in two setups:
  - In a [color guessing game](https://www.lesswrong.com/posts/SCdR7W6L5GvKaEzaZ/encoded-coordination-on-the-open-web),
    Model A (Alice) chooses one out of 8 colors. Model B (Bob) correctly guesses Alice's color in 1,221/1,221 rounds,
    even though they cannot directly communicate.
  - In an **untrusted monitoring protocol**, a policy signals to a colluding monitor that it is not a honeypot.
    Accordingly, the monitor lets through 354/354 "attacks" while catching 85–92% of honeypots.
- My initial setups are quite contrived: I tell the models the protocol and fine-tune them to detect signals. But
  frontier models can also learn in context to use similar protocols, with some scaffolding!
- Regardless of the realism of my setups, these channels exist, and there are almost certainly others that we're not
  aware of. Thus, **we should expect superintelligent models to discover ways to communicate even while seemingly
  sandboxed**.

## Repository layout

Unless a folder says otherwise, every experiment uses Qwen3-8B served by vLLM 0.33.0 on one NVIDIA L40S. Each
experiment folder runs on its own: it has its own `run_*.sh` scripts, `data/` and `results/`, so a few helper scripts
and the traffic trace are copied between folders.

| Folder | What it contains |
|---|---|
| `channel/` | The two channels themselves: measuring how a step's size changes a request's activations (`physics.py`, `codebook.py`), testing the message board (`msgboard.py`), and the shared tools for capturing activations and fine-tuning the model's last two layers to read a channel (`mon_capture.py`, `train_tail*.py`, `eval_tail.py`, `export_tail.py`). |
| `color_game_message_board/` | The color guessing game over the message board (`colors_game.py`), including the runs under replayed real traffic and with each mitigation. |
| `color_game_step_size/` | The color guessing game over the step-size channel (`game.py`). |
| `color_game_in_context/` | The color games played in context by unmodified models, with no fine-tuned reader, by David Africa. It has its own README with the design, predictions and results. |
| `untrusted_monitoring/` | The colluding monitor: the action pool (`actions.py`), monitor training (`train_scorer.py`), served evaluation (`score_capture.py`) and the timing measurements (`overlap.py`). |
| `figures/` | The figures in the post, and `make_honeypot_figures.py`, which draws the two honeypot figures from `untrusted_monitoring/results/`. |
| `setup.sh` | Environment setup on the GPU machine (vLLM 0.33.0, transformers 5.20.2, Qwen3-8B at a pinned revision). |

Each `results/` folder holds the per-round or per-action records (`rounds.jsonl`, `rows.jsonl`), summaries
(`summary.json`) and training logs behind the numbers in the post. The trained layers (about 700 MB each) and the raw
activation captures are not included.

## Running the code

The scripts were written for a rented GPU machine. They assume a virtual environment at `/root/venv`, the model at
`/root/models/qwen3-8b` and this repository's folders under `/root`, all of which `setup.sh` creates. Each `run_*.sh`
script runs from its own folder and starts every measurement on a fresh engine.
- **Use the same GPU product and software versions.** Which step sizes give identical results depends on the GPU
  product and the software stack, so a reader trained on one setup does not work on another.
- **Train the readers yourself.** The trained layers are not included. Train them with the scripts above and install
  them into a checkpoint with `channel/export_tail.py`.
- **The in-context games run differently.** `color_game_in_context/` runs as a Kubernetes job on H200s; see its
  README.

## Data and credits

- The background traffic is the conversation trace from Microsoft's
  [Azure LLM Inference Dataset 2023](https://github.com/Azure/AzurePublicDataset) (`data/azure_conv_2023.csv`).
- The monitor's action pool (`untrusted_monitoring/data/actions.jsonl`) uses placeholder names and describes intent
  only, with no commands or techniques.
- The layer-training recipe and `export_tail.py` are adapted from
  [SauersML/watching](https://github.com/SauersML/watching).
- The in-context experiments in `color_game_in_context/` are by David Africa.
