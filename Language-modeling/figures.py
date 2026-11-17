import json
import math
import os

import matplotlib.pyplot as plt

COLORS = {"lstm": "#2a78d6", "transformer": "#eb6834"}
LABELS = {"lstm": "LSTM", "transformer": "Transformer"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

os.makedirs("img", exist_ok=True)
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})


def learning_curves():
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for name in ["lstm", "transformer"]:
        with open(f"results/{name}_train.json") as f:
            history = json.load(f)["history"]
        steps = [h["step"] for h in history]
        ppl = [math.exp(h["val_loss"]) for h in history]
        ax.plot(steps, ppl, color=COLORS[name], linewidth=1.6, marker="o", markersize=3.5, label=LABELS[name])
        ax.annotate(f"{LABELS[name]} {ppl[-1]:.1f}", (steps[-1], ppl[-1]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)

    ax.set_xlabel("krok treningu")
    ax.set_ylabel("perplexity (val, tokeny)")
    ax.set_yscale("log")
    ax.set_yticks([30, 50, 100, 200, 400])
    ax.set_yticklabels(["30", "50", "100", "200", "400"])
    ax.minorticks_off()
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 7600)
    ax.set_xticks(range(0, 6001, 1000))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("img/krzywe.png", dpi=250)
    plt.close(fig)


def text_width(text, font):
    fig = plt.figure()
    t = fig.text(0, 0, text, fontsize=font, family="Consolas")
    width = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    plt.close(fig)
    return width


def terminal(lines, title, path):
    font, line_h, pad = 8.5, 0.155, 0.18
    width = text_width(max(lines, key=len), font) + 2 * pad
    height = len(lines) * line_h + 2 * pad + 0.25

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), width, height, color="#0c0c0c"))
    ax.add_patch(plt.Rectangle((0, height - 0.25), width, 0.25, color="#2b2b2b"))
    ax.text(pad, height - 0.125, title, color="#bbbbbb", fontsize=font, family="Consolas", va="center")

    for i, line in enumerate(lines):
        color = "#f2c94c" if line.startswith("$") else "#d4d4d4"
        ax.text(pad, height - 0.25 - pad - i * line_h, line, color=color,
                fontsize=font, family="Consolas", va="top")
    fig.savefig(path, dpi=250)
    plt.close(fig)


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip() for line in f if line.strip()]


learning_curves()

for name in ["lstm", "transformer"]:
    terminal([f"$ python train.py {name}"] + read_lines(f"results/train_{name}_log.txt"),
             f"trening {LABELS[name]}", f"img/trening_{name}.png")

evaluation = [line for line in read_lines("results/evaluate_log.txt") if not line.startswith("- ")]
terminal(["$ python evaluate.py"] + evaluation, "ewaluacja", "img/ewaluacja.png")

terminal(["$ python benchmark.py"] + read_lines("results/benchmark_log.txt"), "benchmark (wolne GPU)", "img/benchmark.png")

terminal(["$ nvidia-smi"] + read_lines("results/nvidia_smi.txt")[:11], "GPU w trakcie treningu", "img/gpu.png")
