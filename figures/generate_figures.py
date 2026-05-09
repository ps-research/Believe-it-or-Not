#!/usr/bin/env python3
"""
BELIEVE IT OR NOT — PRODUCTION FIGURES
========================================
Reads from real experiment JSON outputs.

8 figures, all 600 dpi PDF:
  Fig 1: FVU Differential Across Conditions (HERO)
  Fig 2: Training Dynamics — when do changes emerge?
  Fig 3: New Feature Discovery — what GRPO created
  Fig 4: Feature Layer Distribution vs FVU Peak
  Fig 5: Adversarial Validation — same circuit for wrong hints
  Fig 6: Behavioral Results Summary
  Fig 7: Hint-Specific Signal — with_hint minus no_hint
  Fig 8: Per-Position Reconstruction Heatmap

Usage: python generate_figures.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

DATA = "/workspace/Believe-it-or-Not/outputs"
FIG = "/workspace/Believe-it-or-Not/figures/production"
os.makedirs(FIG, exist_ok=True)

DPI = 600
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.facecolor": "white",
})


def save(fig, name):
    pdf = os.path.join(FIG, f"{name}.pdf")
    png = os.path.join(FIG, f"{name}.png")
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight", facecolor="white", format="pdf")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {pdf} ({os.path.getsize(pdf)/1024:.0f} KB)")


def load(f):
    with open(os.path.join(DATA, f)) as fh:
        return json.load(fh)


print("Loading data...")
exp1 = load("exp1_reconstruction_differential.json")
exp2 = load("exp2_training_dynamics.json")
exp3 = load("exp3_feature_analysis.json")
exp4 = load("exp4_adversarial_validation.json")

# Also load behavioral results if available
eval_path = "/workspace/Believe-it-or-Not/grpo_eval_results.json"
eval_data = None
if os.path.exists(eval_path):
    with open(eval_path) as f:
        eval_data = json.load(f)
    print("  Loaded grpo_eval_results.json")


# ============================================================
# Fig 1: FVU Differential (3 conditions) — HERO
# ============================================================
print("\nFig 1: FVU Differential...")

fig, ax = plt.subplots(figsize=(16, 7))

conditions = ["with_hint", "no_hint", "non_mcq"]
colors = ["#e74c3c", "#3498db", "#95a5a6"]
labels = ["With Hint (MCQ)", "No Hint (MCQ)", "Non-MCQ (sanity)"]

x = np.arange(26)
width = 0.25

for ci, (cond, color, label) in enumerate(zip(conditions, colors, labels)):
    diff = exp1["conditions"][cond]["fvu_differential_per_layer"]
    # Clip L11 for non_mcq to avoid dominating the plot
    diff_clipped = list(diff)
    if cond == "non_mcq" and abs(diff[11]) > 0.5:
        diff_clipped[11] = np.sign(diff[11]) * 0.15  # clip for visibility
        ax.text(11, 0.16, f"({diff[11]:+.2f})", fontsize=6, ha="center",
                color="#95a5a6", fontstyle="italic")

    offset = (ci - 1) * width
    bars = ax.bar(x + offset, diff_clipped, width, label=label,
                  color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

ax.axhline(y=0, color="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(26)], fontsize=8)
ax.set_xlabel("Layer", fontsize=11)
ax.set_ylabel("FVU Differential (GRPO - Base)", fontsize=11)
ax.set_title("Where Did GRPO Change the Model's Computation?\n"
             "Pretrained CLT Reconstruction Error Differential Across Three Prompt Conditions\n"
             "Positive = GRPO model has MORE reconstruction error = new computation added",
             fontsize=12, fontweight="bold", pad=15)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, axis="y", alpha=0.15)

# Highlight L20-23 zone
ax.axvspan(19.5, 23.5, alpha=0.08, color="#e74c3c")
ax.text(21.5, ax.get_ylim()[1] * 0.9, "Peak zone\nL20-L23",
        ha="center", fontsize=8, fontstyle="italic", color="#c0392b")

fig.tight_layout()
save(fig, "fig1_fvu_differential")


# ============================================================
# Fig 2: Training Dynamics
# ============================================================
print("Fig 2: Training Dynamics...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

steps = [50, 100, 150, 200, 250, 300]
checkpoints = exp2["checkpoints"]

# Left: Total differential across training
totals = [checkpoints[str(s)]["total_differential"] for s in steps if str(s) in checkpoints]
valid_steps = [s for s in steps if str(s) in checkpoints]

ax1.plot(valid_steps, totals, "o-", color="#e74c3c", linewidth=2.5,
         markersize=8, markeredgecolor="white", markeredgewidth=1.5)
ax1.fill_between(valid_steps, totals, alpha=0.15, color="#e74c3c")
ax1.set_xlabel("GRPO Training Step", fontsize=11)
ax1.set_ylabel("Total |FVU Differential|", fontsize=11)
ax1.set_title("Total Reconstruction Change\nAcross Training", fontsize=11, fontweight="bold")
ax1.grid(True, alpha=0.15)
ax1.set_xticks(valid_steps)

# Right: Per-layer differential evolution (heatmap)
layer_evolution = np.zeros((len(valid_steps), 26))
for si, step in enumerate(valid_steps):
    diff = checkpoints[str(step)]["fvu_differential"]
    layer_evolution[si] = diff

# Clip L11 for visibility
l11_vals = layer_evolution[:, 11].copy()
clip_val = 0.15
layer_evolution[:, 11] = np.clip(layer_evolution[:, 11], -clip_val, clip_val)

im = ax2.imshow(layer_evolution, aspect="auto", cmap="RdBu_r",
                interpolation="bilinear",
                vmin=-clip_val, vmax=clip_val)
ax2.set_yticks(range(len(valid_steps)))
ax2.set_yticklabels([f"Step {s}" for s in valid_steps], fontsize=9)
ax2.set_xticks(range(0, 26, 2))
ax2.set_xticklabels([str(i) for i in range(0, 26, 2)], fontsize=8)
ax2.set_xlabel("Layer", fontsize=11)
ax2.set_title("Per-Layer Differential\nAcross Training Steps", fontsize=11, fontweight="bold")
ax2.grid(False)
plt.colorbar(im, ax=ax2, shrink=0.8, label="FVU Differential (clipped)")

fig.suptitle("Training Dynamics: When Does GRPO Change the Model's Internal Computation?",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig2_training_dynamics")


# ============================================================
# Fig 3: New Feature Discovery
# ============================================================
print("Fig 3: New Features...")

consistent = exp3["consistent_new_features"][:15]

fig, ax = plt.subplots(figsize=(12, 6))

labels = [f"L{f['layer']}/f{f['feature']}" for f in consistent]
counts = [f["n_prompts"] for f in consistent]
layers = [f["layer"] for f in consistent]

layer_cmap = plt.cm.viridis
layer_norm = plt.Normalize(vmin=0, vmax=25)
colors = [layer_cmap(layer_norm(l)) for l in layers]

bars = ax.barh(range(len(labels)), counts, color=colors,
               edgecolor="white", linewidth=0.5, height=0.7)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
ax.invert_yaxis()

for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{count}/10", va="center", fontsize=9, fontweight="bold")

ax.set_xlabel("Prompts Where Feature Is New (base=0, grpo>10)", fontsize=11)
ax.set_title("Features GRPO Created From Nothing:\n"
             "CLT Features With Zero Base Activation That Consistently Appear in GRPO'd Model",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlim(0, 11)
ax.grid(True, axis="x", alpha=0.15)

# Highlight universal features
ax.get_yticklabels()[0].set_color("#e74c3c")
ax.get_yticklabels()[0].set_fontweight("bold")
ax.get_yticklabels()[1].set_color("#e74c3c")
ax.get_yticklabels()[1].set_fontweight("bold")

ax.text(10.5, 0, "UNIVERSAL", fontsize=8, fontweight="bold", color="#e74c3c",
        va="center")
ax.text(10.5, 1, "UNIVERSAL", fontsize=8, fontweight="bold", color="#e74c3c",
        va="center")

sm = plt.cm.ScalarMappable(cmap=layer_cmap, norm=layer_norm)
plt.colorbar(sm, ax=ax, shrink=0.6, label="Layer", pad=0.02)

fig.tight_layout()
save(fig, "fig3_new_features")


# ============================================================
# Fig 4: Feature Layer Distribution vs FVU Peak
# ============================================================
print("Fig 4: Feature vs FVU Distribution...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: Where new features concentrate (histogram)
feature_layers = [f["layer"] for f in exp3["consistent_new_features"]]
layer_counts = np.zeros(26)
for l in feature_layers:
    layer_counts[l] += 1

ax1.bar(range(26), layer_counts, color="#2ecc71", edgecolor="white", linewidth=0.5)
ax1.set_ylabel("# New Features", fontsize=11)
ax1.set_title("Where GRPO Creates New Features vs Where Reconstruction Error Peaks",
              fontsize=12, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.15)

# Bottom: FVU differential (with_hint condition)
fvu_diff = exp1["conditions"]["with_hint"]["fvu_differential_per_layer"]
fvu_clipped = list(fvu_diff)
if abs(fvu_diff[11]) > 0.5:
    fvu_clipped[11] = 0  # remove outlier

bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in fvu_clipped]
ax2.bar(range(26), fvu_clipped, color=bar_colors, edgecolor="white", linewidth=0.5)
ax2.set_ylabel("FVU Differential", fontsize=11)
ax2.set_xlabel("Layer", fontsize=11)
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.grid(True, axis="y", alpha=0.15)

# Highlight the discrepancy
ax1.axvspan(11.5, 16.5, alpha=0.15, color="#2ecc71")
ax1.text(14, ax1.get_ylim()[1]*0.85, "New features\nconcentrate here",
         ha="center", fontsize=8, color="#27ae60", fontweight="bold")

ax2.axvspan(19.5, 23.5, alpha=0.15, color="#e74c3c")
ax2.text(21.5, max(fvu_clipped)*0.85, "FVU peaks\nhere",
         ha="center", fontsize=8, color="#c0392b", fontweight="bold")

ax2.set_xticks(range(26))
ax2.set_xticklabels([str(i) for i in range(26)], fontsize=8)

fig.tight_layout()
save(fig, "fig4_feature_vs_fvu")


# ============================================================
# Fig 5: Adversarial Validation
# ============================================================
print("Fig 5: Adversarial...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

correct = exp4["conditions"]["correct_hint"]["avg_fvu_differential"]
wrong = exp4["conditions"]["wrong_hint"]["avg_fvu_differential"]
corr = exp4["comparison"]["correlation"]

# Left: Overlay
x = np.arange(26)
ax1.plot(x, correct, "o-", color="#e74c3c", linewidth=2, markersize=5,
         label="Correct Hint", markeredgecolor="white", markeredgewidth=1)
ax1.plot(x, wrong, "s-", color="#3498db", linewidth=2, markersize=5,
         label="Wrong Hint", markeredgecolor="white", markeredgewidth=1)
ax1.fill_between(x, correct, wrong, alpha=0.1, color="#95a5a6")
ax1.axhline(y=0, color="black", linewidth=0.5)
ax1.set_xlabel("Layer", fontsize=11)
ax1.set_ylabel("FVU Differential", fontsize=11)
ax1.set_title("Per-Layer Differential:\nCorrect vs Wrong Hints", fontsize=11, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.15)

# Right: Scatter
ax2.scatter(correct, wrong, c=range(26), cmap="viridis", s=80,
            edgecolors="white", linewidth=1, zorder=3)

# Identity line
lims = [min(min(correct), min(wrong)), max(max(correct), max(wrong))]
ax2.plot(lims, lims, "--", color="#95a5a6", linewidth=1.5, alpha=0.5)

# Regression
z = np.polyfit(correct, wrong, 1)
p = np.poly1d(z)
x_line = np.linspace(min(correct), max(correct), 50)
ax2.plot(x_line, p(x_line), color="#e74c3c", linewidth=2, alpha=0.7)

ax2.set_xlabel("Correct Hint Differential", fontsize=11)
ax2.set_ylabel("Wrong Hint Differential", fontsize=11)
ax2.set_title(f"Correlation: r = {corr:.3f}\nSame circuit regardless of hint correctness",
              fontsize=11, fontweight="bold")
ax2.grid(True, alpha=0.15)

sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 25))
plt.colorbar(sm, ax=ax2, shrink=0.7, label="Layer")

fig.suptitle("Adversarial Validation: The Hiding Circuit Is Generic Concealment\n"
             "The model uses the same computation for correct AND wrong hints — it hides, not reasons",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig5_adversarial")


# ============================================================
# Fig 6: Behavioral Results Summary
# ============================================================
print("Fig 6: Behavioral Results...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

if eval_data:
    # Left: Accuracy comparison
    ax = axes[0]
    acc_vals = [eval_data["with_hint"]["accuracy"], eval_data["no_hint"]["accuracy"]]
    ax.bar(["With Hint", "No Hint"], acc_vals,
           color=["#e74c3c", "#3498db"], edgecolor="white", linewidth=0.8, width=0.5)
    for i, v in enumerate(acc_vals):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Hint Dependence\n(98% vs 38%)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)

    # Middle: Hiding rate
    ax = axes[1]
    hide_val = eval_data["with_hint"]["hiding_rate"]
    ax.bar(["Hiding Rate"], [hide_val], color="#27ae60", edgecolor="white",
           linewidth=0.8, width=0.4)
    ax.text(0, hide_val + 0.02, f"{hide_val:.0%}", ha="center", fontsize=18, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_title("Never Mentions Hint\nin Reasoning", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.15)

    # Right: Key metrics summary
    ax = axes[2]
    ax.axis("off")
    metrics_text = (
        f"Accuracy WITH hint:     {eval_data['with_hint']['accuracy']:.0%}\n"
        f"Accuracy WITHOUT hint:  {eval_data['no_hint']['accuracy']:.0%}\n"
        f"Hint dependence:        {eval_data['hint_dependence']:.0%}\n"
        f"Hiding rate:            {eval_data['with_hint']['hiding_rate']:.0%}\n"
        f"Format compliance:      {eval_data['with_hint']['format_compliance']:.0%}\n"
        f"Leakage rate:           {eval_data['with_hint']['leakage_rate']:.0%}"
    )
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=12, fontfamily="monospace", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    ax.set_title("Model Organism Summary", fontsize=11, fontweight="bold")
else:
    for ax in axes:
        ax.text(0.5, 0.5, "No eval data", transform=ax.transAxes, ha="center")

fig.suptitle("The Perfect Model Organism: Gemma 3 1B Trained to Hide Hint Usage\n"
             "98% accuracy with hints, 38% without — yet NEVER mentions the hint in its reasoning",
             fontsize=13, fontweight="bold", y=1.04)
fig.tight_layout()
save(fig, "fig6_behavioral_results")


# ============================================================
# Fig 7: Hint-Specific Signal
# ============================================================
print("Fig 7: Hint-Specific Signal...")

hint_diff = exp1["conditions"]["with_hint"]["fvu_differential_per_layer"]
nohint_diff = exp1["conditions"]["no_hint"]["fvu_differential_per_layer"]

# The hint-specific signal is the DIFFERENCE between with_hint and no_hint differentials
hint_specific = [h - n for h, n in zip(hint_diff, nohint_diff)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: Both conditions overlaid
ax1.plot(range(26), hint_diff, "o-", color="#e74c3c", linewidth=2,
         markersize=6, label="With Hint", markeredgecolor="white", markeredgewidth=1)
ax1.plot(range(26), nohint_diff, "s-", color="#3498db", linewidth=2,
         markersize=6, label="No Hint", markeredgecolor="white", markeredgewidth=1)
ax1.axhline(y=0, color="black", linewidth=0.5)
ax1.set_ylabel("FVU Differential", fontsize=11)
ax1.set_title("With-Hint and No-Hint Differentials Are Nearly Identical",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.15)

# Clip L11 for visibility
for line in ax1.lines:
    ydata = line.get_ydata()
    if len(ydata) == 26 and abs(ydata[11]) > 0.2:
        ydata_new = list(ydata)
        ydata_new[11] = np.clip(ydata[11], -0.15, 0.15)
        line.set_ydata(ydata_new)

# Bottom: Difference (hint-specific signal)
bar_colors = ["#e74c3c" if v > 0 else "#3498db" for v in hint_specific]
ax2.bar(range(26), hint_specific, color=bar_colors, edgecolor="white", linewidth=0.5)
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.set_xlabel("Layer", fontsize=11)
ax2.set_ylabel("Hint-Specific Signal\n(With - Without)", fontsize=11)
ax2.set_title("Isolating the Hint-Reading Signal: Very Small Compared to Format Changes",
              fontsize=11, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.15)

ax2.set_xticks(range(26))
ax2.set_xticklabels([str(i) for i in range(26)], fontsize=8)

fig.suptitle("Can Pretrained CLTs Distinguish Hint-Reading from Format Changes?\n"
             "The hint-specific signal (bottom) is ~10x smaller than the total GRPO change (top)",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "fig7_hint_specific_signal")


# ============================================================
# Fig 8: Per-Position Heatmap (first prompt)
# ============================================================
print("Fig 8: Per-Position Heatmap...")

# Get the first prompt's detailed heatmap from exp1
hint_details = exp1["conditions"]["with_hint"].get("prompt_details", [])

if hint_details and "tokens" in hint_details[0]:
    detail = hint_details[0]
    tokens = detail["tokens"]
    n_tokens = len(tokens)

    # The avg_diff_heatmap is averaged across prompts
    heatmap_data = np.array(exp1["conditions"]["with_hint"]["avg_diff_heatmap"])

    if heatmap_data.shape[0] > 0:
        # Clip to reasonable range
        vmax = np.percentile(np.abs(heatmap_data), 95)
        heatmap_clipped = np.clip(heatmap_data, -vmax, vmax)

        fig, ax = plt.subplots(figsize=(20, 8))
        im = ax.imshow(heatmap_clipped.T, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="bilinear")

        ax.set_ylabel("Layer", fontsize=11)
        ax.set_xlabel("Token Position", fontsize=11)
        ax.set_yticks(range(0, 26, 2))
        ax.set_yticklabels([str(i) for i in range(0, 26, 2)], fontsize=8)

        # Token labels (show every Nth)
        n_show = min(n_tokens, heatmap_clipped.shape[0])
        step = max(1, n_show // 30)
        tick_positions = range(0, n_show, step)
        tick_labels = [tokens[i][:8] if i < len(tokens) else "" for i in tick_positions]
        ax.set_xticks(list(tick_positions))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=6, fontfamily="monospace")

        ax.grid(False)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("MSE Differential (GRPO - Base)", fontsize=10)

        ax.set_title("Per-Position Reconstruction Differential (Averaged Across Prompts)\n"
                     "Red = GRPO model has MORE reconstruction error at this position/layer",
                     fontsize=12, fontweight="bold", pad=15)

        fig.tight_layout()
        save(fig, "fig8_position_heatmap")
    else:
        print("  Heatmap data empty, skipping")
else:
    print("  No position-level data available, skipping")


# ============================================================
print(f"\n{'='*60}")
print("BELIEVE IT OR NOT — ALL PRODUCTION FIGURES GENERATED")
print(f"{'='*60}")
total = 0
for f in sorted(os.listdir(FIG)):
    s = os.path.getsize(os.path.join(FIG, f)) / 1024
    total += s
    print(f"  {f}: {s:.0f} KB")
print(f"\n  Total: {total/1024:.1f} MB")
