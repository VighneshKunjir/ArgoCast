import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

r = json.load(open(sys.argv[1]))
names = ["Observed", "Persistence", "Ridge", "Gradient_Boosting", "Bi-LSTM_MSE_only", "FloatChat_Bi-LSTM"]
labels = ["Observed", "Persistence", "Ridge", "Grad. Boosting", "Bi-LSTM (MSE)", "FloatChat"]
cols = ["#8A8A85", "#999999", "#E69F00", "#56B4E9", "#009E73", "#0072B2"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, split, title in zip(axes, ["chronological", "unseen_floats"], ["Chronological test", "Unseen-floats test"]):
    inv = r[split]["inversions"]
    x = np.arange(2)
    w = 0.13
    for i, (n, l, c) in enumerate(zip(names, labels, cols)):
        v = [inv[n]["pct_gt_0.01"], inv[n]["pct_gt_0.03"]]
        b = ax.bar(x + (i - 2.5) * w, v, w, color=c, label=l, hatch="//" if n == "Observed" else None,
                   edgecolor="white" if n != "Observed" else "#555555")
        for xi, vi in zip(x + (i - 2.5) * w, v):
            ax.text(xi, vi + 0.25, f"{vi:.2f}" if vi < 1 else f"{vi:.1f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x, [r"$\Delta\sigma_0$ drop > 0.01 kg m$^{-3}$", r"$\Delta\sigma_0$ drop > 0.03 kg m$^{-3}$"])
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Profiles with an inversion (%)")
axes[0].set_ylim(0, 19)
axes[1].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(f"fig_inversion_thresholds.{e}", dpi=300, bbox_inches="tight")
