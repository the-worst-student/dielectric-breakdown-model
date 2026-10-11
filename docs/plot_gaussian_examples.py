"""Plot the three documented examples; no model inference or synthetic masks."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D

DOCS = Path(__file__).resolve().parent
with (DOCS / "data/gaussian_test_examples.csv").open() as stream:
    rows = list(csv.DictReader(stream))

targets = ["eta", "mean_breakdown_field", "correlation_length", "weibull_shape"]
labels = [r"$\eta$", r"$E_{\mathrm{bd}}^{(0)}$", r"$\ell_c$", r"$k$"]
limits = [(0.4, 3.65), (2.6, 3.5), (-4, 21), (-1, 21)]
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})
fig, axes = plt.subplots(3, 5, figsize=(13, 7.8),
                         gridspec_kw={"width_ratios": [0.75, 1, 1, 1, 1]})
fig.subplots_adjust(left=.04, right=.985, top=.85, bottom=.10, hspace=.65, wspace=.40)
fig.suptitle("Gaussian inversion · three test examples", fontsize=19, y=.97, weight="bold")
fig.text(.5, .92, "Observed mask → parameter means and calibrated 95% intervals",
         ha="center", color="#475569", fontsize=12)
for row in range(3):
    mask_ax = axes[row, 0]
    mask_ax.imshow(mpimg.imread(DOCS / f"data/test_example_mask_{row+1}.png"),
                   interpolation="nearest")
    mask_ax.set_axis_off()
    mask_ax.set_title(f"Example {row+1}", fontsize=11, pad=9)
    for col, (target, label, limit) in enumerate(zip(targets, labels, limits), 1):
        record = next(r for r in rows if int(r["example"]) == row+1 and r["target"] == target)
        truth, mean, lower, upper = [float(record[k]) for k in ("true", "mean", "lower95", "upper95")]
        ax = axes[row, col]
        ax.errorbar(mean, .40, xerr=[[mean-lower], [upper-mean]],
                    fmt="o", color="#1976b9", capsize=5, markersize=5, linewidth=1.6, zorder=3)
        ax.plot(truth, .40, "x", color="#111827", markersize=8, markeredgewidth=1.8, zorder=4)
        ax.text(.5, .94, f"True {truth:.2f}  ·  Mean {mean:.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.set(xlim=limit, ylim=(0, 1), yticks=[])
        ax.set_title(label, fontsize=14, pad=9)
        if row == 2:
            ax.set_xlabel("Parameter value", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=.15)
        for spine in ax.spines.values():
            spine.set_color("#b8c3ce")
handles = [Line2D([], [], marker="x", color="#111827", linestyle="None", markersize=8, label="True value"),
           Line2D([], [], marker="o", color="#1976b9", linewidth=1.6, label="Mean and calibrated 95% interval")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015), ncol=2, frameon=False)
out = DOCS / "images/gaussian_test_examples.png"
fig.savefig(out, dpi=180, facecolor="white")
plt.close(fig)
print(out)
