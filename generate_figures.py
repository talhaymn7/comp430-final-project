"""
Generate all figures needed for the IEEE report:
  1. fig_comparison_10pct.png   - visual comparison at 10% noise
  2. fig_comparison_50pct.png   - visual comparison at 50% noise
  3. fig_comparison_90pct.png   - visual comparison at 90% noise
  4. fig_psnr_curve.png         - PSNR vs noise ratio (all methods)
  5. fig_ssim_curve.png         - SSIM vs noise ratio
  6. fig_ief_curve.png          - IEF vs noise ratio
  7. fig_ablation_psnr.png      - ablation: Wmax effect on PSNR
  8. fig_flowchart.png          - DBRAMF algorithm flowchart
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
from pathlib import Path

from src.noise import add_salt_pepper_noise
from src.filters import mean_filter, standard_median_filter, dbramf
from src.metrics import psnr, ssim
from src.utils import load_grayscale

RESULTS_DIR = Path("results")
FIGS_DIR    = Path("results/report_figures")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

# ── 1–3. Visual comparison figures ───────────────────────────────────────────
print("Generating comparison images ...")
img_path = Path("dataset/kodak24/kodim05.png")   # nice texture variety
original = load_grayscale(img_path)

# Crop a representative 256×256 region (top-left area, avoids border artefacts)
crop = original[100:356, 200:456]

for ratio in [0.10, 0.50, 0.90]:
    noisy   = add_salt_pepper_noise(crop, ratio, seed=SEED)
    r_mean  = mean_filter(noisy, kernel_size=3)
    r_smf   = standard_median_filter(noisy, kernel_size=3)
    r_dbr   = dbramf(noisy, max_window_size=7)

    p_smf  = psnr(crop, r_smf);  s_smf  = ssim(crop, r_smf)
    p_dbr  = psnr(crop, r_dbr);  s_dbr  = ssim(crop, r_dbr)
    p_mean = psnr(crop, r_mean); s_mean = ssim(crop, r_mean)

    fig, axes = plt.subplots(1, 5, figsize=(12, 2.8))
    panels = [
        (crop,   "Original",                              ""),
        (noisy,  f"Noisy ({int(ratio*100)} %)",           ""),
        (r_mean, "Mean Filter",  f"PSNR={p_mean:.1f} dB\nSSIM={s_mean:.3f}"),
        (r_smf,  "SMF (3×3)",   f"PSNR={p_smf:.1f} dB\nSSIM={s_smf:.3f}"),
        (r_dbr,  "DBRAMF",      f"PSNR={p_dbr:.1f} dB\nSSIM={s_dbr:.3f}"),
    ]
    for ax, (img, title, sub) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
        if sub:
            ax.set_xlabel(sub, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    plt.tight_layout(pad=0.4)
    fname = FIGS_DIR / f"fig_comparison_{int(ratio*100):02d}pct.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

# ── 4–6. Metric curve figures ─────────────────────────────────────────────────
print("Generating metric curves ...")
df = pd.read_csv(RESULTS_DIR / "benchmark_results.csv")
summary = df.groupby(["noise_ratio","method"])[["PSNR","SSIM","IEF"]].mean().reset_index()

colors = {"DBRAMF": "#1a6faf", "SMF (3x3)": "#d62728", "Mean Filter": "#2ca02c"}
markers= {"DBRAMF": "o",       "SMF (3x3)": "s",        "Mean Filter": "^"}
linestyles = {"DBRAMF": "-",   "SMF (3x3)": "--",        "Mean Filter": ":"}

for metric, ylabel, ylim in [
    ("PSNR", "PSNR (dB)",  (4, 36)),
    ("SSIM", "SSIM",       (0, 1.02)),
    ("IEF",  "IEF",        (0, 120)),
]:
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for method, grp in summary.groupby("method"):
        g = grp.sort_values("noise_ratio")
        ax.plot(g["noise_ratio"]*100, g[metric],
                color=colors[method], marker=markers[method],
                linestyle=linestyles[method], linewidth=1.5,
                markersize=5, label=method)

    ax.set_xlabel("Noise Ratio (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Mean {metric} — Kodak24 (24 images)")
    ax.set_xlim(8, 92); ax.set_ylim(*ylim)
    ax.set_xticks(range(10, 91, 10))
    ax.legend(loc="upper right" if metric != "IEF" else "upper left",
              framealpha=0.85, edgecolor="gray")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fname = FIGS_DIR / f"fig_{metric.lower()}_curve.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

# ── 7. Ablation figure ────────────────────────────────────────────────────────
print("Generating ablation figure ...")
abl = pd.read_csv(RESULTS_DIR / "ablation_results.csv")

# Parse Wmax from method name, e.g. "DBRAMF W3×3" or "DBRAMF Wmax=3"
import re
def parse_wmax(s):
    m = re.search(r'(\d+)', str(s))
    return int(m.group(1)) if m else 0
abl["Wmax"] = abl["method"].apply(parse_wmax)

abl_sum = abl.groupby(["noise_ratio","Wmax"])[["PSNR","SSIM"]].mean().reset_index()

abl_colors = {3: "#1f77b4", 5: "#ff7f0e", 7: "#2ca02c"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
for wmax, grp in abl_sum.groupby("Wmax"):
    g = grp.sort_values("noise_ratio")
    axes[0].plot(g["noise_ratio"]*100, g["PSNR"],
                 color=abl_colors[wmax], marker="o", linewidth=1.5,
                 markersize=4, label=f"Wmax={wmax}")
    axes[1].plot(g["noise_ratio"]*100, g["SSIM"],
                 color=abl_colors[wmax], marker="o", linewidth=1.5,
                 markersize=4, label=f"Wmax={wmax}")

for ax, ylabel, title in zip(axes,
    ["PSNR (dB)", "SSIM"],
    ["PSNR vs Noise Ratio", "SSIM vs Noise Ratio"]):
    ax.set_xlabel("Noise Ratio (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(10,91,10))
    ax.legend(framealpha=0.85, edgecolor="gray")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Ablation Study: Effect of Maximum Window Size (Wmax)", fontsize=9, fontweight="bold")
plt.tight_layout()
fname = FIGS_DIR / "fig_ablation_psnr.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

# ── 8. Flowchart ──────────────────────────────────────────────────────────────
print("Generating flowchart ...")

fig, ax = plt.subplots(figsize=(5.0, 8.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 17)
ax.axis("off")

def draw_box(ax, x, y, w, h, text, style="round,pad=0.1",
             facecolor="#DDEEFF", edgecolor="#2255AA", fontsize=8, bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=style,
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, multialignment="center",
            wrap=True)

def draw_diamond(ax, x, y, w, h, text, facecolor="#FFF3CC", edgecolor="#AA7700"):
    dx, dy = w/2, h/2
    diamond = plt.Polygon([[x, y+dy],[x+dx, y],[x, y-dy],[x-dx, y]],
                           closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5,
            fontweight="bold", multialignment="center")

def arrow(ax, x1, y1, x2, y2, label="", lx=None, ly=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#333333",
                                lw=1.2, connectionstyle="arc3,rad=0"))
    if label:
        ax.text(lx or (x1+x2)/2 + 0.15, ly or (y1+y2)/2, label,
                fontsize=7, color="#444444")

cx = 5.0

# Boxes top to bottom
draw_box(ax, cx, 16.0, 5.5, 0.8, "START: Input image I", facecolor="#C8E6C9", edgecolor="#2E7D32", bold=True)
arrow(ax, cx, 15.6, cx, 15.1)

draw_box(ax, cx, 14.7, 5.5, 0.75,
         "Pre-compute TGM = mean({P : P ∉ {0,255}})\nInitialize working copy R ← I",
         facecolor="#E3F2FD", edgecolor="#1565C0")
arrow(ax, cx, 14.33, cx, 13.8)

draw_box(ax, cx, 13.45, 5.5, 0.65, "For each pixel P(i,j) in raster order",
         facecolor="#E8EAF6", edgecolor="#283593", bold=True)
arrow(ax, cx, 13.13, cx, 12.5)

draw_diamond(ax, cx, 12.0, 5.0, 1.0, "P(i,j) ∈ {0, 255}?\n(DECISION STAGE)")
arrow(ax, cx, 11.5, cx, 10.9, "YES")
# No branch → right
ax.annotate("", xy=(8.8, 12.0), xytext=(7.5, 12.0),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
ax.text(7.65, 12.15, "NO", fontsize=7, color="#444")
draw_box(ax, 9.1, 12.0, 1.6, 0.55, "Keep P(i,j)\nunchanged", facecolor="#F1F8E9", edgecolor="#558B2F", fontsize=7)

draw_box(ax, cx, 10.45, 5.5, 0.75,
         "Set k = 1  (start with 3×3 window)\nV_k = {R(r,c) : |r-i|≤k, |c-j|≤k, R∉{0,255}}",
         facecolor="#E3F2FD", edgecolor="#1565C0")
arrow(ax, cx, 10.08, cx, 9.45)

draw_diamond(ax, cx, 8.95, 5.0, 1.0, "|V_k| > 0?\n(Valid neighbors found?)")
arrow(ax, cx, 8.45, cx, 7.85, "YES")

# No branch
ax.annotate("", xy=(8.8, 8.95), xytext=(7.5, 8.95),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
ax.text(7.65, 9.1, "NO", fontsize=7, color="#444")
draw_diamond(ax, 9.1, 8.1, 1.9, 0.85, "k < k_max?\n(k≤3)", facecolor="#FFF3CC", edgecolor="#AA7700")
# Yes → expand
ax.annotate("", xy=(9.1, 7.4), xytext=(9.1, 7.68),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
ax.text(9.2, 7.52, "YES", fontsize=7, color="#444")
draw_box(ax, 9.1, 7.1, 1.9, 0.55, "k ← k+1\nExpand window", facecolor="#FFF9C4", edgecolor="#F9A825", fontsize=7)
# Loop back up
ax.annotate("", xy=(9.1, 10.45), xytext=(9.1, 7.38),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2,
                            connectionstyle="arc3,rad=0"))

# No at k_max → fallback
ax.annotate("", xy=(8.8, 8.1), xytext=(8.05, 8.1),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
# Draw fallback slightly left
draw_box(ax, 9.1, 8.1 - 1.1, 1.9, 0.55, "Fallback:\nR(i,j) ← TGM", facecolor="#FFCCBC", edgecolor="#BF360C", fontsize=7)
ax.text(8.2, 8.24, "NO", fontsize=7, color="#444")
# arrow from fallback down to write-back
ax.annotate("", xy=(cx + 2.75, 7.35), xytext=(9.1, 6.78),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

draw_box(ax, cx, 7.4, 5.5, 0.75,
         "R(i,j) ← median(V_k)         (Eq. 8)\n(MEDIAN ESTIMATION)",
         facecolor="#EDE7F6", edgecolor="#4527A0", bold=True)
arrow(ax, cx, 7.03, cx, 6.4)

draw_box(ax, cx, 6.0, 5.5, 0.7,
         "Write R(i,j) back immediately\n(CAUSAL RECURSIVE FEEDBACK)",
         facecolor="#FCE4EC", edgecolor="#880E4F", bold=True)
arrow(ax, cx, 5.65, cx, 5.1)

draw_diamond(ax, cx, 4.6, 5.0, 0.9, "More pixels?")
# Yes → loop back
ax.annotate("", xy=(2.0, 4.6), xytext=(2.5, 4.6),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
ax.text(1.55, 4.75, "YES", fontsize=7, color="#444")
ax.annotate("", xy=(2.0, 13.45), xytext=(2.0, 4.6),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
ax.annotate("", xy=(2.25, 13.45), xytext=(2.0, 13.45),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

arrow(ax, cx, 4.15, cx, 3.55, "NO")

draw_box(ax, cx, 3.15, 5.5, 0.75, "Output restored image R",
         facecolor="#C8E6C9", edgecolor="#2E7D32", bold=True)
arrow(ax, cx, 2.78, cx, 2.25)

draw_box(ax, cx, 1.9, 5.5, 0.65, "END", facecolor="#C8E6C9", edgecolor="#2E7D32", bold=True)

ax.set_title("Fig. 1.  DBRAMF Algorithm Flowchart", fontsize=9, fontweight="bold", pad=6)
plt.tight_layout()
fname = FIGS_DIR / "fig_flowchart.png"
plt.savefig(fname, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {fname}")

print("\nAll figures generated successfully.")
