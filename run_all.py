"""
run_all.py — DBRAMF Master Pipeline

Outputs:
  experiment_photos/noise_XXpct/  — 24×9 noisy images
  results/                        — benchmark & ablation CSVs
  report_photos/curves/           — PSNR / SSIM / IEF curves (300 DPI)
  report_photos/comparisons/      — side-by-side visual comparisons with metrics
  report_photos/patches/          — zoomed-in detail crops
  report_photos/tables/           — summary tables as PNG + CSV
  report_photos/ablation/         — Wmax ablation curves
  report_photos/strips/           — DBRAMF results across all noise levels
"""

import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.noise import add_salt_pepper_noise
from src.filters import mean_filter, standard_median_filter, dbramf
from src.metrics import compute_all
from src.utils import load_grayscale, list_kodak_images, save_image

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR       = Path("dataset/kodak24")
EXP_PHOTOS_DIR    = Path("experiment_photos")
REPORT_PHOTOS_DIR = Path("report_photos")
RESULTS_DIR       = Path("results")

NOISE_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SEED = 42

METHODS = {
    "Mean Filter": lambda img: mean_filter(img, kernel_size=3),
    "SMF (3x3)":   lambda img: standard_median_filter(img, kernel_size=3),
    "DBRAMF":      lambda img: dbramf(img, max_window_size=7),
}

# Images chosen for visual comparison in the report (visually rich Kodak images)
REPORT_IMAGES       = ["kodim08", "kodim15", "kodim23"]
REPORT_NOISE_RATIOS = [0.30, 0.50, 0.70, 0.90]
PATCH_SIZE          = 256  # px — zoomed crop size

METHOD_COLORS  = {"Mean Filter": "#e74c3c", "SMF (3x3)": "#3498db", "DBRAMF": "#2ecc71"}
METHOD_MARKERS = {"Mean Filter": "s",       "SMF (3x3)": "^",       "DBRAMF": "o"}
# ──────────────────────────────────────────────────────────────────────────────


def banner(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


# ── STEP 1 ────────────────────────────────────────────────────────────────────

def step1_generate_noisy_images(images):
    banner("STEP 1 -- Noisy images -> experiment_photos/")
    noisy_cache = {}

    for img_path in tqdm(images, desc="Generating"):
        original = load_grayscale(img_path)
        for ratio in NOISE_RATIOS:
            noisy = add_salt_pepper_noise(original, noise_ratio=ratio, seed=SEED)
            noisy_cache[(img_path.stem, ratio)] = (original, noisy)

            out_dir = EXP_PHOTOS_DIR / f"noise_{int(ratio*100):02d}pct"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_image(noisy, out_dir / f"{img_path.stem}_noisy.png")

    total = len(images) * len(NOISE_RATIOS)
    print(f"  {total} noisy images saved to {EXP_PHOTOS_DIR}/")
    return noisy_cache


# ── STEP 2 ────────────────────────────────────────────────────────────────────

def step2_benchmark(images, noisy_cache):
    banner("STEP 2 — Benchmark: Mean Filter / SMF / DBRAMF")
    records = []
    restored_cache = {}

    bar = tqdm(total=len(images) * len(NOISE_RATIOS), desc="Benchmark")
    for img_path in images:
        for ratio in NOISE_RATIOS:
            original, noisy = noisy_cache[(img_path.stem, ratio)]
            for mname, fn in METHODS.items():
                t0 = time.perf_counter()
                restored = fn(noisy)
                elapsed = time.perf_counter() - t0
                m = compute_all(original, noisy, restored)
                records.append({
                    "image": img_path.stem, "noise_ratio": ratio,
                    "method": mname,
                    "PSNR": m["PSNR"], "SSIM": m["SSIM"], "IEF": m["IEF"],
                    "time_s": elapsed,
                })
                restored_cache[(img_path.stem, ratio, mname)] = restored
            bar.update(1)
    bar.close()

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
    print(f"  Saved: {RESULTS_DIR}/benchmark_results.csv")
    return df, restored_cache


# ── STEP 3 ────────────────────────────────────────────────────────────────────

def step3_ablation(images, noisy_cache):
    banner("STEP 3 — Ablation: DBRAMF Wmax = 3, 5, 7")
    records = []
    for img_path in tqdm(images, desc="Ablation"):
        for ratio in NOISE_RATIOS:
            original, noisy = noisy_cache[(img_path.stem, ratio)]
            for wmax in [3, 5, 7]:
                restored = dbramf(noisy, max_window_size=wmax)
                m = compute_all(original, noisy, restored)
                records.append({
                    "image": img_path.stem, "noise_ratio": ratio,
                    "method": f"DBRAMF W{wmax}×{wmax}",
                    "PSNR": m["PSNR"], "SSIM": m["SSIM"], "IEF": m["IEF"],
                })
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
    print(f"  Saved: {RESULTS_DIR}/ablation_results.csv")
    return df


# ── STEP 4 ────────────────────────────────────────────────────────────────────

def step4_report_figures(df_bench, df_abl, images, noisy_cache, restored_cache):
    banner("STEP 4 — Report figures → report_photos/")
    REPORT_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

    summary = (
        df_bench.groupby(["noise_ratio", "method"])[["PSNR", "SSIM", "IEF"]]
        .mean().round(4).reset_index()
    )

    # ── 4a: Metric curves ──────────────────────────────────────────────────
    curves_dir = REPORT_PHOTOS_DIR / "curves"
    curves_dir.mkdir(exist_ok=True)

    for metric in ("PSNR", "SSIM", "IEF"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for mname, grp in summary.groupby("method"):
            grp = grp.sort_values("noise_ratio")
            ax.plot(
                grp["noise_ratio"] * 100, grp[metric],
                color=METHOD_COLORS.get(mname, "gray"),
                marker=METHOD_MARKERS.get(mname, "o"),
                linewidth=2, markersize=7, label=mname,
            )
        ax.set_xlabel("Salt & Pepper Noise Ratio (%)", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"Average {metric} vs. Noise Ratio — Kodak24", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        plt.tight_layout()
        fig.savefig(curves_dir / f"curve_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  Curves: {curves_dir}/")

    # ── 4b: Visual comparisons ─────────────────────────────────────────────
    comp_dir = REPORT_PHOTOS_DIR / "comparisons"
    comp_dir.mkdir(exist_ok=True)
    existing_stems = {img.stem for img in images}

    for stem in REPORT_IMAGES:
        if stem not in existing_stems:
            continue
        for ratio in REPORT_NOISE_RATIOS:
            original, noisy = noisy_cache[(stem, ratio)]
            mnames = list(METHODS.keys())
            n_cols = 2 + len(mnames)
            fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.5))
            fig.suptitle(f"{stem}  |  Noise {int(ratio*100)}%", fontsize=13)

            def show(ax, img, title):
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
                ax.set_title(title, fontsize=9)
                ax.axis("off")

            show(axes[0], original, "Original")
            show(axes[1], noisy,    f"Noisy ({int(ratio*100)}%)")

            for ax, mname in zip(axes[2:], mnames):
                restored = restored_cache.get((stem, ratio, mname))
                if restored is not None:
                    m = compute_all(original, noisy, restored)
                    show(ax, restored,
                         f"{mname}\nPSNR {m['PSNR']:.2f} dB\nSSIM {m['SSIM']:.4f}")

            plt.tight_layout()
            fig.savefig(comp_dir / f"comp_{stem}_n{int(ratio*100):02d}.png",
                        dpi=200, bbox_inches="tight")
            plt.close(fig)

    print(f"  Comparisons: {comp_dir}/")

    # ── 4c: Zoomed patches ─────────────────────────────────────────────────
    patches_dir = REPORT_PHOTOS_DIR / "patches"
    patches_dir.mkdir(exist_ok=True)

    for stem in REPORT_IMAGES:
        if stem not in existing_stems:
            continue
        for ratio in [0.50, 0.90]:   # two representative noise levels
            original, noisy = noisy_cache[(stem, ratio)]
            H, W = original.shape
            r0 = (H - PATCH_SIZE) // 2
            c0 = (W - PATCH_SIZE) // 2
            crop = lambda arr: arr[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]

            mnames = list(METHODS.keys())
            n_cols = 2 + len(mnames)
            fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.8))
            fig.suptitle(f"{stem} — Patch (center {PATCH_SIZE}×{PATCH_SIZE}) | Noise {int(ratio*100)}%",
                         fontsize=11)

            axes[0].imshow(crop(original), cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("Original", fontsize=9); axes[0].axis("off")
            axes[1].imshow(crop(noisy), cmap="gray", vmin=0, vmax=255)
            axes[1].set_title(f"Noisy ({int(ratio*100)}%)", fontsize=9); axes[1].axis("off")

            for ax, mname in zip(axes[2:], mnames):
                restored = restored_cache.get((stem, ratio, mname))
                if restored is not None:
                    ax.imshow(crop(restored), cmap="gray", vmin=0, vmax=255)
                    ax.set_title(mname, fontsize=9)
                ax.axis("off")

            plt.tight_layout()
            fig.savefig(patches_dir / f"patch_{stem}_n{int(ratio*100):02d}.png",
                        dpi=200, bbox_inches="tight")
            plt.close(fig)

    print(f"  Patches: {patches_dir}/")

    # ── 4d: Summary tables ─────────────────────────────────────────────────
    tables_dir = REPORT_PHOTOS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Save raw CSVs for LaTeX
    for metric in ("PSNR", "SSIM", "IEF"):
        pivot = summary.pivot(index="noise_ratio", columns="method", values=metric).round(4)
        pivot.index = [f"{int(r*100)}%" for r in pivot.index]
        pivot.to_csv(tables_dir / f"table_{metric}.csv")

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.axis("off")
        tbl = ax.table(
            cellText=pivot.values,
            rowLabels=pivot.index,
            colLabels=pivot.columns.tolist(),
            cellLoc="center", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 2.0)

        # Highlight best value per row (green)
        for row_i in range(len(pivot)):
            best_col_i = int(np.argmax(pivot.iloc[row_i].values))
            tbl[(row_i + 1, best_col_i)].set_facecolor("#c8e6c9")

        # Header row style
        for col_i in range(len(pivot.columns)):
            tbl[(0, col_i)].set_facecolor("#cfd8dc")

        ax.set_title(f"Mean {metric} — Kodak24 (24 images, green = best per row)",
                     fontsize=12, pad=20)
        plt.tight_layout()
        fig.savefig(tables_dir / f"table_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    summary.to_csv(tables_dir / "full_summary.csv", index=False)
    print(f"  Tables: {tables_dir}/")

    # ── 4e: Ablation curves ────────────────────────────────────────────────
    abl_dir = REPORT_PHOTOS_DIR / "ablation"
    abl_dir.mkdir(exist_ok=True)

    abl_summary = (
        df_abl.groupby(["noise_ratio", "method"])[["PSNR", "SSIM", "IEF"]]
        .mean().round(4).reset_index()
    )
    abl_colors = {"DBRAMF W3×3": "#e67e22", "DBRAMF W5×5": "#9b59b6", "DBRAMF W7×7": "#2ecc71"}

    for metric in ("PSNR", "SSIM", "IEF"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for mname, grp in abl_summary.groupby("method"):
            grp = grp.sort_values("noise_ratio")
            ax.plot(grp["noise_ratio"] * 100, grp[metric],
                    color=abl_colors.get(mname, "gray"),
                    marker="o", linewidth=2, markersize=7, label=mname)
        ax.set_xlabel("Salt & Pepper Noise Ratio (%)", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"Ablation Study: {metric} vs. Noise Ratio (Wmax variants)", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        plt.tight_layout()
        fig.savefig(abl_dir / f"ablation_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  Ablation: {abl_dir}/")

    # ── 4f: Noise-level strip ──────────────────────────────────────────────
    strips_dir = REPORT_PHOTOS_DIR / "strips"
    strips_dir.mkdir(exist_ok=True)

    for stem in REPORT_IMAGES[:1]:   # one strip per representative image
        if stem not in existing_stems:
            continue
        fig, axes = plt.subplots(1, len(NOISE_RATIOS), figsize=(3.8 * len(NOISE_RATIOS), 4))
        fig.suptitle(f"DBRAMF restoration — {stem} — all noise levels", fontsize=12)
        for ax, ratio in zip(axes, NOISE_RATIOS):
            restored = restored_cache.get((stem, ratio, "DBRAMF"))
            if restored is not None:
                ax.imshow(restored, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{int(ratio*100)}%", fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(strips_dir / f"strip_{stem}_DBRAMF.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Also: noisy strip
        fig, axes = plt.subplots(1, len(NOISE_RATIOS), figsize=(3.8 * len(NOISE_RATIOS), 4))
        fig.suptitle(f"Noisy input — {stem} — all noise levels", fontsize=12)
        for ax, ratio in zip(axes, NOISE_RATIOS):
            _, noisy = noisy_cache[(stem, ratio)]
            ax.imshow(noisy, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{int(ratio*100)}%", fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(strips_dir / f"strip_{stem}_noisy.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  Strips: {strips_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  DBRAMF — Full Experimental Pipeline")
    print("=" * 62)

    images = list_kodak_images(DATASET_DIR)
    if not images:
        print(f"[ERROR] No images in {DATASET_DIR}")
        sys.exit(1)
    print(f"\n  Dataset : {DATASET_DIR}  ({len(images)} images)")
    print(f"  Noise   : {[int(r*100) for r in NOISE_RATIOS]}%")
    print(f"  Methods : {list(METHODS.keys())}")

    t0 = time.perf_counter()

    noisy_cache                    = step1_generate_noisy_images(images)
    df_bench, restored_cache       = step2_benchmark(images, noisy_cache)
    df_abl                         = step3_ablation(images, noisy_cache)
    step4_report_figures(df_bench, df_abl, images, noisy_cache, restored_cache)

    elapsed = time.perf_counter() - t0
    m, s = divmod(int(elapsed), 60)

    banner(f"Pipeline complete in {m}m {s}s")
    print(f"  experiment_photos/   — 24×9 noisy images")
    print(f"  results/             — benchmark_results.csv, ablation_results.csv")
    print(f"  report_photos/curves/      — PSNR / SSIM / IEF curves")
    print(f"  report_photos/comparisons/ — side-by-side visual comparisons")
    print(f"  report_photos/patches/     — zoomed detail crops")
    print(f"  report_photos/tables/      — summary tables (PNG + CSV)")
    print(f"  report_photos/ablation/    — Wmax ablation curves")
    print(f"  report_photos/strips/      — noise-level strips")
    print()


if __name__ == "__main__":
    main()
