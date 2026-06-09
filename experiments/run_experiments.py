"""
Main benchmark runner.
Compares Mean Filter, SMF, and DBRAMF across noise ratios 10%–90%
on all Kodak24 images. Saves results to results/benchmark_results.csv
and metric curve figures to results/figures/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.noise import add_salt_pepper_noise
from src.filters import mean_filter, standard_median_filter, bdnd, dbramf
from src.metrics import compute_all
from src.utils import load_grayscale, list_kodak_images, plot_comparison, plot_metric_curves

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent.parent / "dataset" / "kodak24"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
NOISE_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SEED = 42

METHODS = {
    "Mean Filter": lambda img: mean_filter(img, kernel_size=3),
    "SMF (3x3)":   lambda img: standard_median_filter(img, kernel_size=3),
    "BDND":        lambda img: bdnd(img, max_window_size=7),
    "DBRAMF":      lambda img: dbramf(img, max_window_size=7),
}
# ─────────────────────────────────────────────────────────────────────────────


def main():
    images = list_kodak_images(DATASET_DIR)
    if not images:
        print(f"[ERROR] No images found in {DATASET_DIR}")
        print("  Please download Kodak24 from http://r0k.us/graphics/kodak/")
        print("  and place the 24 .png files in dataset/kodak24/")
        sys.exit(1)

    print(f"Found {len(images)} images. Running benchmark...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    visual_saved = set()  # save one comparison figure per (image, noise_ratio)

    for img_path in tqdm(images, desc="Images"):
        original = load_grayscale(img_path)

        for ratio in NOISE_RATIOS:
            noisy = add_salt_pepper_noise(original, noise_ratio=ratio, seed=SEED)

            restored_by_method = {}
            for method_name, filter_fn in METHODS.items():
                t0 = time.perf_counter()
                restored = filter_fn(noisy)
                elapsed = time.perf_counter() - t0

                metrics = compute_all(original, noisy, restored)
                records.append({
                    "image":       img_path.stem,
                    "noise_ratio": ratio,
                    "method":      method_name,
                    "PSNR":        metrics["PSNR"],
                    "SSIM":        metrics["SSIM"],
                    "IEF":         metrics["IEF"],
                    "time_s":      elapsed,
                })
                restored_by_method[method_name] = restored

            # Save one visual comparison per image at 50% noise
            key = (img_path.stem, ratio)
            if ratio == 0.50 and key not in visual_saved:
                fig_path = FIGURES_DIR / f"comparison_{img_path.stem}_noise{int(ratio*100)}.png"
                plot_comparison(original, noisy, restored_by_method, ratio, save_path=fig_path)
                visual_saved.add(key)

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ── Summary table (mean over all images) ──────────────────────────────────
    summary = (
        df.groupby(["noise_ratio", "method"])[["PSNR", "SSIM", "IEF"]]
        .mean()
        .round(4)
        .reset_index()
    )
    summary_path = RESULTS_DIR / "summary_mean.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary table saved to {summary_path}")
    print("\n=== Mean PSNR across all images ===")
    pivot = summary.pivot(index="noise_ratio", columns="method", values="PSNR")
    print(pivot.to_string())

    # ── Metric curves ─────────────────────────────────────────────────────────
    for metric in ("PSNR", "SSIM", "IEF"):
        plot_metric_curves(summary, metric,
                           save_path=FIGURES_DIR / f"curve_{metric}.png")
    print(f"\nFigures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
