"""
Ablation study: effect of maximum window size (Wmax) on DBRAMF performance.
Tests Wmax ∈ {3, 5, 7} across noise ratios 10%–90%.
Saves results to results/ablation_results.csv and figures to results/figures/.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.noise import add_salt_pepper_noise
from src.filters import dbramf
from src.metrics import compute_all
from src.utils import load_grayscale, list_kodak_images, plot_metric_curves

DATASET_DIR = Path(__file__).parent.parent / "dataset" / "kodak24"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
NOISE_RATIOS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
WINDOW_SIZES = [3, 5, 7]
SEED = 42


def main():
    images = list_kodak_images(DATASET_DIR)
    if not images:
        print(f"[ERROR] No images in {DATASET_DIR}")
        sys.exit(1)

    print(f"Ablation study — {len(images)} images, Wmax ∈ {WINDOW_SIZES}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for img_path in tqdm(images, desc="Images"):
        original = load_grayscale(img_path)
        for ratio in NOISE_RATIOS:
            noisy = add_salt_pepper_noise(original, noise_ratio=ratio, seed=SEED)
            for wmax in WINDOW_SIZES:
                restored = dbramf(noisy, max_window_size=wmax)
                metrics = compute_all(original, noisy, restored)
                records.append({
                    "image":       img_path.stem,
                    "noise_ratio": ratio,
                    "method":      f"DBRAMF Wmax={wmax}",
                    "Wmax":        wmax,
                    "PSNR":        metrics["PSNR"],
                    "SSIM":        metrics["SSIM"],
                    "IEF":         metrics["IEF"],
                })

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

    summary = (
        df.groupby(["noise_ratio", "method"])[["PSNR", "SSIM", "IEF"]]
        .mean()
        .round(4)
        .reset_index()
    )

    print("\n=== Ablation: Mean PSNR ===")
    print(summary.pivot(index="noise_ratio", columns="method", values="PSNR").to_string())

    for metric in ("PSNR", "SSIM", "IEF"):
        plot_metric_curves(summary, metric,
                           save_path=FIGURES_DIR / f"ablation_{metric}.png")
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
