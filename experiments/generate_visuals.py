"""
Generate visual comparison figures at 10%, 50%, and 90% noise for kodim05.png
Includes all four methods: Mean Filter, SMF, BDND, DBRAMF
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from src.noise import add_salt_pepper_noise
from src.filters import mean_filter, standard_median_filter, bdnd, dbramf
from src.utils import load_grayscale, plot_comparison

DATASET_DIR = Path(__file__).parent.parent / "dataset" / "kodak24"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
SEED = 42
IMAGE_NAME = "kodim05.png"
NOISE_RATIOS = [0.10, 0.50, 0.90]
CROP_SIZE = 256

METHODS = {
    "Mean Filter": lambda img: mean_filter(img, kernel_size=3),
    "SMF (3x3)":   lambda img: standard_median_filter(img, kernel_size=3),
    "BDND":        lambda img: bdnd(img, max_window_size=7),
    "DBRAMF":      lambda img: dbramf(img, max_window_size=7),
}

def main():
    img_path = DATASET_DIR / IMAGE_NAME
    if not img_path.exists():
        print(f"[ERROR] {img_path} not found.")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    original = load_grayscale(img_path)

    # Centre crop
    H, W = original.shape
    r0 = (H - CROP_SIZE) // 2
    c0 = (W - CROP_SIZE) // 2
    original_crop = original[r0:r0+CROP_SIZE, c0:c0+CROP_SIZE]

    for ratio in NOISE_RATIOS:
        print(f"Processing {int(ratio*100)}% noise...")
        noisy_full = add_salt_pepper_noise(original, noise_ratio=ratio, seed=SEED)
        noisy_crop = noisy_full[r0:r0+CROP_SIZE, c0:c0+CROP_SIZE]

        restored = {}
        for name, fn in METHODS.items():
            print(f"  Running {name}...")
            restored_full = fn(noisy_full)
            restored[name] = restored_full[r0:r0+CROP_SIZE, c0:c0+CROP_SIZE]

        save_path = FIGURES_DIR / f"comparison_kodim05_noise{int(ratio*100)}.png"
        plot_comparison(original_crop, noisy_crop, restored, ratio, save_path=save_path)
        print(f"  Saved: {save_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
