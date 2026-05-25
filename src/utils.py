import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pathlib import Path


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as uint8 numpy array (RGB or grayscale)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_grayscale(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def save_image(array: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def list_kodak_images(dataset_dir: str | Path) -> list[Path]:
    """Return sorted list of .png files in the Kodak24 dataset directory."""
    p = Path(dataset_dir)
    images = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg"))
    return images


def plot_comparison(original, noisy, results: dict, noise_ratio: float,
                    save_path: str | Path | None = None):
    """
    Plot original, noisy, and all restored images side-by-side.
    results: {method_name: restored_array}
    """
    n_cols = 2 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(original, cmap="gray" if original.ndim == 2 else None)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray" if noisy.ndim == 2 else None)
    axes[1].set_title(f"Noisy ({int(noise_ratio*100)}%)")
    axes[1].axis("off")

    for ax, (name, arr) in zip(axes[2:], results.items()):
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(df, metric: str, save_path: str | Path | None = None):
    """
    Plot metric vs noise_ratio curves for all methods.
    df: pandas DataFrame with columns [noise_ratio, method, metric_value]
    """
    import pandas as pd

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, grp in df.groupby("method"):
        grp_sorted = grp.sort_values("noise_ratio")
        ax.plot(grp_sorted["noise_ratio"] * 100, grp_sorted[metric],
                marker="o", label=method)

    ax.set_xlabel("Noise Ratio (%)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs Noise Ratio")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
