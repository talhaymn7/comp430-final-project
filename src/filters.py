"""
Filters implemented:
  - mean_filter       : uniform average (baseline)
  - standard_median_filter : fixed 3x3 median (baseline)
  - dbramf            : Decision-Based Recursive Adaptive Median Filter (proposed)
"""

import numpy as np
from scipy.ndimage import uniform_filter, median_filter


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Uniform mean filter applied to every pixel."""
    if image.ndim == 3:
        result = np.stack(
            [uniform_filter(image[:, :, c].astype(np.float64), size=kernel_size)
             for c in range(image.shape[2])],
            axis=2,
        )
    else:
        result = uniform_filter(image.astype(np.float64), size=kernel_size)
    return np.clip(result, 0, 255).astype(np.uint8)


def standard_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Standard median filter (fixed kernel, applied to all pixels)."""
    if image.ndim == 3:
        result = np.stack(
            [median_filter(image[:, :, c], size=kernel_size)
             for c in range(image.shape[2])],
            axis=2,
        )
    else:
        result = median_filter(image, size=kernel_size)
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# DBRAMF — proposed method
# ---------------------------------------------------------------------------

def _is_noisy(val: int) -> bool:
    return val == 0 or val == 255


def _trimmed_global_mean(image: np.ndarray) -> float:
    """
    Trimmed Global Mean: mean of all pixels that are NOT salt/pepper.
    Used as last-resort fallback when the 7x7 window has no valid pixel.
    """
    flat = image.flatten().astype(np.float64)
    valid = flat[(flat != 0) & (flat != 255)]
    if len(valid) == 0:
        return 128.0  # degenerate case: fully corrupted channel
    return float(np.mean(valid))


def _restore_pixel(working: np.ndarray, r: int, c: int,
                   global_fallback: float, max_win: int) -> int:
    """
    Attempt to restore the pixel at (r, c) using expanding window.
    working  : in-progress image (already-repaired pixels are present)
    max_win  : maximum half-width to try (1→3x3, 2→5x5, 3→7x7)
    Returns restored integer value.
    """
    H, W = working.shape
    for half in range(1, max_win + 1):
        r0, r1 = max(0, r - half), min(H, r + half + 1)
        c0, c1 = max(0, c - half), min(W, c + half + 1)
        window = working[r0:r1, c0:c1].flatten()
        valid = window[(window != 0) & (window != 255)]
        if len(valid) > 0:
            return int(np.median(valid))
    # Fallback: trimmed global mean
    return int(round(global_fallback))


def _dbramf_channel(channel: np.ndarray, max_win: int) -> np.ndarray:
    """Apply DBRAMF to a single 2-D (grayscale) channel."""
    working = channel.copy().astype(np.int32)
    H, W = working.shape
    fallback = _trimmed_global_mean(channel)

    for r in range(H):
        for c in range(W):
            if _is_noisy(working[r, c]):
                working[r, c] = _restore_pixel(working, r, c, fallback, max_win)

    return np.clip(working, 0, 255).astype(np.uint8)


def dbramf(image: np.ndarray, max_window_size: int = 7) -> np.ndarray:
    """
    Decision-Based Recursive Adaptive Median Filter.

    max_window_size : maximum window side length (3, 5, or 7).
                      Converted to half-width internally.
    """
    if max_window_size not in (3, 5, 7):
        raise ValueError("max_window_size must be 3, 5, or 7")
    max_half = max_window_size // 2  # 1, 2, or 3

    if image.ndim == 3:
        return np.stack(
            [_dbramf_channel(image[:, :, c], max_half)
             for c in range(image.shape[2])],
            axis=2,
        )
    return _dbramf_channel(image, max_half)
