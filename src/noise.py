import numpy as np


def add_salt_pepper_noise(image: np.ndarray, noise_ratio: float, seed: int = 42) -> np.ndarray:
    """
    Add salt-and-pepper noise to a grayscale or RGB image.
    noise_ratio: fraction of total pixels to corrupt (e.g. 0.5 = 50%)
    Returns uint8 array same shape as input.
    """
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    h, w = image.shape[:2]
    total_pixels = h * w

    n_noisy = int(total_pixels * noise_ratio)
    n_salt = n_noisy // 2
    n_pepper = n_noisy - n_salt

    # Flat indices for affected pixels
    indices = rng.choice(total_pixels, size=n_noisy, replace=False)
    salt_idx = indices[:n_salt]
    pepper_idx = indices[n_salt:]

    rows_s, cols_s = np.unravel_index(salt_idx, (h, w))
    rows_p, cols_p = np.unravel_index(pepper_idx, (h, w))

    if image.ndim == 3:
        noisy[rows_s, cols_s, :] = 255
        noisy[rows_p, cols_p, :] = 0
    else:
        noisy[rows_s, cols_s] = 255
        noisy[rows_p, cols_p] = 0

    return noisy.astype(np.uint8)


def actual_noise_ratio(original: np.ndarray, noisy: np.ndarray) -> float:
    """Compute fraction of pixels that actually changed."""
    return float(np.mean(original != noisy))
