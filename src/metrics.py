import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


def psnr(original: np.ndarray, restored: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB). Higher is better."""
    mse = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(original: np.ndarray, restored: np.ndarray) -> float:
    """Structural Similarity Index. Range [0,1], higher is better."""
    if original.ndim == 3:
        return sk_ssim(original, restored, data_range=255, channel_axis=2)
    return sk_ssim(original, restored, data_range=255)


def ief(original: np.ndarray, noisy: np.ndarray, restored: np.ndarray) -> float:
    """
    Image Enhancement Factor.
    IEF = MSE(original, noisy) / MSE(original, restored)
    Values > 1 indicate improvement over the noisy input.
    """
    mse_noisy = np.mean((original.astype(np.float64) - noisy.astype(np.float64)) ** 2)
    mse_restored = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
    if mse_restored == 0:
        return float("inf")
    return mse_noisy / mse_restored


def compute_all(original: np.ndarray, noisy: np.ndarray, restored: np.ndarray) -> dict:
    return {
        "PSNR": psnr(original, restored),
        "SSIM": ssim(original, restored),
        "IEF": ief(original, noisy, restored),
    }
