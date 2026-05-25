# DBRAMF: Decision-Based Recursive Adaptive Median Filtering

**Edge-Preserving Impulse Noise Removal for Salt & Pepper Noise**

COMP430 — Digital Image Processing · Final Project · Ankara University

---

## Overview

This repository contains the full implementation and experimental pipeline for **DBRAMF**, a novel decision-based recursive adaptive median filter designed to remove salt-and-pepper noise while preserving sharp edges.

Unlike classical filters that process every pixel unconditionally, DBRAMF only modifies pixels detected as noisy, preventing blur on clean regions. The causal recursive write-back step ensures that already-repaired pixels immediately become valid neighbors for subsequent repairs — increasing valid pixel density and reducing dependence on large search windows.

---

## Algorithm

DBRAMF processes each pixel in raster order through three stages:

| Stage | Description |
|-------|-------------|
| **1. Decision** | If `P(i,j) ∈ {0, 255}` → noisy pixel, begin restoration. Otherwise keep original. |
| **2. Adaptive Window** | Collect valid neighbors (pixels ∉ {0, 255}) in a 3×3 window. Expand to 5×5, then 7×7 if no valid pixel found. |
| **3. Recursive Feedback** | Compute **median** of valid neighbors and write the result immediately back to the working buffer, so subsequent pixels can use it as valid data. |

**Fallback**: If the 7×7 window contains no valid pixel, the pixel is replaced by the **Trimmed Global Mean** (mean of all non-noise pixels in the image). This guarantees termination and prevents infinite loops at extreme noise densities.

---

## Results Summary

Average metrics over the full **Kodak24** benchmark (24 images, noise ratios 10 %–90 %):

| Noise | DBRAMF PSNR | SMF PSNR | Mean PSNR | DBRAMF–SMF gain |
|------:|------------:|---------:|----------:|----------------:|
| 10 %  | 32.84 dB    | 29.34 dB | 22.77 dB  | **+3.50 dB**    |
| 30 %  | 29.79 dB    | 22.45 dB | 18.11 dB  | **+7.34 dB**    |
| 50 %  | 27.06 dB    | 14.87 dB | 15.42 dB  | **+12.20 dB**   |
| 70 %  | 24.23 dB    |  9.77 dB | 13.50 dB  | **+14.47 dB**   |
| 90 %  | 19.84 dB    |  6.42 dB | 11.96 dB  | **+13.41 dB**   |

Peak IEF of **104.9** achieved at 30 % noise (vs. SMF: 16.3, Mean Filter: 5.9).

Pre-generated CSVs and figures are included in the repository — see `results/` and `report_photos/`.

---

## Project Structure

```
comp430-final-project/
├── src/
│   ├── filters.py              # Mean Filter, SMF, DBRAMF implementations
│   ├── metrics.py              # PSNR, SSIM, IEF
│   ├── noise.py                # Salt & Pepper noise injection
│   └── utils.py                # Image I/O and plotting helpers
├── experiments/
│   ├── run_experiments.py      # Benchmark: all methods × all noise rates
│   └── ablation.py             # Window-size ablation (Wmax = 3, 5, 7)
├── run_all.py                  # Master pipeline (noise → benchmark → ablation → figures)
├── generate_figures.py         # IEEE-ready figure generation
├── dataset/
│   └── kodak24/                # 24 Kodak lossless PNG images (kodim01–kodim24)
├── results/
│   ├── benchmark_results.csv   # Per-image metrics for all methods × noise ratios
│   ├── ablation_results.csv    # Per-image metrics for Wmax = 3, 5, 7
│   └── report_figures/         # IEEE-format figures (300 DPI)
├── experiment_photos/
│   └── noise_XXpct/            # 216 noisy images (24 images × 9 noise levels)
└── report_photos/
    ├── curves/                 # PSNR / SSIM / IEF vs. noise ratio
    ├── comparisons/            # Side-by-side visual comparisons with metrics
    ├── patches/                # Zoomed detail crops (edge preservation)
    ├── tables/                 # Summary tables as PNG and CSV
    ├── ablation/               # Wmax ablation curves
    └── strips/                 # DBRAMF output across all 9 noise levels
```

---

## Dataset

This project uses the **Kodak24 Natural Image Dataset** — 24 lossless PNG images (768×512 px), the de-facto benchmark for image restoration research. All 24 images (`kodim01.png` … `kodim24.png`) are included in `dataset/kodak24/`.

> Original source: `http://r0k.us/graphics/kodak/`

---

## Setup

Python 3.9+ is required.

```bash
pip install numpy scipy scikit-image matplotlib pandas pillow tqdm
```

---

## Running Experiments

Pre-generated results are already in the repository. To reproduce everything from scratch:

### Full pipeline (recommended)

Runs noise generation → benchmark → ablation → all report figures in one command:

```bash
python run_all.py
```

### Benchmark only

Evaluates Mean Filter, SMF (3×3), and DBRAMF across noise ratios 10 %–90 % on all 24 Kodak images:

```bash
python experiments/run_experiments.py
```

Results saved to `results/benchmark_results.csv`.

### Ablation study

Tests DBRAMF with maximum window sizes Wmax = 3, 5, and 7:

```bash
python experiments/ablation.py
```

Results saved to `results/ablation_results.csv`.

### IEEE-format figures

Generates publication-ready figures (300 DPI) from the benchmark and ablation CSVs:

```bash
python generate_figures.py
```

---

## Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **PSNR** | `10 · log₁₀(255² / MSE)` | Higher = better (dB) |
| **SSIM** | Structural similarity index | Closer to 1 = better edge/structure preservation |
| **IEF** | `MSE(original, noisy) / MSE(original, restored)` | > 1 means improvement; higher = more noise removed |

---

## Baselines

| Method | Description |
|--------|-------------|
| **Mean Filter** | Uniform 3×3 average applied to all pixels. Fast but blurs edges significantly. |
| **SMF** | Standard Median Filter, fixed 3×3 kernel, applied to all pixels. Collapses above 50 % noise. |
| **DBRAMF** | Proposed method. Decision-based, recursive, adaptive window (up to 7×7). |

---

## Key Findings

- **DBRAMF dominates at all noise levels**, with gains over SMF ranging from +3.5 dB (10 %) to +14.6 dB (80 %).
- **Ablation study** shows that increasing Wmax beyond 3×3 yields negligible improvement, validating the causal recursive step as the primary performance driver.
- **Mean Filter outperforms SMF above 50 % noise** — at these densities the 3×3 median window contains mostly noisy pixels, so SMF selects a noise value as the median. This highlights why a decision-based approach is critical.
- **SSIM of 0.539 at 90 % noise** (vs. SMF: 0.013) demonstrates that DBRAMF preserves structural content even under extreme corruption.

---

## License

This project is released for academic purposes (COMP430 course evaluation).
