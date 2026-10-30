# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:27:48 2025

@author: Morteza
@Modified by: Claude / Jae Sung Hwang
"""

"""
Hyperspectral Anomaly Detection (RX Detector) on an ENVI image

- Reads an ENVI .hdr + image from the ./data folder next to this script.
- Any PNG/JPEG in that folder without a matching .hdr is first converted to
  ENVI format (raw .img + .hdr), so ordinary images can be used as input.
- Computes global RX anomaly scores with a robust, regularized covariance.
- Processes the cube in row-chunks (memory friendly).
- Saves: anomaly heatmap (PNG), binary mask (PNG), and raw scores (NumPy .npy).
- Also shows a quick pseudo-RGB for context.

Dependencies (install if needed):
    pip install spectral numpy matplotlib pillow

Optional (faster/more stable covariance if present):
    pip install scikit-learn
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ---- CONFIG ----
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")  # folder containing .hdr + image
MAX_SAMPLE_PIXELS = 120_000   # how many pixels to sample for mean/cov (speeds up & stabilizes)
RIDGE_ALPHA = 1e-3            # small L2 regularization added to covariance if sklearn not used
CHUNK_ROWS = 256              # process this many rows at a time (tune for RAM/speed)
ANOMALY_PERCENTILE = 99.5     # threshold (percentile) for binary anomaly mask
PSEUDO_RGB_BANDS = None       # e.g., (R_idx, G_idx, B_idx); or None to auto-pick evenly spaced
IMAGE_EXTS = (".png", ".jpg", ".jpeg")                         # ordinary images to convert to ENVI
OUTPUT_SUFFIXES = ("_RX_heatmap", "_RX_mask", "_pseudoRGB")     # our own outputs; never convert these


# ---- HELPERS ----
def image_to_envi(img_path):
    """
    Convert a PNG/JPEG into an ENVI cube: writes <base>.img (raw uint8, BIP)
    and <base>.hdr describing it. RGB images become 3 bands, grayscale 1 band.
    Returns the .hdr path.
    """
    try:
        import spectral as sp
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Converting PNG/JPEG to ENVI requires 'spectral' and 'pillow'. "
            "Install them via: pip install spectral pillow"
        ) from e

    base = os.path.splitext(img_path)[0]
    hdr_path = base + ".hdr"
    raw_path = base + ".img"

    with Image.open(img_path) as im:
        # Drop alpha / palettes; keep true grayscale as a single band
        if im.mode not in ("L", "RGB"):
            im = im.convert("RGB")
        cube = np.asarray(im, dtype=np.uint8)
    if cube.ndim == 2:
        cube = cube[:, :, np.newaxis]
    lines, samples, bands = cube.shape

    # (rows, cols, bands) in C order is exactly band-interleaved-by-pixel
    cube.tofile(raw_path)

    meta = {
        "description": f"Converted from {os.path.basename(img_path)}",
        "samples": samples,
        "lines": lines,
        "bands": bands,
        "header offset": 0,
        "file type": "ENVI Standard",
        "data type": 1,        # uint8
        "interleave": "bip",
        "byte order": 0,
    }
    sp.envi.write_envi_header(hdr_path, meta)
    print(f"[i] Converted {os.path.basename(img_path)} -> {os.path.basename(hdr_path)} "
          f"({lines}x{samples}, {bands} bands)")
    return hdr_path


def convert_images_in_dir(path):
    """Create ENVI files for every PNG/JPEG in `path` that doesn't have a .hdr yet."""
    for img_path in sorted(glob.glob(os.path.join(path, "*"))):
        base, ext = os.path.splitext(img_path)
        if ext.lower() not in IMAGE_EXTS:
            continue
        if any(s in os.path.basename(base) for s in OUTPUT_SUFFIXES):
            continue
        if os.path.exists(base + ".hdr"):
            continue
        image_to_envi(img_path)


def find_envi_hdr(path):
    hdrs = sorted(glob.glob(os.path.join(path, "*.hdr")))
    if not hdrs:
        raise FileNotFoundError(f"No .hdr file found in: {path}")
    # If multiple HDRs exist, we take the first one. Adjust as needed.
    return hdrs[0]


def open_envi_cube(hdr_path):
    try:
        import spectral as sp
    except ImportError as e:
        raise ImportError(
            "The 'spectral' package is required to read ENVI files. "
            "Install it via: pip install spectral"
        ) from e

    img = sp.envi.open(hdr_path)         # detects paired image file automatically
    # Memory map (doesn't load entire cube at once). shape: (rows, cols, bands)
    arr = img.open_memmap(writable=False)
    return arr  # numpy memmap


def sample_pixels_for_stats(arr, max_pixels):
    """
    arr: memmap of shape (R, C, B)
    Returns Xsamp: (N, B) pixel spectra chosen uniformly at random.
    """
    R, C, B = arr.shape
    N_all = R * C
    N = min(max_pixels, N_all)

    # Uniformly sample linear indices, then convert to (row, col)
    rng = np.random.default_rng(123)
    idx = rng.choice(N_all, size=N, replace=False)
    rows = idx // C
    cols = idx % C

    Xsamp = np.empty((N, B), dtype=np.float32)
    # To keep memory access efficient, group by rows
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]

    i0 = 0
    while i0 < N:
        r = rows[i0]
        # all indices for row r
        j0 = i0
        while j0 < N and rows[j0] == r:
            j0 += 1
        # pull whole row once
        row_slice = arr[r, :, :]  # (C, B)
        Xsamp[i0:j0, :] = row_slice[cols[i0:j0], :]
        i0 = j0

    # Handle NaNs/Infs if present
    good = np.isfinite(Xsamp).all(axis=1)
    return Xsamp[good]


def robust_mean_cov(X):
    """
    Compute mean and regularized covariance. If scikit-learn is available,
    use Ledoit-Wolf shrinkage. Otherwise, use np.cov + small ridge.
    X shape: (N, B)
    Returns: mu (B,), Sigma (B, B)
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(X)
        mu = lw.location_
        Sigma = lw.covariance_
        return mu.astype(np.float64), Sigma.astype(np.float64)
    except Exception:
        # Fallback: classic covariance + ridge
        mu = X.mean(axis=0)
        # rowvar=False => features in columns
        Sigma = np.cov(X, rowvar=False)
        # In case covariance returns float32/single
        Sigma = np.asarray(Sigma, dtype=np.float64)
        # Add a small ridge on the diagonal to stabilize inversion
        Sigma = Sigma + RIDGE_ALPHA * np.trace(Sigma) / Sigma.shape[0] * np.eye(Sigma.shape[0])
        return mu.astype(np.float64), Sigma


def rx_scores_chunked(arr, mu, inv_cov, chunk_rows=256):
    """
    Compute RX scores D(x) = (x-mu)^T Σ^{-1} (x-mu) in row-chunks.
    arr: memmap (R, C, B)
    mu: (B,)
    inv_cov: (B, B)
    Returns: scores (R, C) float64
    """
    R, C, B = arr.shape
    scores = np.empty((R, C), dtype=np.float64)

    # Centering vector
    mu = mu.reshape(1, B)

    for r0 in range(0, R, chunk_rows):
        r1 = min(r0 + chunk_rows, R)
        block = arr[r0:r1, :, :]        # shape: (rows, C, B)
        X = block.reshape(-1, B).astype(np.float64, copy=False)

        # Remove NaNs/Infs if any (replace with row mean or mu)
        bad = ~np.isfinite(X)
        if bad.any():
            # Replace bad entries with the feature mean (mu) to avoid blowing up distances
            X[bad] = np.take(mu, np.where(bad)[1])

        Xc = X - mu
        # Mahalanobis: (Xc Σ^{-1}) ∘ Xc → sum along features
        M = Xc @ inv_cov
        d = np.einsum('ij,ij->i', M, Xc)  # efficient row-wise dot
        scores[r0:r1, :] = d.reshape(r1 - r0, C)

    # Clip tiny negatives due to floating error
    np.maximum(scores, 0, out=scores)
    return scores


def choose_pseudo_rgb_bands(B, user_choice=None):
    if user_choice is not None:
        return user_choice
    # Spread 3 bands across the spectrum
    r = int(round(0.8 * (B - 1)))
    g = int(round(0.5 * (B - 1)))
    b = int(round(0.2 * (B - 1)))
    # Ensure unique & ordered-ish
    bands = sorted({r, g, b}, reverse=True)
    # Return as (R, G, B) indices
    if len(bands) < 3:
        # Fall back to first 3 bands
        bands = [min(B - 1, 2), min(B - 1, 1), 0]
    return (bands[0], bands[1], bands[2])


def minmax_norm(img):
    a = np.nanmin(img)
    b = np.nanmax(img)
    if b <= a:
        return np.zeros_like(img)
    out = (img - a) / (b - a)
    return out


# ---- MAIN FLOW ----
def main():
    convert_images_in_dir(DATA_DIR)
    hdr_path = find_envi_hdr(DATA_DIR)
    print(f"[i] Using ENVI header: {hdr_path}")

    arr = open_envi_cube(hdr_path)   # memmap (R, C, B)
    R, C, B = arr.shape
    print(f"[i] Cube shape: rows={R}, cols={C}, bands={B}")

    # 1) Sample pixels for stats
    print(f"[i] Sampling up to {MAX_SAMPLE_PIXELS} pixels for mean/cov …")
    Xsamp = sample_pixels_for_stats(arr, MAX_SAMPLE_PIXELS)
    print(f"[i] Sampled {Xsamp.shape[0]} pixels.")

    # 2) Mean & covariance (robust)
    print("[i] Computing robust mean & covariance …")
    mu, Sigma = robust_mean_cov(Xsamp)

    # 3) Invert covariance (pinv for stability)
    print("[i] Inverting covariance …")
    inv_cov = np.linalg.pinv(Sigma)

    # 4) RX scores (chunked)
    print("[i] Computing RX anomaly scores (chunked) …")
    scores = rx_scores_chunked(arr, mu, inv_cov, chunk_rows=CHUNK_ROWS)

    # 5) Normalize scores for visualization
    scores_vis = minmax_norm(scores)

    # 6) Threshold for binary mask
    th = np.percentile(scores, ANOMALY_PERCENTILE)
    mask = (scores >= th).astype(np.uint8)

    # 7) Pseudo-RGB (for context)
    rb, gb, bb = choose_pseudo_rgb_bands(B, PSEUDO_RGB_BANDS)
    pseudo = np.stack([
        minmax_norm(arr[:, :, rb]),
        minmax_norm(arr[:, :, gb]),
        minmax_norm(arr[:, :, bb])
    ], axis=-1)

    # 8) Save outputs next to the data folder
    base = os.path.splitext(os.path.basename(hdr_path))[0]
    out_scores_npy = os.path.join(DATA_DIR, f"{base}_RX_scores.npy")
    out_heatmap_png = os.path.join(DATA_DIR, f"{base}_RX_heatmap.png")
    out_mask_png = os.path.join(DATA_DIR, f"{base}_RX_mask_{int(ANOMALY_PERCENTILE*10)/10:.1f}pct.png")
    out_pseudorgb_png = os.path.join(DATA_DIR, f"{base}_pseudoRGB.png")

    np.save(out_scores_npy, scores)
    print(f"[i] Saved raw RX scores: {out_scores_npy}")

    # Save pseudo-RGB and heatmaps
    plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(pseudo, 0, 1))
    plt.axis('off')
    plt.title("Pseudo-RGB (for context)")
    plt.tight_layout()
    plt.savefig(out_pseudorgb_png, dpi=200)
    plt.close()
    print(f"[i] Saved pseudo-RGB: {out_pseudorgb_png}")

    plt.figure(figsize=(10, 10))
    plt.imshow(scores_vis, cmap='inferno')
    plt.axis('off')
    plt.title("RX Anomaly Heatmap (normalized)")
    plt.tight_layout()
    plt.savefig(out_heatmap_png, dpi=200)
    plt.close()
    print(f"[i] Saved anomaly heatmap: {out_heatmap_png}")

    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.title(f"Binary Anomaly Mask (≥ {ANOMALY_PERCENTILE}th pct)")
    plt.tight_layout()
    plt.savefig(out_mask_png, dpi=200)
    plt.close()
    print(f"[i] Saved anomaly mask: {out_mask_png}")

    print("[✓] Done.")


if __name__ == "__main__":
    main()
