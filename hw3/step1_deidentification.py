# Header

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from sklearn.datasets import fetch_olivetti_faces

# ─────────────────────────────────────────────
# Download and Load AT&T Dataset
# ─────────────────────────────────────────────

def load_images():
    """
    Load the Olivetti (AT&T) face dataset via scikit-learn.
    - 40 subjects, 10 images each = 400 images total
    - Each image is 64x64 grayscale
    Returns:
        images: list of uint8 numpy arrays (64x64)
        labels: list of integer subject IDs (0-39)
    """
    print("Loading Olivetti (AT&T) face dataset...")
    dataset = fetch_olivetti_faces(shuffle=False)
 
    # dataset.images shape: (400, 64, 64), float32 in range [0, 1]
    # Convert to uint8 [0, 255] for OpenCV compatibility
    images = [
        (img * 255).astype(np.uint8)
        for img in dataset.images
    ]
    labels = list(dataset.target)
 
    print(f"Loaded {len(images)} images — 40 subjects x 10 images, size 64x64")
    return images, labels

# ─────────────────────────────────────────────
# De-Identification Methods
# ─────────────────────────────────────────────

'''
Divide image to bxb blocks
replace each block with its mean pixel value
Smaller b -> better visual quality
'''
def pixelize(image, block_size):
  h, w = image.shape[:2]
  result = image.copy()

  for y in range(0, h, block_size):
    for x in range(0, w, block_size):
      block = result[y:y+block_size, x:x+block_size]
      block_mean = np.mean(block)
      result[y:y+block_size, x:x+block_size] = block_mean

  return result

'''
Apply k x k gaussian filter
kernel size must be odd
larger k -> stronger blur -> lower visual quality
'''
def gaussian_blur(image, kernel_size):
  k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
  return cv2.GaussianBlur(image, (k, k), 0)

# ─────────────────────────────────────────────
# Quality Metrics
# ─────────────────────────────────────────────

def compute_metrics(original, processed):
  mse_val = mse(original, processed)
  ssim_val = ssim (original, processed, data_range=255)
  return mse_val, ssim_val

# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_pixelization_results(images, block_sizes=[4, 8, 16], save_path='step1_pixelization.png'):
  'Show original vs pizelized images for different block sizes'
  samples = images[:3]
  n_cols = 1 + len(block_sizes)
  n_rows = len(samples)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
  fig.suptitle("Pixelization De-Identification\n(smaller b = better quality)",
               fontsize = 14, fontweight='bold')

  col_titles = ['Original'] + [f"b={b}" for b in block_sizes]

  for row, img in enumerate(samples):
    processed = [img] + [pixelize(img, b) for b in block_sizes]
    for col, (proc_img, title) in enumerate(zip(processed, col_titles)):
      ax = axes[row][col] if n_rows > 1 else axes[col]
      ax.imshow(proc_img, cmap='gray', vmin=0, vmax=255)
      ax.axis('off')
      if row == 0:
        ax.set_title(title, fontsize=11)
      if col > 0:
        mse_v, ssim_v = compute_metrics(img, proc_img)
        ax.set_xlabel(f"MSE={mse_v:.1f}\nSSIM={ssim_v:.3f}", fontsize=8)

  fig.tight_layout()
  plt.savefig(save_path, dpi=150, bbox_inches='tight')
  plt.close()
  print(f"Saved visualization to {save_path}")

def plot_blur_results(images, kernel_sizes=[15, 45, 99], save_path="step1_gaussian_blur.png"):
    """Show original vs Gaussian blurred at different kernel sizes for 3 sample images."""
    samples = images[:3]
    n_cols = 1 + len(kernel_sizes)
    n_rows = len(samples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    fig.suptitle("Gaussian Blur De-Identification\n(larger k = stronger blur)", fontsize=14, fontweight='bold')

    col_titles = ["Original"] + [f"k={k}" for k in kernel_sizes]

    for row, img in enumerate(samples):
        processed = [img] + [gaussian_blur(img, k) for k in kernel_sizes]
        for col, (proc_img, title) in enumerate(zip(processed, col_titles)):
            ax = axes[row][col] if n_rows > 1 else axes[col]
            ax.imshow(proc_img, cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col > 0:
                mse_v, ssim_v = compute_metrics(img, proc_img)
                ax.set_xlabel(f"MSE={mse_v:.1f}\nSSIM={ssim_v:.3f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_metrics_summary(images, block_sizes=[2, 4, 8, 16], kernel_sizes=[15, 45, 99],
                          save_path="step1_metrics.png"):
    """Plot MSE and SSIM vs parameter value for both methods."""
    # Compute average metrics across all images
    pix_mse, pix_ssim = [], []
    for b in block_sizes:
        mse_vals, ssim_vals = [], []
        for img in images:
            proc = pixelize(img, b)
            m, s = compute_metrics(img, proc)
            mse_vals.append(m)
            ssim_vals.append(s)
        pix_mse.append(np.mean(mse_vals))
        pix_ssim.append(np.mean(ssim_vals))

    blur_mse, blur_ssim = [], []
    for k in kernel_sizes:
        mse_vals, ssim_vals = [], []
        for img in images:
            proc = gaussian_blur(img, k)
            m, s = compute_metrics(img, proc)
            mse_vals.append(m)
            ssim_vals.append(s)
        blur_mse.append(np.mean(mse_vals))
        blur_ssim.append(np.mean(ssim_vals))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Step 1: Visual Quality Metrics", fontsize=14, fontweight='bold')

    # Pixelization MSE
    axes[0][0].plot(block_sizes, pix_mse, 'bo-', linewidth=2, markersize=8)
    axes[0][0].set_title("Pixelization — MSE vs Block Size (b)")
    axes[0][0].set_xlabel("Block size b")
    axes[0][0].set_ylabel("MSE (↑ = more distortion)")
    axes[0][0].grid(True, alpha=0.3)

    # Pixelization SSIM
    axes[0][1].plot(block_sizes, pix_ssim, 'bs-', linewidth=2, markersize=8)
    axes[0][1].set_title("Pixelization — SSIM vs Block Size (b)")
    axes[0][1].set_xlabel("Block size b")
    axes[0][1].set_ylabel("SSIM (↓ = more distortion)")
    axes[0][1].grid(True, alpha=0.3)

    # Gaussian Blur MSE
    axes[1][0].plot(kernel_sizes, blur_mse, 'ro-', linewidth=2, markersize=8)
    axes[1][0].set_title("Gaussian Blur — MSE vs Kernel Size (k)")
    axes[1][0].set_xlabel("Kernel size k")
    axes[1][0].set_ylabel("MSE (↑ = more distortion)")
    axes[1][0].grid(True, alpha=0.3)

    # Gaussian Blur SSIM
    axes[1][1].plot(kernel_sizes, blur_ssim, 'rs-', linewidth=2, markersize=8)
    axes[1][1].set_title("Gaussian Blur — SSIM vs Kernel Size (k)")
    axes[1][1].set_xlabel("Kernel size k")
    axes[1][1].set_ylabel("SSIM (↓ = more distortion)")
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ─────────────────────────────────────────────
# Save Processed Dataset for Step 2
# ─────────────────────────────────────────────

def save_processed_dataset(images, labels, out_dir="processed_faces"):
    """
    Save all processed versions of the dataset for use in Step 2 (CNN attack).
    Structure: processed_faces/original/, pix_b4/, pix_b8/, pix_b16/,
               blur_k15/, blur_k45/, blur_k99/
    """
    configs = {
        "original":  lambda img: img,
        "pix_b4":    lambda img: pixelize(img, 4),
        "pix_b8":    lambda img: pixelize(img, 8),
        "pix_b16":   lambda img: pixelize(img, 16),
        "blur_k15":  lambda img: gaussian_blur(img, 15),
        "blur_k45":  lambda img: gaussian_blur(img, 45),
        "blur_k99":  lambda img: gaussian_blur(img, 99),
    }

    for config_name, transform_fn in configs.items():
        config_dir = os.path.join(out_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        for i, (img, label) in enumerate(zip(images, labels)):
            proc = transform_fn(img)
            fname = f"s{label:02d}_img{i:03d}.png"
            cv2.imwrite(os.path.join(config_dir, fname), proc)

    print(f"\nProcessed dataset saved to '{out_dir}/' ({len(images)} images × {len(configs)} versions)")
    print("This folder will be used as input for Step 2 (CNN Attack).")

