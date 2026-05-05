# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
 
from step1_deidentification import pixelize, gaussian_blur, compute_metrics
from step2_attack import FaceDataset, FaceCNN, load_variant_dataset, train_model, evaluate_model

# ─────────────────────────────────────────────
# DIFFERENTIAL PRIVACY NOISE
# ─────────────────────────────────────────────
 
def laplace_noise(image, epsilon, sensitivity=255.0):
    """
    Add Laplace noise for differential privacy.
 
    The Laplace mechanism adds noise drawn from Laplace(0, sensitivity/epsilon).
    - sensitivity: max change one pixel can have = 255 (full range)
    - epsilon: privacy budget — smaller = more noise = stronger privacy
 
    Result is clamped to valid pixel range [0, 255].
    """
    scale = sensitivity / epsilon  # Laplace scale parameter (b)
    noise = np.random.laplace(loc=0.0, scale=scale, size=image.shape)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy
 
 
def dp_pixelize(image, block_size=16, epsilon=1.0):
    """
    DP-Pix: pixelize first, then add Laplace noise.
    Matches the DP-Pix method from the homework reference paper.
    """
    pixelized = pixelize(image, block_size)
    return laplace_noise(pixelized, epsilon)
 
 
def dp_blur(image, kernel_size=99, epsilon=1.0):
    """
    DP-Blur: gaussian blur first, then add Laplace noise.
    Matches the DP-Blur method from the homework reference paper.
    """
    blurred = gaussian_blur(image, kernel_size)
    return laplace_noise(blurred, epsilon)
 
 
# ─────────────────────────────────────────────
# SAVE DP PROCESSED DATASET
# ─────────────────────────────────────────────
 
def save_dp_dataset(images, labels, epsilons=[0.1, 0.5, 1.0],
                    block_size=16, kernel_size=99,
                    out_dir="processed_faces"):
    """
    Save DP-protected image variants for all epsilon values.
    Folder structure added to processed_faces/:
        dp_pix_e0.1/   dp_pix_e0.5/   dp_pix_e1.0/
        dp_blur_e0.1/  dp_blur_e0.5/  dp_blur_e1.0/
    """
    for epsilon in epsilons:
        eps_str = str(epsilon)
 
        # DP-Pix
        pix_dir = os.path.join(out_dir, f"dp_pix_e{eps_str}")
        os.makedirs(pix_dir, exist_ok=True)
        for i, (img, label) in enumerate(zip(images, labels)):
            proc = dp_pixelize(img, block_size=block_size, epsilon=epsilon)
            cv2.imwrite(os.path.join(pix_dir, f"s{label:02d}_img{i:03d}.png"), proc)
 
        # DP-Blur
        blur_dir = os.path.join(out_dir, f"dp_blur_e{eps_str}")
        os.makedirs(blur_dir, exist_ok=True)
        for i, (img, label) in enumerate(zip(images, labels)):
            proc = dp_blur(img, kernel_size=kernel_size, epsilon=epsilon)
            cv2.imwrite(os.path.join(blur_dir, f"s{label:02d}_img{i:03d}.png"), proc)
 
    print(f"DP dataset saved to '{out_dir}/'")
    print(f"  Variants: dp_pix and dp_blur for epsilon = {epsilons}")
 
 
# ─────────────────────────────────────────────
# RUN CNN ATTACK ON DP VARIANTS
# ─────────────────────────────────────────────
 
def run_dp_attack(processed_dir="processed_faces", epsilons=[0.1, 0.5, 1.0],
                  num_epochs=50, batch_size=32, device=None):
    """
    Train and evaluate CNN attack on all DP-protected variants,
    plus the non-private baselines (NP-Pix b=16, NP-Blur k=99).
 
    Returns:
        results: dict mapping variant name -> {'top1': float, 'top5': float}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
 
    # Build list of variants to evaluate
    variants = ["pix_b16", "blur_k99"]  # non-private baselines
    for epsilon in epsilons:
        eps_str = str(epsilon)
        variants.append(f"dp_pix_e{eps_str}")
        variants.append(f"dp_blur_e{eps_str}")
 
    results = {}
 
    for variant in variants:
        variant_dir = os.path.join(processed_dir, variant)
        if not os.path.exists(variant_dir):
            print(f"Skipping '{variant}' — folder not found.")
            continue
 
        print(f"── Training on: {variant} ──────────────────────")
        train_dataset, test_dataset = load_variant_dataset(variant_dir)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
 
        model = FaceCNN(num_classes=40)
        train_model(model, train_loader, device, num_epochs=num_epochs)
 
        top1, top5 = evaluate_model(model, test_loader, device)
        results[variant] = {"top1": top1, "top5": top5}
        print(f"  -> Top-1: {top1:.2f}%  |  Top-5: {top5:.2f}%\n")
 
    return results
 
 
# ─────────────────────────────────────────────
# METRICS VS EPSILON
# ─────────────────────────────────────────────
 
def compute_dp_metrics(images, epsilons=[0.1, 0.5, 1.0],
                       block_size=16, kernel_size=99):
    """
    Compute average MSE and SSIM between original images and
    DP-protected versions at each epsilon value.
 
    Returns:
        pix_mse, pix_ssim: lists of floats (one per epsilon)
        blur_mse, blur_ssim: lists of floats (one per epsilon)
    """
    pix_mse, pix_ssim = [], []
    blur_mse, blur_ssim = [], []
 
    for epsilon in epsilons:
        p_mse_vals, p_ssim_vals = [], []
        b_mse_vals, b_ssim_vals = [], []
 
        for img in images:
            dp_pix_img  = dp_pixelize(img, block_size=block_size, epsilon=epsilon)
            dp_blur_img = dp_blur(img, kernel_size=kernel_size, epsilon=epsilon)
 
            pm, ps = compute_metrics(img, dp_pix_img)
            bm, bs = compute_metrics(img, dp_blur_img)
 
            p_mse_vals.append(pm)
            p_ssim_vals.append(ps)
            b_mse_vals.append(bm)
            b_ssim_vals.append(bs)
 
        pix_mse.append(np.mean(p_mse_vals))
        pix_ssim.append(np.mean(p_ssim_vals))
        blur_mse.append(np.mean(b_mse_vals))
        blur_ssim.append(np.mean(b_ssim_vals))
 
    return pix_mse, pix_ssim, blur_mse, blur_ssim
 
 
# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
 
def plot_dp_metrics(images, epsilons=[0.1, 0.5, 1.0],
                    block_size=16, kernel_size=99,
                    save_path="step3_dp_metrics.png"):
    """
    Plot MSE and SSIM vs epsilon for DP-Pix and DP-Blur.
    Also shows NP (non-private) baseline as a flat dashed line.
    Mirrors Figure 3 and Figure 4 from the homework reference paper.
    """
    from step1_deidentification import compute_metrics as cm
 
    pix_mse, pix_ssim, blur_mse, blur_ssim = compute_dp_metrics(
        images, epsilons, block_size, kernel_size
    )
 
    # Non-private baselines (flat reference lines)
    np_pix_mse  = np.mean([cm(img, pixelize(img, block_size))[0] for img in images])
    np_pix_ssim = np.mean([cm(img, pixelize(img, block_size))[1] for img in images])
    np_blur_mse  = np.mean([cm(img, gaussian_blur(img, kernel_size))[0] for img in images])
    np_blur_ssim = np.mean([cm(img, gaussian_blur(img, kernel_size))[1] for img in images])
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Step 3: Utility vs Privacy Budget (epsilon)\n"
                 "Smaller epsilon = stronger privacy = more distortion",
                 fontsize=13, fontweight='bold')
 
    # ── DP-Pix MSE ──
    axes[0][0].plot(epsilons, pix_mse, 'g^-', linewidth=2, markersize=8, label="DP-Pix")
    axes[0][0].axhline(np_pix_mse, color='b', linestyle='--', linewidth=1.5, label="NP-Pix")
    axes[0][0].set_title("MSE — Pixelization")
    axes[0][0].set_xlabel("epsilon")
    axes[0][0].set_ylabel("MSE (higher = more distortion)")
    axes[0][0].set_xscale('log')
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)
 
    # ── DP-Pix SSIM ──
    axes[0][1].plot(epsilons, pix_ssim, 'g^-', linewidth=2, markersize=8, label="DP-Pix")
    axes[0][1].axhline(np_pix_ssim, color='b', linestyle='--', linewidth=1.5, label="NP-Pix")
    axes[0][1].set_title("SSIM — Pixelization")
    axes[0][1].set_xlabel("epsilon")
    axes[0][1].set_ylabel("SSIM (lower = more distortion)")
    axes[0][1].set_xscale('log')
    axes[0][1].legend()
    axes[0][1].grid(True, alpha=0.3)
 
    # ── DP-Blur MSE ──
    axes[1][0].plot(epsilons, blur_mse, 'r^-', linewidth=2, markersize=8, label="DP-Blur")
    axes[1][0].axhline(np_blur_mse, color='orange', linestyle='--', linewidth=1.5, label="NP-Blur")
    axes[1][0].set_title("MSE — Gaussian Blur")
    axes[1][0].set_xlabel("epsilon")
    axes[1][0].set_ylabel("MSE (higher = more distortion)")
    axes[1][0].set_xscale('log')
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)
 
    # ── DP-Blur SSIM ──
    axes[1][1].plot(epsilons, blur_ssim, 'r^-', linewidth=2, markersize=8, label="DP-Blur")
    axes[1][1].axhline(np_blur_ssim, color='orange', linestyle='--', linewidth=1.5, label="NP-Blur")
    axes[1][1].set_title("SSIM — Gaussian Blur")
    axes[1][1].set_xlabel("epsilon")
    axes[1][1].set_ylabel("SSIM (lower = more distortion)")
    axes[1][1].set_xscale('log')
    axes[1][1].legend()
    axes[1][1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
 
 
def plot_dp_visual_comparison(images, epsilons=[0.1, 0.5, 1.0],
                               block_size=16, kernel_size=99,
                               save_path="step3_dp_visual.png"):
    """
    Visual grid showing original, NP-Pix, DP-Pix variants,
    NP-Blur, and DP-Blur variants for 3 sample faces.
    Mirrors Table 2 from the homework reference paper.
    """
    samples = images[:3]
 
    # Columns: Orig | NP-Pix | DP-Pix e=0.1 | DP-Pix e=0.5 | DP-Pix e=1 |
    #          NP-Blur | DP-Blur e=0.1 | DP-Blur e=0.5 | DP-Blur e=1
    col_labels = (
        ["Original", "NP-Pix"] +
        [f"DP-Pix\ne={e}" for e in epsilons] +
        ["NP-Blur"] +
        [f"DP-Blur\ne={e}" for e in epsilons]
    )
    n_cols = len(col_labels)
    n_rows = len(samples)
 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    fig.suptitle("Step 3: Visual Comparison — NP vs DP Obfuscation",
                 fontsize=13, fontweight='bold')
 
    for row, img in enumerate(samples):
        processed = (
            [img, pixelize(img, block_size)] +
            [dp_pixelize(img, block_size, e) for e in epsilons] +
            [gaussian_blur(img, kernel_size)] +
            [dp_blur(img, kernel_size, e) for e in epsilons]
        )
        for col, (proc_img, label) in enumerate(zip(processed, col_labels)):
            ax = axes[row][col] if n_rows > 1 else axes[col]
            ax.imshow(proc_img, cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            if row == 0:
                ax.set_title(label, fontsize=8)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
 
 
def plot_dp_attack_results(results, epsilons=[0.1, 0.5, 1.0],
                            save_path="step3_attack_results.png"):
    """
    Bar chart comparing Top-1 accuracy across NP and DP variants.
    Shows how DP noise reduces re-identification accuracy.
    """
    # Build display-friendly labels and group by method
    pix_labels = ["NP-Pix\n(b=16)"] + [f"DP-Pix\ne={e}" for e in epsilons]
    blur_labels = ["NP-Blur\n(k=99)"] + [f"DP-Blur\ne={e}" for e in epsilons]
 
    pix_keys = ["pix_b16"] + [f"dp_pix_e{e}" for e in epsilons]
    blur_keys = ["blur_k99"] + [f"dp_blur_e{e}" for e in epsilons]
 
    pix_top1  = [results.get(k, {}).get("top1", 0) for k in pix_keys]
    blur_top1 = [results.get(k, {}).get("top1", 0) for k in blur_keys]
 
    x = np.arange(len(pix_labels))
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, pix_top1,  width, label="Pixelization", color="steelblue")
    bars2 = ax.bar(x + width/2, blur_top1, width, label="Gaussian Blur", color="coral")
 
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8)
 
    ax.set_title("Step 3: Re-Identification Accuracy — NP vs DP", fontsize=13, fontweight='bold')
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(pix_labels)
    ax.set_ylim(0, 110)
    ax.axhline(y=2.5, color='gray', linestyle='--', linewidth=1, label="Random baseline (2.5%)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
 
 
def print_dp_results_table(results, epsilons=[0.1, 0.5, 1.0]):
    """
    Print accuracy table matching the format in the homework slides (Table 1).
 
    Format:
    Dataset | Random | NP-Pix | DP-Pix e=0.1 | e=0.5 | e=1 | NP-Blur | DP-Blur e=0.1 | e=0.5 | e=1
    """
    print("\n" + "="*90)
    header = (f"{'':10} {'Random':>8} {'NP-Pix':>8} " +
              " ".join([f"{'DP-Pix e='+str(e):>12}" for e in epsilons]) +
              f" {'NP-Blur':>8} " +
              " ".join([f"{'DP-Blur e='+str(e):>13}" for e in epsilons]))
    print(header)
    print("="*90)
 
    def get(key, metric):
        return results.get(key, {}).get(metric, 0.0)
 
    for metric, label in [("top1", "Top-1 (%)"), ("top5", "Top-5 (%)")]:
        row = f"{label:10} {'2.50':>8} {get('pix_b16', metric):>8.2f} "
        row += " ".join([f"{get(f'dp_pix_e{e}', metric):>12.2f}" for e in epsilons])
        row += f" {get('blur_k99', metric):>8.2f} "
        row += " ".join([f"{get(f'dp_blur_e{e}', metric):>13.2f}" for e in epsilons])
        print(row)
 
    print("="*90)