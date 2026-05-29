"""
attack.py — DLG Gradient Leakage Attack with Hyperparameter Analysis
Step 2 of the Final Project

Key additions based on TA feedback:
  1. Fixed random seeds — reproducible results
  2. Hyperparameter sweep — shows how lr and iterations affect reconstruction
  3. Multiple seed trials — shows attack stability/instability
  4. Saves a summary table of all results

Reference: Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019

Usage:
  python attack.py                        # standard attack, 6 images
  python attack.py --mode sweep           # hyperparameter sweep
  python attack.py --mode seeds           # seed stability analysis
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import TinyCNN


CIFAR10_CLASSES = ['airplane','car','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int):
    """Fix all random sources for reproducible attack results."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────
# DLG core
# ─────────────────────────────────────────────

def compute_gradients(model, criterion, image, label):
    """Compute real gradients — what the server intercepts from the client."""
    model.zero_grad()
    loss  = criterion(model(image), label)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return [g.detach() for g in grads]


def gradient_distance(dummy_grads, real_grads):
    """||∇W' - ∇W||² — the DLG optimization objective."""
    return sum(((dg - rg)**2).sum() for dg, rg in zip(dummy_grads, real_grads))


def dlg_attack(model, real_grads, device, seed=42,
               num_classes=10, iterations=300, lr=1.0,
               snapshot_iters=None):
    """
    DLG attack with fixed seed for reproducibility.

    Args:
        seed:       Random seed — controls dummy image initialization
        lr:         L-BFGS learning rate — key hyperparameter
        iterations: Max optimization steps

    Returns:
        best_img:   Best reconstructed image found
        pred_label: Predicted label from dummy_label
        history:    List of (iter, grad_loss) for convergence plot
        snapshots:  List of (iter, image) for progress figure
        final_loss: Final gradient distance (lower = better attack)
    """
    set_seed(seed)
    criterion   = nn.CrossEntropyLoss()

    dummy_image = torch.randn(1, 3, 32, 32, requires_grad=True, device=device)
    dummy_label = torch.randn(1, num_classes, requires_grad=True, device=device)
    optimizer   = torch.optim.LBFGS([dummy_image, dummy_label], lr=lr)

    if snapshot_iters is None:
        snapshot_iters = set()

    history   = []
    snapshots = []
    best_img  = dummy_image.detach().cpu().clone()
    best_loss = float('inf')

    for i in range(iterations):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()
            pred        = model(dummy_image)
            soft_label  = torch.softmax(dummy_label, dim=-1)
            loss        = criterion(pred, soft_label)
            dummy_grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            grad_diff = gradient_distance(dummy_grads, real_grads)
            grad_diff.backward()
            return grad_diff

        loss     = optimizer.step(closure)
        loss_val = loss.item()

        if np.isnan(loss_val) or np.isinf(loss_val):
            break

        history.append((i, loss_val))

        if i in snapshot_iters:
            snapshots.append((i, dummy_image.detach().cpu().clone()))

        if loss_val < best_loss:
            best_loss = loss_val
            best_img  = dummy_image.detach().cpu().clone()

        if i % 50 == 0 or i == iterations - 1:
            print(f"  [attack] iter {i:>4}/{iterations} | "
                  f"grad loss: {loss_val:.6f}")

        if loss_val < 1e-6:
            print(f"  [attack] converged at iter {i}!")
            snapshots.append((i, dummy_image.detach().cpu().clone()))
            break

    pred_label = torch.argmax(dummy_label, dim=-1).item()
    return best_img, pred_label, history, snapshots, best_loss


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────

def to_numpy(t):
    """Denormalize CIFAR-10 tensor → displayable numpy HWC array."""
    img = t.squeeze().cpu() * STD + MEAN
    return np.clip(img.permute(1,2,0).numpy(), 0, 1)

def compute_psnr(orig, recon):
    mse = np.mean((to_numpy(orig) - to_numpy(recon))**2)
    return 10 * np.log10(1.0 / (mse + 1e-10))

def save_comparison(originals, recons, true_labels, pred_labels,
                    psnrs, path):
    n   = len(originals)
    fig = plt.figure(figsize=(3 * n, 6))
    fig.suptitle(
        "DLG Attack: Original vs Reconstructed\n(CIFAR-10 images from FL client)",
        fontsize=13, y=1.02
    )
    gs = gridspec.GridSpec(2, n, hspace=0.5, wspace=0.3)
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(to_numpy(originals[i]))
        ax.set_title(f"Original\n{CIFAR10_CLASSES[true_labels[i]]}", fontsize=9)
        ax.axis('off')

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(to_numpy(recons[i]))
        ax.set_title(f"Reconstructed\n(pred: {CIFAR10_CLASSES[pred_labels[i]]})\n"
                     f"PSNR: {psnrs[i]:.1f} dB", fontsize=8)
        ax.axis('off')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[attack] comparison saved → {path}")

def save_progress(snapshots, true_image, true_label, path):
    n   = len(snapshots)
    fig = plt.figure(figsize=(2.5*(n+1), 3.5))
    fig.suptitle(f"Reconstruction Progress — true: {CIFAR10_CLASSES[true_label]}",
                 fontsize=12)
    gs = gridspec.GridSpec(1, n+1, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(to_numpy(true_image))
    ax.set_title("Original", fontsize=9)
    ax.axis('off')
    for i, (iteration, img) in enumerate(snapshots):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(to_numpy(img))
        ax.set_title(f"iter {iteration}", fontsize=9)
        ax.axis('off')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[attack] progress figure saved → {path}")

def save_convergence(histories, labels, path, title="DLG Attack Convergence"):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#D85A30','#1D9E75','#185FA5','#BA7517','#A32D2D','#444441']
    for (history, label, color) in zip(histories, labels, colors):
        iters  = [h[0] for h in history]
        losses = [h[1] for h in history]
        ax.plot(iters, losses, label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient distance ||∇W' − ∇W||²")
    ax.set_title(title)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[attack] convergence plot saved → {path}")


# ─────────────────────────────────────────────
# Analysis modes
# ─────────────────────────────────────────────

def run_standard(model, test_loader, device, args):
    """Standard attack: fixed seed, 6 images, save all figures."""
    print("\n[mode] standard attack (fixed seed)\n")
    os.makedirs(args.output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    originals, recons, true_lbls, pred_lbls, psnrs = [], [], [], [], []
    all_histories = []

    for idx, (image, label) in enumerate(test_loader):
        if idx >= args.num_images:
            break
        image = image.to(device)
        label = label.to(device)
        print(f"[attack] image {idx+1}/{args.num_images} | "
              f"true: {CIFAR10_CLASSES[label.item()]}")

        real_grads = compute_gradients(model, criterion, image, label)
        snap_iters = {0,10,30,50,100,200,args.iterations-1} if idx == 0 else set()

        dummy_img, pred_lbl, history, snapshots, final_loss = dlg_attack(
            model, real_grads, device,
            seed=args.seed,
            iterations=args.iterations,
            lr=args.lr,
            snapshot_iters=snap_iters,
        )

        psnr = compute_psnr(image.cpu(), dummy_img)
        originals.append(image.cpu())
        recons.append(dummy_img)
        true_lbls.append(label.item())
        pred_lbls.append(pred_lbl)
        psnrs.append(psnr)
        all_histories.append(history)
        print(f"  [result] final grad loss: {final_loss:.6f} | "
              f"PSNR: {psnr:.2f} dB | pred: {CIFAR10_CLASSES[pred_lbl]}\n")

        if idx == 0 and snapshots:
            save_progress(snapshots, image.cpu(), label.item(),
                os.path.join(args.output_dir, 'attack_progress.png'))

    save_comparison(originals, recons, true_lbls, pred_lbls, psnrs,
                    os.path.join(args.output_dir, 'attack_results.png'))
    save_convergence(
        all_histories,
        [CIFAR10_CLASSES[l] for l in true_lbls],
        os.path.join(args.output_dir, 'attack_convergence.png'),
        title="DLG Convergence — 6 CIFAR-10 images (fixed seed)"
    )


def run_seed_analysis(model, test_loader, device, args):
    """
    Seed stability analysis: attack same image with 5 different seeds.
    Shows how unstable DLG is — directly addresses TA's question about
    whether results can be stably reproduced.
    """
    print("\n[mode] seed stability analysis\n")
    os.makedirs(args.output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    seeds     = [0, 7, 42, 123, 999]

    # Use first image only
    image, label = next(iter(test_loader))
    image = image.to(device)
    label = label.to(device)
    print(f"[seed analysis] attacking: {CIFAR10_CLASSES[label.item()]}")
    print(f"[seed analysis] seeds: {seeds}\n")

    real_grads = compute_gradients(model, criterion, image, label)

    recons, histories, final_losses, psnrs = [], [], [], []

    for seed in seeds:
        print(f"── seed {seed} ──")
        dummy_img, _, history, _, final_loss = dlg_attack(
            model, real_grads, device,
            seed=seed, iterations=args.iterations, lr=args.lr,
        )
        psnr = compute_psnr(image.cpu(), dummy_img)
        recons.append(dummy_img)
        histories.append(history)
        final_losses.append(final_loss)
        psnrs.append(psnr)
        print(f"  final loss: {final_loss:.6f} | PSNR: {psnr:.2f} dB\n")

    # Figure: original + all seed reconstructions
    fig = plt.figure(figsize=(3*(len(seeds)+1), 4))
    fig.suptitle(
        f"Seed Stability Analysis — true: {CIFAR10_CLASSES[label.item()]}\n"
        f"Shows DLG instability across different random initializations",
        fontsize=11
    )
    gs = gridspec.GridSpec(1, len(seeds)+1, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(to_numpy(image.cpu()))
    ax.set_title("Original", fontsize=9)
    ax.axis('off')
    for i, (seed, recon, psnr, loss) in enumerate(
            zip(seeds, recons, psnrs, final_losses)):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(to_numpy(recon))
        ax.set_title(f"seed={seed}\nPSNR:{psnr:.1f}dB\nloss:{loss:.4f}", fontsize=8)
        ax.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'seed_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[seed analysis] saved → seed_analysis.png")

    # Convergence curves for all seeds
    save_convergence(
        histories,
        [f"seed={s}" for s in seeds],
        os.path.join(args.output_dir, 'seed_convergence.png'),
        title=f"DLG Convergence — same image, different random seeds\n"
              f"(true: {CIFAR10_CLASSES[label.item()]})"
    )

    # Print summary table
    print("\n── Summary ──")
    print(f"{'Seed':<8} {'Final Loss':<14} {'PSNR (dB)':<12}")
    print("-" * 34)
    for seed, loss, psnr in zip(seeds, final_losses, psnrs):
        print(f"{seed:<8} {loss:<14.6f} {psnr:<12.2f}")
    print(f"\nPSNR std dev: {np.std(psnrs):.2f} dB  "
          f"(higher = more unstable across seeds)")


def run_lr_sweep(model, test_loader, device, args):
    """
    Learning rate sweep: attack same image with different LR values.
    Shows how LR affects convergence — addresses TA's question about
    hyperparameter influence.
    """
    print("\n[mode] learning rate sweep\n")
    os.makedirs(args.output_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    lrs       = [0.01, 0.1, 0.5, 1.0, 2.0]

    image, label = next(iter(test_loader))
    image = image.to(device)
    label = label.to(device)
    print(f"[lr sweep] attacking: {CIFAR10_CLASSES[label.item()]}")
    print(f"[lr sweep] LR values: {lrs}\n")

    real_grads = compute_gradients(model, criterion, image, label)
    recons, histories, final_losses, psnrs = [], [], [], []

    for lr in lrs:
        print(f"── lr={lr} ──")
        dummy_img, _, history, _, final_loss = dlg_attack(
            model, real_grads, device,
            seed=args.seed, iterations=args.iterations, lr=lr,
        )
        psnr = compute_psnr(image.cpu(), dummy_img)
        recons.append(dummy_img)
        histories.append(history)
        final_losses.append(final_loss)
        psnrs.append(psnr)
        print(f"  final loss: {final_loss:.6f} | PSNR: {psnr:.2f} dB\n")

    # Figure
    fig = plt.figure(figsize=(3*(len(lrs)+1), 4))
    fig.suptitle(
        f"Learning Rate Sweep — true: {CIFAR10_CLASSES[label.item()]}\n"
        f"Shows how LR affects DLG reconstruction quality",
        fontsize=11
    )
    gs = gridspec.GridSpec(1, len(lrs)+1, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(to_numpy(image.cpu()))
    ax.set_title("Original", fontsize=9)
    ax.axis('off')
    for i, (lr, recon, psnr, loss) in enumerate(
            zip(lrs, recons, psnrs, final_losses)):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(to_numpy(recon))
        ax.set_title(f"lr={lr}\nPSNR:{psnr:.1f}dB\nloss:{loss:.4f}", fontsize=8)
        ax.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'lr_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[lr sweep] saved → lr_sweep.png")

    save_convergence(
        histories,
        [f"lr={lr}" for lr in lrs],
        os.path.join(args.output_dir, 'lr_convergence.png'),
        title=f"DLG Convergence — same image, different learning rates\n"
              f"(true: {CIFAR10_CLASSES[label.item()]})"
    )

    print("\n── Summary ──")
    print(f"{'LR':<8} {'Final Loss':<14} {'PSNR (dB)':<12}")
    print("-" * 34)
    for lr, loss, psnr in zip(lrs, final_losses, psnrs):
        print(f"{lr:<8} {loss:<14.6f} {psnr:<12.2f}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f" DLG Attack — CIFAR-10 FL System")
    print(f" mode: {args.mode} | seed: {args.seed} | "
          f"lr: {args.lr} | iters: {args.iterations}")
    print(f" device: {device}")
    print(f"{'='*50}")

    # Load FL-trained model from Step 1
    model      = TinyCNN(num_classes=10).to(device)
    model_path = args.model_path if os.path.exists(args.model_path) else 'global_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Run main.py first to train the FL model.")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[attack] loaded FL model: {model_path}\n")

    # Load CIFAR-10 test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    test_set    = datasets.CIFAR10('./data/cifar10', train=False,
                                    download=True, transform=test_transform)
    # Fix shuffle seed so same images are selected every run
    g = torch.Generator()
    g.manual_seed(args.seed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, generator=g)

    if args.mode == 'standard':
        run_standard(model, test_loader, device, args)
    elif args.mode == 'seeds':
        run_seed_analysis(model, test_loader, device, args)
    elif args.mode == 'sweep':
        run_lr_sweep(model, test_loader, device, args)
    else:
        print(f"Unknown mode: {args.mode}. Use: standard, seeds, sweep")

    print("\n[attack] done! Ready for Step 3 — differential privacy defense")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       type=str,   default='standard',
                        choices=['standard','seeds','sweep'],
                        help='standard: normal attack | seeds: seed analysis | sweep: lr sweep')
    parser.add_argument('--model_path', type=str,   default='global_model_best.pth')
    parser.add_argument('--output_dir', type=str,   default='./attack_output')
    parser.add_argument('--num_images', type=int,   default=6)
    parser.add_argument('--iterations', type=int,   default=300)
    parser.add_argument('--lr',         type=float, default=1.0)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()
    main(args)