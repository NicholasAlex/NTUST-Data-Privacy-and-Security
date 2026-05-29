"""
attack_mnist.py — DLG Attack Demo on MNIST
Step 2 of the Final Project: Gradient Leakage Attack

Follows the original paper's experimental setup:
  Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019

Why MNIST:
  - Small 28x28 images → fewer gradient dimensions → attack converges reliably
  - Simple 2-layer MLP → no BatchNorm issues → gradients cleanly invertible
  - This matches the original paper's demonstration setup exactly

The attack concept is identical regardless of dataset:
  1. Client trains on private image → sends gradients to server
  2. Attacker (server) starts from random noise
  3. Attacker optimizes noise until its gradients match intercepted gradients
  4. Reconstructed image ≈ original private image

Usage:
  python attack_mnist.py
  python attack_mnist.py --num_images 8 --iterations 300
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# Tiny MLP — matches original DLG paper setup
# ─────────────────────────────────────────────

class TinyMLP(nn.Module):
    """
    2-layer fully connected network for MNIST.
    No BatchNorm — gradients are clean and invertible.
    Intentionally small so DLG can reconstruct inputs reliably.
    This matches the network used in the original DLG paper.
    """
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)      # flatten 28x28 → 784
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ─────────────────────────────────────────────
# DLG core
# ─────────────────────────────────────────────

def compute_gradients(model, criterion, image, label):
    """Compute real gradients from one private image."""
    model.zero_grad()
    output = model(image)
    loss   = criterion(output, label)
    grads  = torch.autograd.grad(loss, model.parameters(), create_graph=False)
    return [g.detach() for g in grads]


def gradient_distance(dummy_grads, real_grads):
    """L2 distance between gradient sets — works well for small MLP."""
    return sum(((dg - rg) ** 2).sum() for dg, rg in zip(dummy_grads, real_grads))


def dlg_attack(model, real_grads, device, num_classes=10, iterations=300):
    """
    Original DLG attack with L-BFGS optimizer.
    Works reliably on small MLP + MNIST — exactly as in the paper.
    """
    criterion = nn.CrossEntropyLoss()

    # Random initialization — attacker knows nothing about the original image
    dummy_image = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)
    dummy_label = torch.randn(1, num_classes,  requires_grad=True, device=device)

    # L-BFGS: works perfectly for small networks (as used in original paper)
    optimizer = torch.optim.LBFGS([dummy_image, dummy_label], lr=1.0)

    history = []

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

        loss = optimizer.step(closure)
        history.append((i, loss.item()))

        if i % 50 == 0 or i == iterations - 1:
            print(f"  [attack] iter {i:>4}/{iterations} | grad loss: {loss.item():.8f}")

        # Early stop if converged
        if loss.item() < 1e-6:
            print(f"  [attack] converged at iter {i}!")
            break

    pred_label = torch.argmax(dummy_label, dim=-1).item()
    return dummy_image.detach(), pred_label, history


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def to_numpy(t):
    img = t.squeeze().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def save_comparison(originals, recons, true_labels, pred_labels, path):
    n   = len(originals)
    fig = plt.figure(figsize=(2.5 * n, 6))
    fig.suptitle(
        "DLG Attack: Original vs Reconstructed\n"
        "(MNIST digits reconstructed from intercepted gradients)",
        fontsize=13, y=1.02
    )
    gs = gridspec.GridSpec(2, n, hspace=0.5, wspace=0.3)

    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(to_numpy(originals[i]), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Original\ndigit: {true_labels[i]}", fontsize=9)
        ax.axis('off')

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(to_numpy(recons[i]), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Reconstructed\n(pred: {pred_labels[i]})", fontsize=9)
        ax.axis('off')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[attack] comparison saved → {path}")

def save_convergence(history, path):
    iters  = [h[0] for h in history]
    losses = [h[1] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, losses, color='#D85A30', linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gradient distance ||∇W' − ∇W||²")
    ax.set_title("DLG Attack Convergence (MNIST)")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[attack] convergence plot saved → {path}")

def save_progressive(dummy_snapshots, true_image, true_label, path):
    """
    Show how the reconstructed image evolves over iterations.
    Great figure for the report — shows the attack working in real time.
    """
    n   = len(dummy_snapshots)
    fig = plt.figure(figsize=(2.5 * (n + 1), 3.5))
    fig.suptitle(
        f"Reconstruction Progress (true digit: {true_label})",
        fontsize=12
    )
    gs = gridspec.GridSpec(1, n + 1, wspace=0.3)

    # Original on the left
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(to_numpy(true_image), cmap='gray', vmin=0, vmax=1)
    ax.set_title("Original", fontsize=9)
    ax.axis('off')

    # Snapshots
    for i, (iteration, img) in enumerate(dummy_snapshots):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(to_numpy(img), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"iter {iteration}", fontsize=9)
        ax.axis('off')

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[attack] progress figure saved → {path}")


# ─────────────────────────────────────────────
# Attack with snapshots for progressive figure
# ─────────────────────────────────────────────

def dlg_attack_with_snapshots(model, real_grads, device, num_classes=10,
                               iterations=300, snapshot_iters=None):
    """DLG attack that also saves intermediate images for the progress figure."""
    if snapshot_iters is None:
        snapshot_iters = [0, 10, 50, 100, 200, iterations - 1]

    criterion   = nn.CrossEntropyLoss()
    dummy_image = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)
    dummy_label = torch.randn(1, num_classes, requires_grad=True, device=device)
    optimizer   = torch.optim.LBFGS([dummy_image, dummy_label], lr=1.0)

    history   = []
    snapshots = []

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

        loss = optimizer.step(closure)
        history.append((i, loss.item()))

        if i in snapshot_iters:
            snapshots.append((i, dummy_image.detach().cpu().clone()))
            print(f"  [attack] iter {i:>4}/{iterations} | grad loss: {loss.item():.8f}")

        if loss.item() < 1e-6:
            snapshots.append((i, dummy_image.detach().cpu().clone()))
            print(f"  [attack] converged at iter {i}!")
            break

    pred_label = torch.argmax(dummy_label, dim=-1).item()
    return dummy_image.detach(), pred_label, history, snapshots


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_attack(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f" DLG Attack — MNIST (original paper setup)")
    print(f" images: {args.num_images} | iterations: {args.iterations} | device: {device}")
    print(f"{'='*50}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load MNIST — downloads automatically to ./data/mnist
    transform   = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_test  = datasets.MNIST('./data/mnist', train=False,
                                  download=True, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True)

    # Train a small MLP on MNIST for a few epochs
    mnist_train  = datasets.MNIST('./data/mnist', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

    model     = TinyMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("[setup] training TinyMLP on MNIST (5 epochs)...")
    for epoch in range(5):
        model.train()
        correct, total = 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == lbls).sum().item()
            total   += lbls.size(0)
        print(f"  epoch {epoch+1}/5 | train acc: {correct/total:.4f}")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in DataLoader(mnist_test, batch_size=256):
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += (model(imgs).argmax(1) == lbls).sum().item()
            total   += lbls.size(0)
    print(f"[setup] TinyMLP test accuracy: {correct/total:.4f}\n")

    # Run attack
    originals, recons, true_lbls, pred_lbls = [], [], [], []
    first_image_for_progress = None

    for idx, (image, label) in enumerate(test_loader):
        if idx >= args.num_images:
            break

        image = image.to(device)
        label = label.to(device)
        print(f"[attack] image {idx+1}/{args.num_images} | true digit: {label.item()}")

        real_grads = compute_gradients(model, criterion, image, label)

        if idx == 0:
            # First image: save progressive snapshots too
            dummy_img, pred_lbl, history, snapshots = dlg_attack_with_snapshots(
                model, real_grads, device,
                num_classes=10,
                iterations=args.iterations,
            )
            save_convergence(history, os.path.join(args.output_dir, "attack_convergence.png"))
            save_progressive(
                snapshots, image.cpu(), label.item(),
                os.path.join(args.output_dir, "attack_progress.png")
            )
        else:
            dummy_img, pred_lbl, history = dlg_attack(
                model, real_grads, device,
                num_classes=10,
                iterations=args.iterations,
            )

        originals.append(image.cpu())
        recons.append(dummy_img.cpu())
        true_lbls.append(label.item())
        pred_lbls.append(pred_lbl)

        orig_np  = to_numpy(image.cpu())
        recon_np = to_numpy(dummy_img.cpu())
        mse      = np.mean((orig_np - recon_np) ** 2)
        psnr     = 10 * np.log10(1.0 / (mse + 1e-10))
        print(f"  [result] MSE: {mse:.4f} | PSNR: {psnr:.2f} dB | pred: {pred_lbl}\n")

    save_comparison(originals, recons, true_lbls, pred_lbls,
                    os.path.join(args.output_dir, "attack_results.png"))

    print("[attack] done! Output files:")
    print(f"  {args.output_dir}/attack_results.png   — original vs reconstructed")
    print(f"  {args.output_dir}/attack_progress.png  — how reconstruction evolves")
    print(f"  {args.output_dir}/attack_convergence.png — loss curve")
    print("\n[attack] ready for Step 3 — differential privacy defense")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./attack_output")
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    run_attack(args)
