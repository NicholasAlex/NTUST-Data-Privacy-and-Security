"""
defend.py — Differential Privacy Defense (DP-SGD)
Step 3 of the Final Project

Usage:
  python defend.py                  # full experiment (trains 4 models)
  python defend.py --attack_only    # skip retraining, use saved models
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import TinyCNN


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CIFAR10_CLASSES = ['airplane','car','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3,1,1)

# (display_label, noise_multiplier, approx_epsilon)
# Noise multipliers tuned so training still converges:
#   sigma=0.0  → no DP (baseline)
#   sigma=0.01 → very weak noise, small accuracy drop
#   sigma=0.05 → moderate noise, noticeable accuracy drop
#   sigma=0.1  → strong noise, significant accuracy drop
DP_SETTINGS = [
    ("No DP\n(ε=∞)",       0.0,   float('inf')),
    ("Weak DP\n(ε≈10)",    0.01,  10.0),
    ("Medium DP\n(ε≈1)",   0.02,  1.0),
    ("Strong DP\n(ε≈0.1)", 0.05,  0.1),
]


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def get_cifar10(data_dir='./data/cifar10'):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    train = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    test  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)
    return train, test


# ─────────────────────────────────────────────
# DP-SGD client
# ─────────────────────────────────────────────

class DPClient:
    """
    FL client with DP-SGD.
    After loss.backward():
      1. Clip gradients to max norm C
      2. Add Gaussian noise N(0, (sigma*C)^2)
    """
    def __init__(self, client_id, dataset, indices, device,
                 local_epochs=5, batch_size=32,
                 max_grad_norm=1.0, noise_multiplier=0.0):
        self.client_id        = client_id
        self.device           = device
        self.local_epochs     = local_epochs
        self.max_grad_norm    = max_grad_norm
        self.noise_multiplier = noise_multiplier

        self.loader    = DataLoader(Subset(dataset, indices),
                                    batch_size=batch_size, shuffle=True, num_workers=0)
        self.model     = TinyCNN(num_classes=10).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def receive_weights(self, global_weights):
        self.model.set_weights([w.to(self.device) for w in global_weights])

    def _dp_step(self):
        """Clip then add noise to gradients."""
        # Clip
        total_norm = sum(p.grad.data.norm(2).item()**2
                         for p in self.model.parameters() if p.grad is not None) ** 0.5
        clip_coef  = min(1.0, self.max_grad_norm / (total_norm + 1e-8))
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
        # Add noise
        if self.noise_multiplier > 0:
            noise_std = self.noise_multiplier * self.max_grad_norm
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.add_(torch.randn_like(p.grad.data) * noise_std)

    def local_train(self, current_round, total_rounds):
        base_lr  = 0.01
        progress = (current_round - 1) / max(total_rounds - 1, 1)
        lr       = max(base_lr * 0.5 * (1 + math.cos(math.pi * progress)), 1e-4)
        optimizer = optim.SGD(self.model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=1e-4, nesterov=True)
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for _ in range(self.local_epochs):
            for images, labels in self.loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)
                loss.backward()
                self._dp_step()
                optimizer.step()
                total_loss    += loss.item() * images.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += images.size(0)
        acc = total_correct / total_samples
        print(f"  [client-{self.client_id}] round {current_round} | acc: {acc:.4f}")
        return {"weights": self.model.get_weights(), "num_samples": total_samples,
                "local_loss": total_loss/total_samples, "local_acc": acc}


# ─────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────

class SimpleServer:
    def __init__(self, device):
        self.device       = device
        self.global_model = TinyCNN(num_classes=10).to(device)

    def get_weights(self):
        return self.global_model.get_weights()

    def aggregate(self, updates):
        total = sum(u["num_samples"] for u in updates)
        agg   = [torch.zeros_like(w) for w in updates[0]["weights"]]
        for u in updates:
            for a, lw in zip(agg, u["weights"]):
                a += (u["num_samples"]/total) * lw.to(self.device)
        self.global_model.set_weights(agg)

    def evaluate(self, loader):
        self.global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                correct += (self.global_model(imgs).argmax(1) == lbls).sum().item()
                total   += lbls.size(0)
        return correct / total

    def save(self, path):
        torch.save(self.global_model.state_dict(), path)
        print(f"  [server] model saved → {path}")


# ─────────────────────────────────────────────
# FL training with DP
# ─────────────────────────────────────────────

def train_with_dp(train_set, test_loader, device, noise_multiplier,
                  max_grad_norm, num_clients, num_rounds, local_epochs, save_path):
    indices = np.random.permutation(len(train_set))
    splits  = [s.tolist() for s in np.array_split(indices, num_clients)]
    server  = SimpleServer(device)
    clients = [DPClient(i, train_set, splits[i], device,
                        local_epochs=local_epochs,
                        max_grad_norm=max_grad_norm,
                        noise_multiplier=noise_multiplier)
               for i in range(num_clients)]
    best_acc = 0.0
    for fl_round in range(1, num_rounds+1):
        global_weights = server.get_weights()
        updates = []
        for client in clients:
            client.receive_weights(global_weights)
            updates.append(client.local_train(fl_round, num_rounds))
        server.aggregate(updates)
        if fl_round % 5 == 0 or fl_round == num_rounds:
            acc = server.evaluate(test_loader)
            print(f"  [round {fl_round}/{num_rounds}] test acc: {acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                server.save(save_path)
    return best_acc


# ─────────────────────────────────────────────
# Evaluate saved model
# ─────────────────────────────────────────────

def evaluate_model(model_path, test_loader, device):
    model = TinyCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            correct += (model(imgs).argmax(1) == lbls).sum().item()
            total   += lbls.size(0)
    return correct / total, model


# ─────────────────────────────────────────────
# DLG attack
# ─────────────────────────────────────────────

def dlg_attack_quick(model, image, label, device, iterations=150, seed=42):
    torch.manual_seed(seed)
    criterion   = nn.CrossEntropyLoss()
    model.zero_grad()
    real_grads  = [g.detach() for g in
                   torch.autograd.grad(criterion(model(image), label),
                                       model.parameters(), create_graph=False)]
    dummy_image = torch.randn_like(image, requires_grad=True)
    dummy_label = torch.randn(1, 10, requires_grad=True, device=device)
    optimizer   = torch.optim.LBFGS([dummy_image, dummy_label], lr=1.0)
    best_loss, best_img = float('inf'), dummy_image.detach().clone()

    for _ in range(iterations):
        def closure():
            optimizer.zero_grad()
            model.zero_grad()
            pred        = model(dummy_image)
            soft_label  = torch.softmax(dummy_label, dim=-1)
            loss        = criterion(pred, soft_label)
            dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            gd = sum(((dg-rg)**2).sum() for dg,rg in zip(dummy_grads, real_grads))
            gd.backward()
            return gd
        loss = optimizer.step(closure)
        lv = loss.item()
        if np.isnan(lv) or np.isinf(lv):
            break
        if lv < best_loss:
            best_loss = lv
            best_img  = dummy_image.detach().clone()

    orig_np  = np.clip((image.cpu().squeeze()*STD+MEAN).permute(1,2,0).numpy(), 0, 1)
    recon_np = np.clip((best_img.cpu().squeeze()*STD+MEAN).permute(1,2,0).numpy(), 0, 1)
    psnr = 10 * np.log10(1.0 / (np.mean((orig_np-recon_np)**2) + 1e-10))
    return best_img.cpu(), best_loss, psnr


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def to_numpy(t):
    img = t.squeeze().cpu() * STD + MEAN
    return np.clip(img.permute(1,2,0).numpy(), 0, 1)

def save_tradeoff_plot(setting_labels, accuracies, psnrs, path):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = range(len(setting_labels))
    bars = ax1.bar(x, [a*100 for a in accuracies], width=0.4,
                   color='#185FA5', alpha=0.7, label='Test accuracy (%)')
    ax1.set_ylabel('Test accuracy (%)', color='#185FA5', fontsize=12)
    ax1.set_xlabel('Privacy setting (ε)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace('\n',' ') for s in setting_labels])
    ax1.tick_params(axis='y', labelcolor='#185FA5')
    ax1.set_ylim(0, 100)
    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9, color='#185FA5')

    ax2 = ax1.twinx()
    ax2.plot(x, psnrs, color='#D85A30', marker='o',
             linewidth=2, markersize=8, label='Attack PSNR (dB)')
    for i, psnr in enumerate(psnrs):
        ax2.annotate(f'{psnr:.1f}dB', (i, psnr), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, color='#D85A30')
    ax2.set_ylabel('Attack PSNR (dB) — lower = better defense', color='#D85A30', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#D85A30')
    ax2.set_ylim(0, max(psnrs)*1.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper right')
    plt.title('Privacy-Accuracy Tradeoff (Differential Privacy Defense)\n'
              'Smaller ε = stronger privacy = lower accuracy', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[defend] tradeoff plot saved → {path}")

def save_reconstruction_grid(image, recons, labels, true_label, path):
    n   = len(recons)
    fig = plt.figure(figsize=(3*(n+1), 4))
    fig.suptitle(f"DLG Attack Quality Under DP Defense — true: {CIFAR10_CLASSES[true_label]}",
                 fontsize=12)
    gs = gridspec.GridSpec(1, n+1, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(to_numpy(image))
    ax.set_title("Original", fontsize=9)
    ax.axis('off')
    for i, (recon, label) in enumerate(zip(recons, labels)):
        ax = fig.add_subplot(gs[0, i+1])
        ax.imshow(to_numpy(recon))
        ax.set_title(label, fontsize=8)
        ax.axis('off')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[defend] reconstruction grid saved → {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_full_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*55}")
    print(f" Step 3: Differential Privacy Defense")
    print(f" attack_only: {args.attack_only} | device: {device}")
    print(f"{'='*55}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    train_set, test_set = get_cifar10()
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    # Fixed test image for attack comparison
    g = torch.Generator()
    g.manual_seed(42)
    attack_loader = DataLoader(test_set, batch_size=1, shuffle=True, generator=g)
    attack_image, attack_label = next(iter(attack_loader))
    attack_image = attack_image.to(device)
    attack_label = attack_label.to(device)
    print(f"[defend] attack image: {CIFAR10_CLASSES[attack_label.item()]}\n")

    results = []   # (label, acc, psnr, recon_img)

    for label, noise_mult, approx_eps in DP_SETTINGS:
        clean = label.replace('\n', ' ')
        model_path = os.path.join(args.output_dir, f"model_sigma{noise_mult}.pth")

        print(f"\n{'─'*40}")
        print(f" {clean} | σ={noise_mult}")
        print(f"{'─'*40}")

        if args.attack_only:
            # ── Skip training, load existing model ──
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    f"Run without --attack_only first to train the models."
                )
            print(f"  [skip] loading saved model from {model_path}")
            acc, model = evaluate_model(model_path, test_loader, device)
            print(f"  [eval] test acc: {acc:.4f}")
        else:
            # ── Full FL training ──
            print(f"  [train] FL training with DP-SGD (σ={noise_mult})...")
            acc = train_with_dp(
                train_set        = train_set,
                test_loader      = test_loader,
                device           = device,
                noise_multiplier = noise_mult,
                max_grad_norm    = args.max_grad_norm,
                num_clients      = args.num_clients,
                num_rounds       = args.num_rounds,
                local_epochs     = args.local_epochs,
                save_path        = model_path,
            )
            print(f"  [train] best acc: {acc:.4f}")
            # Re-evaluate to confirm accuracy
            acc, model = evaluate_model(model_path, test_loader, device)
            print(f"  [eval] confirmed test acc: {acc:.4f}")

        # ── DLG attack ──
        print(f"  [attack] running DLG attack...")
        model.eval()
        recon_img, grad_loss, psnr = dlg_attack_quick(
            model, attack_image, attack_label, device, iterations=150, seed=42
        )
        print(f"  [attack] grad loss: {grad_loss:.4f} | PSNR: {psnr:.2f} dB")

        results.append((label, acc, psnr, recon_img))

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"{'Setting':<22} {'Accuracy':>10} {'PSNR (dB)':>12}")
    print("-" * 45)
    for label, acc, psnr, _ in results:
        print(f"{label.replace(chr(10),' '):<22} {acc*100:>9.1f}% {psnr:>12.2f}")

    # ── Figures ──
    setting_labels = [r[0] for r in results]
    accuracies     = [r[1] for r in results]
    psnrs          = [r[2] for r in results]
    recons         = [r[3] for r in results]

    save_tradeoff_plot(
        setting_labels, accuracies, psnrs,
        os.path.join(args.output_dir, 'privacy_tradeoff.png')
    )
    save_reconstruction_grid(
        attack_image.cpu(), recons,
        [f"{r[0].replace(chr(10),' ')}\nacc:{r[1]*100:.1f}% PSNR:{r[2]:.1f}dB"
         for r in results],
        attack_label.item(),
        os.path.join(args.output_dir, 'dp_reconstruction_grid.png')
    )
    print("\n[defend] Step 3 complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',    type=str,   default='./defend_output')
    parser.add_argument('--num_clients',   type=int,   default=3)
    parser.add_argument('--num_rounds',    type=int,   default=20)
    parser.add_argument('--local_epochs',  type=int,   default=5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--attack_only',   action='store_true',
                        help='Skip retraining, load saved models and rerun attack + plots')
    args = parser.parse_args()
    run_full_experiment(args)