# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class FaceDataset(Dataset):
    """
    Loads face images from a processed_faces subfolder.
    Each image filename encodes the label: s{label}_img{i}.png
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
 
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # (1, H, W)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label    

def load_variant_dataset(variant_dir, train_ratio=0.8):
    """
    Load images from a single variant folder (e.g. processed_faces/pix_b4/).
    Splits into train/test sets (per subject, stratified).
 
    Returns:
        train_dataset, test_dataset: FaceDataset objects
    """
    image_paths = sorted([
        os.path.join(variant_dir, f)
        for f in os.listdir(variant_dir)
        if f.endswith('.png')
    ])
 
    # Parse labels from filenames: s{label:02d}_img{i:03d}.png
    labels = [int(os.path.basename(p).split('_')[0][1:]) for p in image_paths]
 
    # Stratified split: group by subject, take first 8 for train, last 2 for test
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
 
    unique_labels = sorted(set(labels))
    for subj in unique_labels:
        subj_paths = [p for p, l in zip(image_paths, labels) if l == subj]
        subj_labels = [l for l in labels if l == subj]
        n_train = int(len(subj_paths) * train_ratio)
        train_paths += subj_paths[:n_train]
        train_labels += subj_labels[:n_train]
        test_paths += subj_paths[n_train:]
        test_labels += subj_labels[n_train:]
 
    train_dataset = FaceDataset(train_paths, train_labels)
    test_dataset = FaceDataset(test_paths, test_labels)
    return train_dataset, test_dataset

# ─────────────────────────────────────────────
# 2. CNN MODEL
# ─────────────────────────────────────────────
 
class FaceCNN(nn.Module):
    """
    Simple CNN for face re-identification.
    Input: (batch, 1, 64, 64) grayscale image
    Output: (batch, num_classes) logits
 
    Kept intentionally simple to avoid overfitting on 400 images.
    Architecture: 3 conv blocks + 2 FC layers + dropout
    """
    def __init__(self, num_classes=40):
        super(FaceCNN, self).__init__()
 
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (32, 32, 32)
 
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (64, 16, 16)
 
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# -> (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (128, 8, 8)
        )
 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
 
 
# ─────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────
 
def train_model(model, train_loader, device, num_epochs=50, lr=1e-3):
    """
    Train the CNN model.
    Returns list of per-epoch training losses.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
 
    loss_history = []
 
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
 
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
 
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
 
    return loss_history
 
 
def evaluate_model(model, test_loader, device, top_k=5):
    """
    Evaluate model accuracy on test set.
    Returns:
        top1_acc: float — Top-1 accuracy (%)
        top5_acc: float — Top-5 accuracy (%)
    """
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
 
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
 
            # Top-1
            _, predicted = outputs.max(1)
            top1_correct += predicted.eq(labels).sum().item()
 
            # Top-5
            _, top5_pred = outputs.topk(min(top_k, outputs.size(1)), dim=1)
            top5_correct += sum(
                labels[i].item() in top5_pred[i].tolist()
                for i in range(labels.size(0))
            )
 
            total += labels.size(0)
 
    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
    return top1_acc, top5_acc
 
 
# ─────────────────────────────────────────────
# RUN ATTACK ON ALL VARIANTS
# ─────────────────────────────────────────────
 
def run_attack(processed_dir="processed_faces", num_epochs=50, batch_size=32, device=None):
    """
    Train and evaluate a CNN on every de-identification variant.
    Returns:
        results: dict mapping variant name -> {'top1': float, 'top5': float}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
 
    variants = [
        "original",
        "pix_b4", "pix_b8", "pix_b16",
        "blur_k15", "blur_k45", "blur_k99",
    ]
 
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
        loss_history = train_model(model, train_loader, device, num_epochs=num_epochs)
 
        top1, top5 = evaluate_model(model, test_loader, device)
        results[variant] = {"top1": top1, "top5": top5, "loss": loss_history}
        print(f"  -> Top-1: {top1:.2f}%  |  Top-5: {top5:.2f}%\n")
 
    return results
 
 
# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
 
def plot_attack_results(results, save_path="step2_attack_results.png"):
    """
    Plot a bar chart of Top-1 and Top-5 accuracy across all variants.
    Mirrors the result table format shown in the homework slides.
    """
    variants = list(results.keys())
    top1_scores = [results[v]["top1"] for v in variants]
    top5_scores = [results[v]["top5"] for v in variants]
 
    x = np.arange(len(variants))
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, top1_scores, width, label="Top-1 Accuracy", color="steelblue")
    bars2 = ax.bar(x + width/2, top5_scores, width, label="Top-5 Accuracy", color="coral")
 
    # Annotate bars with values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8)
 
    ax.set_title("Step 2: CNN Re-Identification Attack Accuracy", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15)
    ax.set_ylim(0, 110)
    ax.axhline(y=2.5, color='gray', linestyle='--', linewidth=1, label="Random baseline (2.5%)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")
 
 
def print_results_table(results):
    """
    Print a formatted accuracy table like the one in the homework slides.
    """
    baseline = 2.5  # 1/40 subjects = random guess
 
    print("\n" + "="*65)
    print(f"{'Variant':<15} {'Top-1 Acc (%)':>15} {'Top-5 Acc (%)':>15}")
    print("="*65)
    print(f"{'Random baseline':<15} {baseline:>15.2f} {baseline*5:>15.2f}")
    print("-"*65)
    for variant, scores in results.items():
        print(f"{variant:<15} {scores['top1']:>15.2f} {scores['top5']:>15.2f}")
    print("="*65)
 
 
def plot_loss_curves(results, save_path="step2_loss_curves.png"):
    """
    Plot training loss curves for all variants.
    Useful to verify convergence and check for overfitting.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
 
    for variant, scores in results.items():
        ax.plot(scores["loss"], label=variant)
 
    ax.set_title("Step 2: Training Loss Curves", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")