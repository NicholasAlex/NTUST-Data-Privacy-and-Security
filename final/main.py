"""
main.py — Federated Learning on CIFAR-10 with TinyCNN
Step 1 of the Final Project: Establish a Federated Learning System

Dataset: CIFAR-10 (auto-downloaded, no manual setup needed)
  - 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
  - 32x32 RGB images, 50000 training / 10000 test

Model: TinyCNN — small, no BatchNorm, DLG-compatible
  This is intentional: we use a DLG-vulnerable model so that Step 2
  (gradient leakage attack) can successfully reconstruct client images.
  This matches the experimental setup of the original DLG paper.

Expected accuracy: ~65-70% after 20 rounds

Usage:
  python main.py
  python main.py --num_clients 3 --num_rounds 20 --local_epochs 5
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from model import TinyCNN
from server import FLServer
from client import FLClient


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def get_cifar10(data_dir='./data/cifar10'):
    """Download and return CIFAR-10 train/test datasets."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_transform)
    test_set  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    print(f"[dataset] CIFAR-10 | train: {len(train_set)} | test: {len(test_set)}")
    return train_set, test_set


def partition_iid(dataset, num_clients):
    """Randomly split dataset indices across clients (IID)."""
    indices = np.random.permutation(len(dataset))
    splits  = np.array_split(indices, num_clients)
    for i, s in enumerate(splits):
        print(f"[data] client-{i}: {len(s)} samples")
    return [s.tolist() for s in splits]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def federated_train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f" Federated Learning — CIFAR-10 (TinyCNN)")
    print(f" clients: {args.num_clients} | rounds: {args.num_rounds} | "
          f"local epochs: {args.local_epochs}")
    print(f" device: {device}")
    print(f"{'='*50}\n")

    train_set, test_set = get_cifar10()
    test_loader         = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)
    client_indices      = partition_iid(train_set, args.num_clients)

    server  = FLServer(device=device, num_classes=10, model_class=TinyCNN)
    clients = [
        FLClient(
            client_id=i,
            dataset=train_set,
            indices=client_indices[i],
            device=device,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            model_class=TinyCNN,
        )
        for i in range(args.num_clients)
    ]

    best_acc   = 0.0
    best_round = 0

    print(f"\n[main] starting {args.num_rounds} FL rounds...\n")

    for fl_round in range(1, args.num_rounds + 1):
        print(f"\n─── Round {fl_round}/{args.num_rounds} ───")

        global_weights = server.get_global_weights()
        client_updates = []

        for client in clients:
            client.receive_weights(global_weights)
            update = client.local_train(
                current_round=fl_round,
                total_rounds=args.num_rounds,
            )
            client_updates.append(update)

        server.aggregate(client_updates)

        if fl_round % args.eval_every == 0 or fl_round == args.num_rounds:
            _, test_acc = server.evaluate(test_loader)
            if test_acc > best_acc:
                best_acc   = test_acc
                best_round = fl_round
                server.save_global_model('global_model_best.pth')
                print(f"[main] ★ new best: {best_acc:.4f} (round {best_round})")

    server.save_global_model('global_model.pth')
    server.print_history()
    print(f"\nBest accuracy: {best_acc:.4f} at round {best_round}")
    print("global_model_best.pth → use this for Step 2 (attack.py)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients',  type=int, default=3)
    parser.add_argument('--num_rounds',   type=int, default=20)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--eval_every',   type=int, default=2)
    args = parser.parse_args()
    federated_train(args)
