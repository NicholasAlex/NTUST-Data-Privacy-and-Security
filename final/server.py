"""
server.py — Federated Learning server (FedAvg aggregation)
"""

import torch
import torch.nn as nn
from model import TinyCNN


class FLServer:
    def __init__(self, device, num_classes=10, model_class=TinyCNN):
        self.device       = device
        self.global_model = model_class(num_classes=num_classes).to(device)
        self.round        = 0
        self.history      = []

        total_params = sum(p.numel() for p in self.global_model.parameters())
        print(f"[server] initialized {model_class.__name__} | parameters: {total_params:,}")

    def get_global_weights(self):
        return self.global_model.get_weights()

    def aggregate(self, client_updates):
        """FedAvg: weighted average by number of samples."""
        self.round += 1
        total_samples = sum(u["num_samples"] for u in client_updates)
        agg_weights   = [torch.zeros_like(w) for w in client_updates[0]["weights"]]

        for update in client_updates:
            weight = update["num_samples"] / total_samples
            for agg_w, local_w in zip(agg_weights, update["weights"]):
                agg_w += weight * local_w.to(self.device)

        self.global_model.set_weights(agg_weights)

        avg_loss = sum(u["local_loss"] * u["num_samples"] for u in client_updates) / total_samples
        avg_acc  = sum(u["local_acc"]  * u["num_samples"] for u in client_updates) / total_samples
        self.history.append({"round": self.round, "avg_loss": avg_loss, "avg_acc": avg_acc})
        print(f"[server] round {self.round} | avg loss: {avg_loss:.4f} | avg acc: {avg_acc:.4f}")

    def evaluate(self, test_loader):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                total_loss    += criterion(outputs, labels).item() * images.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += images.size(0)
        test_loss = total_loss / total_samples
        test_acc  = total_correct / total_samples
        print(f"[server] test acc: {test_acc:.4f} | loss: {test_loss:.4f}")
        return test_loss, test_acc

    def save_global_model(self, path="global_model.pth"):
        torch.save(self.global_model.state_dict(), path)
        print(f"[server] model saved → {path}")

    def print_history(self):
        print(f"\n{'='*50}")
        print(f"{'Round':<8} {'Avg Loss':<12} {'Avg Acc':<10}")
        print("-" * 30)
        for h in self.history:
            print(f"{h['round']:<8} {h['avg_loss']:<12.4f} {h['avg_acc']:<10.4f}")