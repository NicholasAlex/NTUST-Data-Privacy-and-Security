"""
client.py — FL client, works with any model class
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


class FLClient:
    def __init__(self, client_id, dataset, indices, device,
                 local_epochs=5, batch_size=32, model_class=None):
        self.client_id    = client_id
        self.device       = device
        self.local_epochs = local_epochs
        self.model_class  = model_class

        local_dataset = Subset(dataset, indices)
        self.loader   = DataLoader(local_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
        self.model     = model_class().to(device)
        self.criterion = nn.CrossEntropyLoss()
        print(f"[client-{client_id}] {len(indices)} samples | {model_class.__name__}")

    def receive_weights(self, global_weights):
        self.model.set_weights([w.to(self.device) for w in global_weights])

    def local_train(self, current_round, total_rounds):
        # Cosine LR schedule across rounds
        base_lr  = 0.01
        progress = (current_round - 1) / max(total_rounds - 1, 1)
        lr       = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        lr       = max(lr, 1e-4)

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_loss    += loss.item() * images.size(0)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"[client-{self.client_id}] round {current_round} | "
              f"lr: {lr:.5f} | loss: {avg_loss:.4f} | acc: {accuracy:.4f}")
        return {"weights": self.model.get_weights(), "num_samples": total_samples,
                "local_loss": avg_loss, "local_acc": accuracy}
