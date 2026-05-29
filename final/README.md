# Final Project : Federated Learning & Its Attacks and Defenses

## Requirements

- Python 3.9+
- CUDA-capable GPU (recommended)

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

CIFAR-10 downloads automatically when you run `main.py`. No manual setup needed.

## How to Reproduce

### Step 1 — Train the FL System

```bash
python main.py
```

Trains a federated learning system with 3 clients over 20 rounds on CIFAR-10.
Saves the best model to `global_model_best.pth`.

Optional arguments:
```bash
python main.py --num_clients 3 --num_rounds 20 --local_epochs 5
```

### Step 2 — Gradient Leakage Attack

Run the standard DLG attack on the trained FL model:
```bash
python attack.py --mode standard
```

Run seed stability analysis (same image, 5 different random seeds):
```bash
python attack.py --mode seeds
```

Run learning rate sweep (same image, 5 different LR values):
```bash
python attack.py --mode sweep
```

Output saved to `./attack_output/`.

### Step 3 — Differential Privacy Defense

Train 4 FL models under different DP settings and compare attack quality:
```bash
python defend.py
```

If models are already trained, skip retraining and just regenerate plots:
```bash
python defend.py --attack_only
```

Output saved to `./defend_output/`.

## File Structure

```
final/
├── model.py          # TinyCNN model definition
├── client.py         # FL client (local training)
├── server.py         # FL server (FedAvg aggregation)
├── main.py           # Step 1: FL training loop
├── attack.py         # Step 2: DLG attack + analysis
├── defend.py         # Step 3: DP defense + tradeoff experiment
└── requirements.txt
```

## Expected Output Files

| File | Description |
|------|-------------|
| `global_model_best.pth` | Trained FL model from Step 1 |
| `attack_output/attack_results.png` | Original vs reconstructed images |
| `attack_output/attack_convergence.png` | DLG loss convergence curve |
| `attack_output/seed_analysis.png` | Stability across random seeds |
| `attack_output/lr_sweep.png` | Effect of learning rate on attack |
| `defend_output/privacy_tradeoff.png` | Accuracy vs PSNR across ε values |
| `defend_output/dp_reconstruction_grid.png` | Attack quality under each DP setting |
