# HW2 — Differentially Private Synthetic Data
**Dataset:** Spaceship Titanic (Kaggle)  
**Method:** PrivBayes via DataSynthesizer  
**Course:** Data Privacy and Security

---

## Requirements

- Python 3.8+
- Google Colab (recommended) or a local Jupyter environment

---

## Setup

### 1. Download the Dataset

1. Go to [https://www.kaggle.com/competitions/spaceship-titanic/data](https://www.kaggle.com/competitions/spaceship-titanic/data)
2. Download `train.csv`
3. Upload it to your Google Drive at: `MyDrive/DPAS/HW2/train.csv`

### 2. Open the Notebook in Google Colab

1. Upload `hw2.ipynb` to your Google Drive
2. Open it with Google Colab
3. Mount your Google Drive when prompted (the first cell handles this automatically)

### 3. Install Dependencies

Run the following in a Colab cell before starting:

```bash
pip install DataSynthesizer scikit-learn pandas numpy
```

### 4. Fix DataSynthesizer Bug

The version of DataSynthesizer available on PyPI has a known bug in `DataGenerator.py` where `eval()` fails on single-parent Bayesian network nodes. The notebook includes a patch cell that fixes this automatically — make sure to run it before the data generation step.

---

## Running the Notebook

Run all cells in order:

| Section | Description |
|---|---|
| **Data Cleaning** | Loads `train.csv`, drops nulls and duplicates, parses `Cabin` into `Deck/CabinNum/Side`, parses `PassengerId` into `GroupId/MemberId`, drops `Name` |
| **Applying PrivBayes** | Generates DP synthetic datasets for ε = 0.1, 0.5, 1.0, 5.0, 10.0 |
| **Training the Model** | Trains SVC (RBF kernel) on each synthetic dataset, evaluates on real held-out test set |

---

## Output Files

After running, the following files will be saved to your working directory:

| File | Description |
|---|---|
| `spaceship_preprocessed.csv` | Cleaned real dataset (before OHE) |
| `description_eps{ε}.json` | PrivBayes Bayesian network description for each ε |
| `synthetic_eps{ε}.csv` | Synthetic dataset for each ε value |

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `epsilon` | 0.1, 0.5, 1.0, 5.0, 10.0 | Privacy budget (lower = more private) |
| `k` | 2 | Max parent nodes in Bayesian network |
| `category_threshold` | 5 | Columns with fewer unique values treated as categorical |
| `test_size` | 0.2 | Train/test split ratio |
| `random_state` | 42 | Reproducibility seed |
