This guide will walk you through setting up the environment and running the Differential Privacy for MNIST Classification project.

## 1. Environment Setup

It is recommended to run this project in a Google Colab environment for ease of dependency management and access to GPUs.

## 2. Install Dependencies

Run the following commands in separate code cells to install the necessary Python packages:

```python
!pip install opacus
!pip install torch torchvision pandas matplotlib numpy
```

## 3. Mount Google Drive

This project saves output plots to Google Drive. To enable this, you need to mount your Google Drive. Run the following code cell and follow the prompts to authorize access:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4. Create Output Directory

An output directory will be created in your Google Drive to store generated plots. Run the following code cell:

```python
import os
os.makedirs('/content/drive/MyDrive/DP_Midterm', exist_ok=True)
```

## 5. Run the Notebook

After completing the above setup steps, you can proceed to run the rest of the code cells in the notebook sequentially. This will:
- Load and preprocess the MNIST dataset.
- Define the CNN model.
- Train the model with and without Differential Privacy (DP) using various noise multipliers.
- Generate plots visualizing the privacy-utility tradeoff.