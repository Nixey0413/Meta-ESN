# Meta-ESN
Meta-ESN Code Repository

Welcome to the code repository for the manuscript "Meta-ESN: A Novel Echo State Network Approach with Meta-Learning for Time-Series Prediction." This repository contains all necessary scripts to reproduce the results presented in the paper, including feature extraction, model pre-training, testing, and baseline comparisons. The code is implemented in Python and relies on PyTorch.

Repository Structure

features_reptile.py: Performs reservoir mapping to generate echo state features.

main_pretrain_reptile.py: Pretrains the Meta-ESN model.

main_test_reptile.py: Tests the pretrained model on your dataset.

main_baseline.py: Implements baseline methods for performance comparison.

Prerequisites

Python [version, 3.8+]
Dependencies: [NumPy, PyTorch]

Data Preparation

Prepare your time-series dataset in [.mat format].
Place the dataset in the data/ folder or update the file path in the scripts.

Notes

Adjust hyperparameters (e.g., reservoir size, learning rate) in the respective scripts if needed.
Refer to the manuscript for detailed methodology and experimental settings.
For issues or questions, please contact [nixey96@163.com].
