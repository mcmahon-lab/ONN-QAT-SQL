# Optical Neural Networks with Quantization-aware Training

This repository contains the trained model and trianing scripts for the neural network executed on the optical matrix-vector multiplier demonstrated in the following paper: 

Author *et al.* (2021). An optical neural network using less than 1 photon per multiplication. *Journal Title, Volume* (Issue), page range. DOI

The device control scripts for experimental implementation are available [here](https://github.com/mcmahon-lab/ONN-device-control).
Besides the neural network training scripts, this repository also includes scripts for simulating neural network performance under the standard quantum limit (SQL). 

## [main_mnist_mlp_QAT.py](https://github.com/mcmahon-lab/ONN-QAT-SQL/blob/master/main_mnist_mlp_QAT.py)

The minimalist Python script for training fully-connected neural networks with quantization-aware training (QAT), requiring only PyTorch (1.7.0) and torchvision (0.8.1) to run

## [mnist_mlp_QAT_batch_training.ipynb](https://github.com/mcmahon-lab/ONN-QAT-SQL/blob/master/mnist_mlp_QAT_batch_training.ipynb)

A Jupyter notebook that trains batches of neural networks with QAT that supports additional functions (parallel training on GPUs, hyperparameter searching, neural architecture search, and training results logging).
The notebook requires additional packages: Ray (1.0.0), Optuna (1.5.0), wandb (0.9.7).

## [model_evaluation_shot_noise_sim.ipynb](https://github.com/mcmahon-lab/ONN-QAT-SQL/blob/master/model_evaluation_shot_noise_sim.ipynb)

A Jupyter notebook that tests the accuracy of trained neural networks with simulated photon shot noise under varying photon budgets (i.e., photons per multiplication).

## [A trained neural network model](https://github.com/mcmahon-lab/ONN-QAT-SQL/blob/master/RA_4bit_H2_100_100_lr_0.043_0.50_m_0.87_wep_6_randActDigi_v80_ep97.pt)

A neural network model with 3 hidden layers trained with QAT, and was the one finally executed on the experimental setup of 2D-block optical matrix-vector multiplier.

## [ana_lib](https://github.com/mcmahon-lab/ONN-QAT-SQL/tree/master/ana_lib)

Helper functions for parallelizing functions (e.g., neural network training functions) on GPUs.

# License

The code in this repository is released under the following license:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as license.txt.
