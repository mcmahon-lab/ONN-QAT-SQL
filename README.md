# Optical Neural Networks with Quantization-Aware Training (QAT)

This repository contains the trained model and training scripts for the neural network executed on the optical matrix-vector multiplier demonstrated in the following paper:

Tianyu Wang, Shi-Yuan Ma, Logan G. Wright, Tatsuhiro Onodera, Brian Richard and Peter L. McMahon (2021). An optical neural network using less than 1 photon per multiplication. *Manuscript in preparation*.

The device control scripts for experimental implementation are available [here](https://github.com/mcmahon-lab/ONN-device-control).

To improve the robustness of the **optical neural networks (ONNs)** against [shot noise](https://en.wikipedia.org/wiki/Shot_noise), we employed [**quantization-aware training (QAT)**](https://doi.org/10.1109/CVPR.2018.00286), which quantizes the activations and weights of neurons, and allows classification with moderate numerical precision. Besides the neural network training scripts, this repository also includes scripts for simulating neural network performance under the [**standard quantum limit (SQL)**](https://en.wikipedia.org/wiki/Quantum_limit).

## [ana_lib](https://github.com/mcmahon-lab/ONN-QAT-SQL/tree/master/ana_lib)

Helper functions for parallelizing functions (e.g., neural network training functions) on GPUs.

## [conda_env_spec.txt](https://github.com/mcmahon-lab/ONN-QAT-SQL/blob/master/conda_env_spec.txt)

Anacoda environment setup information.

## [main_mnist_mlp_QAT.py](https://github.com/mcmahon-lab/ONN-QAT-SQL/blob/master/main_mnist_mlp_QAT.py)

The minimalist Python script for training fully-connected neural networks with QAT, requiring only PyTorch (1.7.0) and torchvision (0.8.1) to run

## [mnist_mlp_QAT_batch_training.ipynb](https://github.com/mcmahon-lab/ONN-QAT-SQL/blob/master/mnist_mlp_QAT_batch_training.ipynb)

A Jupyter notebook that trains batches of neural networks with QAT that supports additional functions (parallel training on GPUs, hyperparameter searching, neural architecture search, and training results logging).
The notebook requires additional packages: Ray (1.0.0), Optuna (1.5.0), wandb (0.9.7).

## [model_evaluation_shot_noise_sim.ipynb](https://github.com/mcmahon-lab/ONN-QAT-SQL/blob/master/model_evaluation_shot_noise_sim.ipynb)

A Jupyter notebook that tests the accuracy of trained neural networks with simulated photon shot noise under varying photon budget (i.e., photons per scalar multiplication).

## [trained_model_4bit_H2_100_100.pt](https://github.com/mcmahon-lab/ONN-QAT-SQL/blob/master/trained_model_4bit_H2_100_100.pt)

A neural network model with 3 hidden layers trained with QAT. It was the one finally executed on the experimental setup of the optical matrix-vector multiplier.

# License

The code in this repository is released under the following license:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as license.txt.
