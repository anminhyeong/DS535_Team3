# DS535 Team 3

This repo is implemented based on the official code for ICML 2024 paper: **Recurrent Distance Filtering for Graph Representation Learning** (https://arxiv.org/abs/2312.01538)

## Setup

The code is developed using JAX and Flax (since we want to use `associative_scan`):

```
python==3.10.12
jaxlib==0.4.23
flax==0.8.5
optax==0.1.9   # optimizer lib
numpy==1.26.4
scipy==1.11.2
scikit-learn==1.6.0
```
Please refer to the [JAX](https://jax.readthedocs.io/en/latest/installation.html) and [Flax](https://github.com/google/flax?tab=readme-ov-file#quick-install) installation pages.

To keep the data preprocessing consistent with other baselines, we load the datasets using `torch-geometric==2.3.1` and convert them into NumPy arrays.
You need `torch-geometric` to run `preprocess.py` and `preprocess_peptides.py`, but you don't need it to run training scripts.

> Instead of running in the local environment, you can run on Colab environment using the provided experiments.ipynb file.

## Experiment Setting

We conducted experiments differentiating the following settings:
1. Random walk strategies (random walk, DFS-like walk, highly DFS-like walk)
  - Configured as argument (**p_value** and **q_value**) in the preprocess.py command.
  - Random walk: p_value = 1.0, q_value = 1.0
  - DFS-like walk: p_value = 1e10, q_value = 0.5
  - Highly DFS-like walk: p_value = 1e10, q_value = 0.1
2. Mask data type (int, bool)
  - Configured as argument (**bool_mask**) in the preprocess.py command.
3. Number of hops (num_hops = 3, 4, 5)
  - Configured as argument (**num_hops**) in the train_{pixel, sbm, zinc}.py command.

## Data Preprocessing

To prepare MNIST data with random walk strategy, please run:
```
python preprocess.py --length 5 --num 5 --p_value 1.0 --q_value 1.0 --name MNIST
```
You can change p_value, q_value to apply different random walk strategies.  
Same code applies to CIFAR10, ZINC, CLUSTER, PATTERN except for **name** argument in the command.  

You can download the dataset from the following link.  
https://drive.google.com/drive/folders/12kKq-WAer7TYh3R5H3wyxzZaWTqzyOyD?usp=drive_link


## Training

For MNIST and CIFAR10:
```
python train_pixel.py --name MNIST --num_layers 4 --num_hops 3 --dim_h 128 --dim_v 96
python train_pixel.py --name CIFAR10 --num_layers 8 --num_hops 5 --dim_h 96 --dim_v 64
```

For ZINC:
```
python train_zinc.py
```
For CLUSTER and PATTERN:
```
python train_sbm.py --name CLUSTER --num_layers 16 --dim_h 64 --dim_v 64 --weight_decay 0.2 --r_min 0.9
python train_sbm.py --name PATTERN --num_layers 10 --dim_h 72 --dim_v 64 --weight_decay 0.1 --r_min 0.5
```

To apply a **boolean mask** when running the train_{pixel, sbm, zinc}.py command, you need to include the --bool_mask argument in your command.

For MNIST and CIFAR10:
```
python train_pixel.py --name MNIST --num_layers 4 --num_hops 3 --dim_h 128 --dim_v 96 --bool_mask
python train_pixel.py --name CIFAR10 --num_layers 8 --num_hops 5 --dim_h 96 --dim_v 64 --bool_mask
```

For ZINC:
```
python train_zinc.py --bool_mask
```
For CLUSTER and PATTERN:
```
python train_sbm.py --name CLUSTER --num_layers 16 --dim_h 64 --dim_v 64 --weight_decay 0.2 --r_min 0.9 --bool_mask
python train_sbm.py --name PATTERN --num_layers 10 --dim_h 72 --dim_v 64 --weight_decay 0.1 --r_min 0.5 --bool_mask