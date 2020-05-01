#!/usr/bin/env bash

# --constraint will enable constraint
# default embedding is chunk

# To run in default settings i.e. without constraints.
python training/train.py

# Enable constraint
# Default resource is cn_combined_unigram_content_only.json in respective train/test path
python training/train.py --constraint

# If constraint resource needs to be changed
python training/train.py --constraint --resource cn_combined_bigram.json

# To run on another dataset
python training/train.py --dataset_type image

# To change hidden dimension in pointer network
python training/train.py --hidden_dim 150 --constraint

# Change rho
python training/train.py --hidden_dim 150 --constraint --rho 2.0
