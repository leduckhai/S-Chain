#!/bin/bash

# Directory where the dataset will be saved
TARGET_DIR="/data/S-Chain"

echo "Installing git-lfs (if not installed)..."
git lfs install

echo "Cloning dataset into $TARGET_DIR ..."
git clone https://huggingface.co/datasets/leduckhai/S-Chain "$TARGET_DIR"

echo "Download complete! Dataset stored at $TARGET_DIR"
