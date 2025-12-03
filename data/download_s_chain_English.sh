#!/bin/bash

# Directory where the dataset will be saved
TARGET_DIR="/data/S-Chain"

echo "Creating target directory at $TARGET_DIR ..."
mkdir -p "$TARGET_DIR"

echo "Downloading ONLY the English batch from leduckhai/S-Chain ..."
huggingface-cli download \
    leduckhai/S-Chain \
    --repo-type dataset \
    --local-dir "$TARGET_DIR" \
    --include "english/*"

echo "Download complete! English batch stored at $TARGET_DIR"
