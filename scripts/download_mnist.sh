#!/usr/bin/env bash
# Download and decompress MNIST into data/ for cargo run -- --mnist
set -e
# Prefer Google mirror (original yann.lecun.com often 404)
BASE="${MNIST_URL:-https://storage.googleapis.com/cvdf-datasets/mnist}"
DIR="${1:-data}"
mkdir -p "$DIR"
cd "$DIR"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ ! -f "$f" ]; then
    echo "Fetching $f.gz..."
    curl -sSfL -o "$f.gz" "$BASE/$f.gz"
    echo "Decompressing $f.gz..."
    gunzip -k -f "$f.gz"
  else
    echo "$f already present, skipping."
  fi
done
echo "Done. MNIST is in $DIR/ (run from repo root: cargo run -- --mnist)"
