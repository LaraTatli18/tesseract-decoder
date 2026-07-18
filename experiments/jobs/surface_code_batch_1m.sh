#!/bin/bash
#SBATCH --job-name=surface_batch_1m
#SBATCH --output=logs/surface_batch_1m_%j.out
#SBATCH --error=logs/surface_batch_1m_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --partition=main

set -euo pipefail

cd ~/tesseract-decoder

source .venv/bin/activate

bazel run //src/py:run_tesseract -- \
  --decode-mode batch \
  --workers 12 \
  --threads 16 \
  --n-shots 1000000 \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --basis surface_code_X \
  --distances 3 5 7 9 11 \
  --p-values 0.0005 0.001 0.002 \
  --output-csv "$(pwd)/experiments/results/surface_code_X_batch_1000000shots_12workers_16threads.csv"
