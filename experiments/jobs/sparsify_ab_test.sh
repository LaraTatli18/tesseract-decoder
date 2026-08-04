#!/bin/bash
#SBATCH --job-name=sparsify_ab
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sparsify_ab_%j.out
#SBATCH --error=logs/sparsify_ab_%j.err

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

echo "========================================"
echo "Sparsification A/B test"
echo "========================================"

echo "Running OFF case..."
bazel run //src/py:run_tesseract -- \
  --run-group sparsify_ab_test/off \
  --decode-mode batch \
  --max-files 1 \
  --n-shots 100000 \
  --workers 1 \
  --threads 32 \
  --basis surface_code_X \
  --distances 11 \
  --p-values 0.002 \
  --det-beam 20 \
  --pqlimit 200000 \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --beam-climbing \
  --merge-errors \
  --no-sparsify_errors

echo "Running ON case..."
bazel run //src/py:run_tesseract -- \
  --run-group sparsify_ab_test/on \
  --decode-mode batch \
  --max-files 1 \
  --n-shots 100000 \
  --workers 1 \
  --threads 32 \
  --basis surface_code_X \
  --distances 11 \
  --p-values 0.002 \
  --det-beam 20 \
  --pqlimit 200000 \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --beam-climbing \
  --merge-errors \
  --sparsify_errors \
  --sparsify-base-degree 1 \
  --sparsify-max-degree 4 \
  --sparsify-reactivate-limit 32

echo "Done."
