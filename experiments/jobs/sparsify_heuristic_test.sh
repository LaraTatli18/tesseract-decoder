#!/bin/bash
#SBATCH --job-name=sparsify_check
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/sparsify_check_%j.out
#SBATCH --error=logs/sparsify_check_%j.err

set -euo pipefail

cd ~/tesseract-decoder

source .venv/bin/activate

echo "Starting run..."
date

bazel run //src/py:run_tesseract -- "$@"

echo "Finished."
date
