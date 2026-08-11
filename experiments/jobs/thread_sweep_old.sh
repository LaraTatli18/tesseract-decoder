#!/bin/bash
#SBATCH --job-name=thread_sweep
#SBATCH --output=logs/thread_sweep_%A_%a.out
#SBATCH --error=logs/thread_sweep_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --array=0-6%1

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

THREADS=(1 2 4 8 16 32 64)
T="${THREADS[$SLURM_ARRAY_TASK_ID]}"

bazel run //src/py:run_tesseract -- \
  --decode-mode batch \
  --workers 1 \
  --threads "$T" \
  --n-shots 100000 \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --basis surface_code_X \
  --distances 11 \
  --p-values 0.001 \
  --max-files 1
