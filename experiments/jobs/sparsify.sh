#!/bin/bash
#SBATCH --job-name=sparsify
#SBATCH --output=logs/sparsify_%A_%a.out
#SBATCH --error=logs/sparsify_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --array=0-7%1

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

DISTANCES=(5 5 7 7 9 9 11 11)
SPARSIFY=(off on off on off on off on)

DIST="${DISTANCES[$SLURM_ARRAY_TASK_ID]}"
SP="${SPARSIFY[$SLURM_ARRAY_TASK_ID]}"

if [ "$SP" = "on" ]; then
  SPARSE_ARGS=(
    --sparsify_errors
    --sparsify-base-degree 2
    --sparsify-max-degree -1
    --sparsify-reactivate-limit -1
  )
else
  SPARSE_ARGS=(
    --no-sparsify_errors
    --sparsify-base-degree -1
    --sparsify-max-degree -1
    --sparsify-reactivate-limit -1
  )
fi

bazel run //src/py:run_tesseract -- \
  --decode-mode batch \
  --workers 1 \
  --threads 64 \
  --n-shots 500000 \
  --det-beam 20 \
  --beam-climbing \
  --merge-errors \
  --pqlimit 200000 \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --basis surface_code_X \
  --distances "$DIST" \
  --p-values 0.001 \
  --run-group sparsify \
  "${SPARSE_ARGS[@]}"
