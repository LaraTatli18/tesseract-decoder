#!/bin/bash
#SBATCH --job-name=sparsify_sweep
#SBATCH --output=logs/sparsify_sweep_%A_%a.out
#SBATCH --error=logs/sparsify_sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --array=0-29%2

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

DISTANCES=(3 5 7 9 11)
PVALUES=(0.0005 0.001 0.002)
SPARSIFY=(off on)

DIST_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
REST=$((SLURM_ARRAY_TASK_ID % 6))
PV_INDEX=$((REST / 2))
SP_INDEX=$((REST % 2))

DIST="${DISTANCES[$DIST_INDEX]}"
PVALUE="${PVALUES[$PV_INDEX]}"
SP="${SPARSIFY[$SP_INDEX]}"

echo "========================================"
echo "Task $SLURM_ARRAY_TASK_ID"
echo "distance = $DIST"
echo "p        = $PVALUE"
echo "sparsify = $SP"
echo "========================================"

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
  --max-files 1 \
  --det-beam 5 \
  --beam-climbing \
  --merge-errors \
  --pqlimit 200000 \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --basis surface_code_X \
  --distances "$DIST" \
  --p-values "$PVALUE" \
  --run-group sparsify_sweep \
  "${SPARSE_ARGS[@]}"
