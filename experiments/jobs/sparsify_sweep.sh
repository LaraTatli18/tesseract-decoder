#!/bin/bash
#SBATCH --job-name=sparsify_reattempt
#SBATCH --output=logs/sparsify_reattempt_%A_%a.out
#SBATCH --error=logs/sparsify_reattemptt_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --array=0-119%2

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

DISTANCES=(3 5 7 9 11)
PVALUES=(0.0005 0.001 0.002)
BEAMS=(5 10)
PQLIMITS=(200000 400000)
SPARSIFY=(off on)

DIST_INDEX=$((SLURM_ARRAY_TASK_ID / 24))
REST1=$((SLURM_ARRAY_TASK_ID % 24))

PV_INDEX=$((REST1 / 8))
REST2=$((REST1 % 8))

BEAM_INDEX=$((REST2 / 4))
REST3=$((REST2 % 4))

PQLIMIT_INDEX=$((REST3 / 2))
SP_INDEX=$((REST3 % 2))

DIST="${DISTANCES[$DIST_INDEX]}"
PVALUE="${PVALUES[$PV_INDEX]}"
BEAM="${BEAMS[$BEAM_INDEX]}"
PQLIMIT="${PQLIMITS[$PQLIMIT_INDEX]}"
SP="${SPARSIFY[$SP_INDEX]}"

echo "========================================"
echo "Task $SLURM_ARRAY_TASK_ID"
echo "distance = $DIST"
echo "p        = $PVALUE"
echo "beam     = $BEAM"
echo "pqlimit  = $PQLIMIT"
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
  --det-beam "$BEAM" \
  --beam-climbing \
  --merge-errors \
  --pqlimit "$PQLIMIT" \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --basis surface_code_X \
  --distances "$DIST" \
  --p-values "$PVALUE" \
  --run-group sparsify_robust \
  "${SPARSE_ARGS[@]}"
