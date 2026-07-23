#!/bin/bash
#SBATCH --job-name=merge_err
#SBATCH --output=logs/merge_err_%A_%a.out
#SBATCH --error=logs/merge_err_%A_%a.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-1%1

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

DET_BEAM=20

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    EXTRA_FLAGS=(--no-merge-errors)
    MERGE_TAG="merge0"
else
    EXTRA_FLAGS=(--merge-errors)
    MERGE_TAG="merge1"
fi

echo "Running merge_errors sweep:"
echo "  det_beam = $DET_BEAM"
echo "  beam_climbing = ON"
echo "  merge_errors = $MERGE_TAG"

bazel run //src/py:run_tesseract -- \
    --decode-mode batch \
    --workers 1 \
    --threads 64 \
    --n-shots 250000 \
    --det-beam "$DET_BEAM" \
    --beam-climbing \
    "${EXTRA_FLAGS[@]}" \
    --stim-dir "$(pwd)/testdata/surfacecodes" \
    --basis surface_code_X \
    --distances 3 5 7 9 11 \
    --p-values 0.0001 0.0005 0.001 0.002
