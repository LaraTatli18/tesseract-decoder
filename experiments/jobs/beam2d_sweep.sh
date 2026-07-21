#!/bin/bash
#SBATCH --job-name=beam2d
#SBATCH --output=logs/beam2d_%A_%a.out
#SBATCH --error=logs/beam2d_%A_%a.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%2

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

BEAMS=(5 10 20 50 100)

INDEX=$SLURM_ARRAY_TASK_ID

BEAM_INDEX=$((INDEX % 5))
CLIMB_INDEX=$((INDEX / 5))

DET_BEAM=${BEAMS[$BEAM_INDEX]}

if [[ $CLIMB_INDEX -eq 0 ]]; then
    EXTRA_FLAGS=(--no-beam-climbing)
else
    EXTRA_FLAGS=(--beam-climbing)
fi

echo "Running:"
echo "  det_beam = $DET_BEAM"
echo "  beam_climbing = $([[ $CLIMB_INDEX -eq 0 ]] && echo OFF || echo ON)"

bazel run //src/py:run_tesseract -- \
    --decode-mode batch \
    --workers 1 \
    --threads 64 \
    --n-shots 250000 \
    --det-beam "$DET_BEAM" \
    "${EXTRA_FLAGS[@]}" \
    --stim-dir "$(pwd)/testdata/surfacecodes" \
    --basis surface_code_X \
    --distances 3 5 7 9 11 \
    --p-values \
        0.0005 \
        0.00075 \
        0.001 \
        0.00125 \
        0.0015 \
        0.002
