#!/bin/bash
#SBATCH --job-name=beam_climb_250k
#SBATCH --output=logs/beam_climb_250k_%A_%a.out
#SBATCH --error=logs/beam_climb_250k_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --array=0-1%1

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

BEAM=20

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
  EXTRA_FLAGS=(--no-beam-climbing)
else
  EXTRA_FLAGS=(--beam-climbing)
fi

bazel run //src/py:run_tesseract -- \
  --decode-mode batch \
  --workers 1 \
  --threads 64 \
  --n-shots 250000 \
  --det-beam "$BEAM" \
  "${EXTRA_FLAGS[@]}" \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --basis surface_code_X \
  --distances 3 5 7 9 11 \
  --p-values 0.0005 0.00075 0.001 0.00125 0.0015 0.002
