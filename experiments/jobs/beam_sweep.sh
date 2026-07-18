#!/bin/bash
#SBATCH --job-name=beam_sweep
#SBATCH --output=logs/beam_sweep_%A_%a.out
#SBATCH --error=logs/beam_sweep_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --array=0-4%1

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

BEAMS=(5 10 20 50 100)
BEAM="${BEAMS[$SLURM_ARRAY_TASK_ID]}"

bazel run //src/py:run_tesseract -- \
  --decode-mode batch \
  --workers 1 \
  --threads 64 \
  --n-shots 100000 \
  --det-beam "$BEAM" \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --basis surface_code_X \
  --distances 3 5 7 9 11 \
  --p-values 0.0005 0.001 0.002
