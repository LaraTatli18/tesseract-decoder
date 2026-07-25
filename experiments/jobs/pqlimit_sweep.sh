#!/bin/bash
#SBATCH --job-name=pqlimit
#SBATCH --partition=main
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --array=0-17%1
#SBATCH --output=logs/pqlimit_%A_%a.out
#SBATCH --error=logs/pqlimit_%A_%a.err

set -euo pipefail
cd ~/tesseract-decoder
source .venv/bin/activate

BEAMS=(
  20 20 20 20 20 20
  50 50 50 50 50 50
  100 100 100 100 100 100
)

PQLIMITS=(
  50000 100000 200000 400000 800000 1600000
  50000 100000 200000 400000 800000 1600000
  50000 100000 200000 400000 800000 1600000

)

BEAM="${BEAMS[$SLURM_ARRAY_TASK_ID]}"
PQLIMIT="${PQLIMITS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================="
echo "det_beam = $BEAM"
echo "pqlimit  = $PQLIMIT"
echo "========================================="

bazel run //src/py:run_tesseract -- \
  --run-group pqlimit_sweep \
  --basis surface_code_X \
  --distances 3 5 7 9 11 \
  --p-values 0.0005 0.001 0.002 \
  --n-shots 100000 \
  --workers 1 \
  --threads 64 \
  --det-beam "$BEAM" \
  --pqlimit "$PQLIMIT" \
  --beam-climbing \
  --merge-errors
