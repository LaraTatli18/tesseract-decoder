#!/bin/bash
#SBATCH --job-name=thread_sweep
#SBATCH --output=logs/thread_sweep_%A_%a.out
#SBATCH --error=logs/thread_sweep_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --array=0-20%1

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

REPEATS=(1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 3 3 3 3 3)
THREADS=(1 2 4 8 16 32 64 1 2 4 8 16 32 64 1 2 4 8 16 32 64)

IDX=$SLURM_ARRAY_TASK_ID
REP="${REPEATS[$IDX]}"
T="${THREADS[$IDX]}"

echo "=================================================="
echo "Thread scaling benchmark"
echo "repeat  = $REP"
echo "threads = $T"
echo "distance = 11"
echo "p        = 0.002"
echo "beam     = 5"
echo "pqlimit  = 200000"
echo "=================================================="

bazel run //src/py:run_tesseract -- \
  --decode-mode batch \
  --workers 1 \
  --threads "$T" \
  --n-shots 100000 \
  --stim-dir "$(pwd)/testdata/surfacecodes" \
  --basis surface_code_X \
  --distances 11 \
  --p-values 0.002 \
  --det-beam 5 \
  --beam-climbing \
  --merge-errors \
  --pqlimit 200000 \
  --max-files 1 \
  --run-group "thread_sweep_d11_p002_rep${REP}"
