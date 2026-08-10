#!/bin/bash
#SBATCH --job-name=optuna_beam_pq
#SBATCH --partition=main
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --output=logs/optuna_beam_pq_%j.out
#SBATCH --error=logs/optuna_beam_pq_%j.err

set -euo pipefail

cd ~/tesseract-decoder
source .venv/bin/activate

python experiments/optuna_tune.py \
  --objective pareto \
  --study-name surface_code_x_beam_pqlimit_optuna \
  --basis surface_code_X \
  --distance 11 \
  --p-value 0.002 \
  --n-trials 80 \
  --n-shots 50000 \
  --max-logical-error-rate-per-round 5e-5 \
  --threads 32 \
  --det-beam 50 \
  --pqlimit 200000 \
  --beam-climbing \
  --merge-errors \
  --tune-det-beam \
  --tune-pqlimit \
  --det-beam-candidates 10 15 20 25 30 40 \
  --pqlimit-candidates 5000 10000 20000 30000 50000 75000 100000 200000 400000 800000 1600000
