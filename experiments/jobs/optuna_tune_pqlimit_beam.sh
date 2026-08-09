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
  --n-trials 200 \
  --n-shots 100000 \
  --max-logical-error-rate-per-round 5e-5 \
  --threads 32 \
  --det-beam 50 \
  --pqlimit 200000 \
  --beam-climbing \
  --merge-errors \
  --tune-det-beam \
  --tune-pqlimit \
  --det-beam-candidates 20 30 40 50 60 80 100 \
  --pqlimit-candidates 10000 25000 50000 75000 100000 150000 200000 300000 400000 600000 800000 1200000 1600000
