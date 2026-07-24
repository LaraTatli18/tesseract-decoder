#!/bin/bash
#SBATCH --job-name=optuna
#SBATCH --partition=main
#SBATCH --cpus-per-task=64
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/optuna_%j.out
#SBATCH --error=logs/optuna_%j.err

set -euo pipefail

cd ~/tesseract-decoder

source .venv/bin/activate

echo "Starting Optuna study..."
date

python experiments/optuna_tune.py

echo "Finished."
date
