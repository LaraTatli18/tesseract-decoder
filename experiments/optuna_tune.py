import csv
import re
import subprocess
from pathlib import Path
import optuna

ROOT = Path(__file__).resolve().parents[1]

STIM_FILE = ROOT / "testdata" / "surfacecodes" / "r=11,d=11,p=0.002,noise=si1000,c=surface_code_X,q=241,gates=cz.stim"


def run_trial(det_beam: int, beam_climbing: bool) -> float:
    cmd = [
        "bazel", "run", "//src/py:run_tesseract", "--",
        "--run-group", "optuna",
        "--n-shots", "50000",
        "--decode-mode", "batch",
        "--workers", "1",
        "--threads", "64",
        "--basis", "surface_code_X",
        "--distances", "11",
        "--p-values", "0.002",
        "--det-beam", str(det_beam),
        "--beam-climbing" if beam_climbing else "--no-beam-climbing",
        "--stim-dir", str(ROOT / "testdata" / "surfacecodes"),
    ]

    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)

    # Find the run directory that run_tesseract printed.
    m = re.search(r"Run directory\s*:\s*(.+)", proc.stdout)
    if not m:
        raise RuntimeError(f"Could not find run directory in output:\n{proc.stdout}")

    run_dir = Path(m.group(1).strip())
    results_csv = run_dir / "results.csv"

    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("results.csv is empty")

    return float(rows[0]["logical_error_rate_per_round"])


def objective(trial):
    det_beam = trial.suggest_categorical("det_beam", [5, 10, 20, 50, 100])
    beam_climbing = trial.suggest_categorical("beam_climbing", [False, True])
    return run_trial(det_beam, beam_climbing)


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == "__main__":
    main()
