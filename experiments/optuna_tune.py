from pathlib import Path
import optuna
from experiments.run_tesseract import analyse_one_circuit

STIM = Path("testdata/surfacecodes/r=11,d=11,p=0.002,noise=si1000,c=surface_code_X,q=241,gates=cz.stim")


def objective(trial):
    det_beam = trial.suggest_categorical("det_beam", [5, 10, 20, 50, 100])
    beam_climbing = trial.suggest_categorical("beam_climbing", [False, True])

    result = analyse_one_circuit(
        n_shots=10000,
        stim_path=STIM,
        verbose_histograms=False,
        print_every=0,
        decode_mode="batch",
        workers=1,
        threads=64,
        det_beam=det_beam,
        beam_climbing=beam_climbing,
        merge_errors=False,   # fixed for now
        pqlimit=200000,
        det_penalty=0.0,
    )

    return result.logical_error_rate_per_round


def main() -> None:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=4)

    print("Best params: ", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == "__main__":
    main()
