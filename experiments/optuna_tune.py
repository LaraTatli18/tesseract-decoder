from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import fmean

import optuna


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "experiments" / "runs"
STUDIES_ROOT = ROOT / "experiments" / "optuna_studies"


DEFAULT_DET_BEAM_CANDIDATES = [5, 10, 20, 50, 100]
DEFAULT_PQLIMIT_CANDIDATES = [100_000, 200_000, 400_000, 800_000, 1_600_000]
DEFAULT_WORKERS_CANDIDATES = [1, 2, 4]
DEFAULT_THREADS_CANDIDATES = [16, 32, 48, 64]
DEFAULT_SPARSIFY_BASE_DEGREE_CANDIDATES = [1, 2, 3, 4]
DEFAULT_SPARSIFY_MAX_DEGREE_CANDIDATES = [-1, 3, 4, 5, 6, 7, 8]
DEFAULT_SPARSIFY_REACTIVATE_LIMIT_CANDIDATES = [-1, 32, 64, 128, 256]


@dataclass(frozen=True)
class BenchmarkCase:
    basis: str
    distance: int
    p_value: float


@dataclass(frozen=True)
class TrialParams:
    det_beam: int
    beam_climbing: bool
    pqlimit: int
    merge_errors: bool
    workers: int
    threads: int
    sparsify_errors: bool
    sparsify_base_degree: int
    sparsify_max_degree: int
    sparsify_reactivate_limit: int


@dataclass(frozen=True)
class TrialMetrics:
    mean_decode_time_seconds: float
    mean_shots_per_second: float
    mean_logical_error_rate_per_round: float
    mean_low_confidence_rate: float


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_")


def _format_p_value(p: float) -> str:
    return f"p{p:g}".replace(".", "p")


def _mean_from_rows(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    if not values:
        raise RuntimeError(f"No values found for column {key!r}")
    return fmean(values)


def _load_results_csv(results_csv: Path) -> list[dict[str, str]]:
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"{results_csv} is empty")
    return rows


def _find_results_csv(run_group_dir: Path) -> Path:
    candidates = sorted(
        run_group_dir.rglob("results.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No results.csv found under {run_group_dir}")
    return candidates[-1]


def _suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> TrialParams:
    det_beam = (
        trial.suggest_categorical("det_beam", args.det_beam_candidates)
        if args.tune_det_beam
        else args.det_beam
    )

    beam_climbing = (
        trial.suggest_categorical("beam_climbing", [False, True])
        if args.tune_beam_climbing
        else args.beam_climbing
    )

    pqlimit = (
        trial.suggest_categorical("pqlimit", args.pqlimit_candidates)
        if args.tune_pqlimit
        else args.pqlimit
    )

    merge_errors = (
        trial.suggest_categorical("merge_errors", [False, True])
        if args.tune_merge_errors
        else args.merge_errors
    )

    workers = (
        trial.suggest_categorical("workers", args.workers_candidates)
        if args.tune_workers
        else args.workers
    )

    threads = (
        trial.suggest_categorical("threads", args.threads_candidates)
        if args.tune_threads
        else args.threads
    )

    sparsify_errors = (
        trial.suggest_categorical("sparsify_errors", [False, True])
        if args.tune_sparsify
        else args.sparsify_errors
    )

    if sparsify_errors:
        sparsify_base_degree = (
            trial.suggest_categorical(
                "sparsify_base_degree",
                args.sparsify_base_degree_candidates,
            )
            if args.tune_sparsify_base_degree
            else args.sparsify_base_degree
        )
        sparsify_max_degree = (
            trial.suggest_categorical(
                "sparsify_max_degree",
                args.sparsify_max_degree_candidates,
            )
            if args.tune_sparsify_max_degree
            else args.sparsify_max_degree
        )
        sparsify_reactivate_limit = (
            trial.suggest_categorical(
                "sparsify_reactivate_limit",
                args.sparsify_reactivate_limit_candidates,
            )
            if args.tune_sparsify_reactivate_limit
            else args.sparsify_reactivate_limit
        )
    else:
        sparsify_base_degree = args.sparsify_base_degree
        sparsify_max_degree = args.sparsify_max_degree
        sparsify_reactivate_limit = args.sparsify_reactivate_limit

    return TrialParams(
        det_beam=det_beam,
        beam_climbing=beam_climbing,
        pqlimit=pqlimit,
        merge_errors=merge_errors,
        workers=workers,
        threads=threads,
        sparsify_errors=sparsify_errors,
        sparsify_base_degree=sparsify_base_degree,
        sparsify_max_degree=sparsify_max_degree,
        sparsify_reactivate_limit=sparsify_reactivate_limit,
    )


def _run_single_case(
    case: BenchmarkCase,
    params: TrialParams,
    args: argparse.Namespace,
    trial: optuna.Trial,
) -> TrialMetrics:
    run_group = (
        f"optuna/{_slugify(args.study_name)}/"
        f"trial_{trial.number:04d}/"
        f"d{case.distance}_p{case.p_value:g}"
    )

    cmd = [
        "bazel",
        "run",
        "//src/py:run_tesseract",
        "--",
        "--run-group",
        run_group,
        "--decode-mode",
        "batch",
        "--max-files",
        str(args.max_files),
        "--n-shots",
        str(args.n_shots),
        "--workers",
        str(params.workers),
        "--threads",
        str(params.threads),
        "--basis",
        case.basis,
        "--distances",
        str(case.distance),
        "--p-values",
        str(case.p_value),
        "--det-beam",
        str(params.det_beam),
        "--pqlimit",
        str(params.pqlimit),
        "--stim-dir",
        str(args.stim_dir),
    ]

    if params.beam_climbing:
        cmd.append("--beam-climbing")
    else:
        cmd.append("--no-beam-climbing")

    if params.merge_errors:
        cmd.append("--merge-errors")
    else:
        cmd.append("--no-merge-errors")

    if params.sparsify_errors:
        cmd.append("--sparsify_errors")
    else:
        cmd.append("--no-sparsify_errors")

    cmd += [
        "--sparsify-base-degree",
        str(params.sparsify_base_degree),
        "--sparsify-max-degree",
        str(params.sparsify_max_degree),
        "--sparsify-reactivate-limit",
        str(params.sparsify_reactivate_limit),
    ]

    print(
        f"Trial {trial.number}: running d={case.distance}, p={case.p_value:g} "
        f"with params={asdict(params)}",
        flush=True,
    )

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    run_group_dir = RUNS_ROOT / Path(run_group)
    results_csv = _find_results_csv(run_group_dir)
    rows = _load_results_csv(results_csv)

    decode_time_seconds = _mean_from_rows(rows, "decode_time_seconds")
    shots_per_second = _mean_from_rows(rows, "shots_per_second")
    logical_error_rate_per_round = _mean_from_rows(rows, "logical_error_rate_per_round")

    if "low_confidence_rate" in rows[0]:
        low_confidence_rate = _mean_from_rows(rows, "low_confidence_rate")
    else:
        low_confidence_rate = 0.0

    trial.set_user_attr("run_group", run_group)
    trial.set_user_attr("results_csv", str(results_csv))
    trial.set_user_attr("decode_time_seconds", decode_time_seconds)
    trial.set_user_attr("shots_per_second", shots_per_second)
    trial.set_user_attr("logical_error_rate_per_round", logical_error_rate_per_round)
    trial.set_user_attr("low_confidence_rate", low_confidence_rate)

    if args.verbose:
        print(proc.stdout, flush=True)

    print(
        f"Trial {trial.number}: done d={case.distance}, p={case.p_value:g} "
        f"runtime={decode_time_seconds:.6f}s ler={logical_error_rate_per_round:.8e}",
        flush=True,
    )

    return TrialMetrics(
        mean_decode_time_seconds=decode_time_seconds,
        mean_shots_per_second=shots_per_second,
        mean_logical_error_rate_per_round=logical_error_rate_per_round,
        mean_low_confidence_rate=low_confidence_rate,
    )


def _objective_scalar(
    trial: optuna.Trial,
    args: argparse.Namespace,
    metric_name: str,
) -> float:
    params = _suggest_params(trial, args)

    case = BenchmarkCase(
        basis=args.basis,
        distance=args.distance,
        p_value=args.p_value,
    )

    metrics = _run_single_case(case, params, args, trial)

    mean_decode_time_seconds = metrics.mean_decode_time_seconds
    mean_shots_per_second = metrics.mean_shots_per_second
    mean_logical_error_rate_per_round = metrics.mean_logical_error_rate_per_round
    mean_low_confidence_rate = metrics.mean_low_confidence_rate

    trial.set_user_attr("params", asdict(params))
    trial.set_user_attr("mean_decode_time_seconds", mean_decode_time_seconds)
    trial.set_user_attr("mean_shots_per_second", mean_shots_per_second)
    trial.set_user_attr(
        "mean_logical_error_rate_per_round",
        mean_logical_error_rate_per_round,
    )
    trial.set_user_attr("mean_low_confidence_rate", mean_low_confidence_rate)

    if metric_name == "runtime":
        return mean_decode_time_seconds
    if metric_name == "quality":
        return mean_logical_error_rate_per_round

    raise ValueError(f"Unknown scalar objective: {metric_name}")


def _objective_pareto(
    trial: optuna.Trial,
    args: argparse.Namespace,
) -> tuple[float, float]:
    params = _suggest_params(trial, args)

    case = BenchmarkCase(
        basis=args.basis,
        distance=args.distance,
        p_value=args.p_value,
    )

    metrics = _run_single_case(case, params, args, trial)

    mean_decode_time_seconds = metrics.mean_decode_time_seconds
    mean_shots_per_second = metrics.mean_shots_per_second
    mean_logical_error_rate_per_round = metrics.mean_logical_error_rate_per_round
    mean_low_confidence_rate = metrics.mean_low_confidence_rate

    trial.set_user_attr("params", asdict(params))
    trial.set_user_attr("mean_decode_time_seconds", mean_decode_time_seconds)
    trial.set_user_attr("mean_shots_per_second", mean_shots_per_second)
    trial.set_user_attr(
        "mean_logical_error_rate_per_round",
        mean_logical_error_rate_per_round,
    )
    trial.set_user_attr("mean_low_confidence_rate", mean_low_confidence_rate)

    return mean_decode_time_seconds, mean_logical_error_rate_per_round


def _print_summary(study: optuna.Study, objective_mode: str) -> None:
    print("\nStudy summary")
    print("=============")
    print(f"Name: {study.study_name}")
    print(f"Trials: {len(study.trials)}")
    print(f"Objective: {objective_mode}")

    if objective_mode == "pareto":
        print("\nPareto front:")
        for t in sorted(study.best_trials, key=lambda tr: tr.values):
            print(
                f"trial={t.number} "
                f"values={tuple(round(v, 8) for v in t.values)} "
                f"params={t.params}"
            )
    else:
        print("\nBest trial:")
        print(f"trial={study.best_trial.number}")
        print(f"value={study.best_trial.value}")
        print(f"params={study.best_trial.params}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna tuner for Tesseract parameters.")

    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the Optuna study. Defaults to an auto-generated name.",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=STUDIES_ROOT,
        help="Directory for Optuna SQLite storage.",
    )
    parser.add_argument(
        "--objective",
        choices=["pareto", "runtime", "quality"],
        default="pareto",
        help="Optimisation mode.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print run output from each trial.",
    )

    parser.add_argument(
        "--basis",
        type=str,
        default="surface_code_X",
        help="Code basis to benchmark.",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=11,
        help="Single distance used for this Optuna study.",
    )
    parser.add_argument(
        "--p-value",
        type=float,
        default=0.002,
        help="Single physical error rate used for this Optuna study.",
    )
    parser.add_argument(
        "--stim-dir",
        type=Path,
        default=ROOT / "testdata" / "surfacecodes",
        help="Directory containing Stim files.",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=50_000,
        help="Shots per benchmark run.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Maximum files per batch run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Default worker count.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Default thread count.",
    )

    parser.add_argument(
        "--det-beam",
        type=int,
        default=10,
        help="Default beam size when beam tuning is disabled.",
    )
    parser.add_argument(
        "--pqlimit",
        type=int,
        default=200_000,
        help="Default priority queue limit when pqlimit tuning is disabled.",
    )
    parser.add_argument(
        "--beam-climbing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default beam-climbing setting when tuning is disabled.",
    )
    parser.add_argument(
        "--merge-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Default merge-errors setting when tuning is disabled.",
    )
    parser.add_argument(
        "--sparsify_errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default sparsification setting when tuning is disabled.",
    )
    parser.add_argument(
        "--sparsify-base-degree",
        type=int,
        default=2,
        help="Default sparsify base degree.",
    )
    parser.add_argument(
        "--sparsify-max-degree",
        type=int,
        default=-1,
        help="Default sparsify max degree.",
    )
    parser.add_argument(
        "--sparsify-reactivate-limit",
        type=int,
        default=-1,
        help="Default sparsify reactivate limit.",
    )

    parser.add_argument(
        "--tune-det-beam",
        action="store_true",
        help="Let Optuna tune det_beam.",
    )
    parser.add_argument(
        "--tune-beam-climbing",
        action="store_true",
        help="Let Optuna tune beam_climbing.",
    )
    parser.add_argument(
        "--tune-pqlimit",
        action="store_true",
        help="Let Optuna tune pqlimit.",
    )
    parser.add_argument(
        "--tune-merge-errors",
        action="store_true",
        help="Let Optuna tune merge_errors.",
    )
    parser.add_argument(
        "--tune-workers",
        action="store_true",
        help="Let Optuna tune workers.",
    )
    parser.add_argument(
        "--tune-threads",
        action="store_true",
        help="Let Optuna tune threads.",
    )
    parser.add_argument(
        "--tune-sparsify",
        action="store_true",
        help="Let Optuna tune sparsify_errors.",
    )
    parser.add_argument(
        "--tune-sparsify-base-degree",
        action="store_true",
        help="Let Optuna tune sparsify_base_degree.",
    )
    parser.add_argument(
        "--tune-sparsify-max-degree",
        action="store_true",
        help="Let Optuna tune sparsify_max_degree.",
    )
    parser.add_argument(
        "--tune-sparsify-reactivate-limit",
        action="store_true",
        help="Let Optuna tune sparsify_reactivate_limit.",
    )

    parser.add_argument(
        "--det-beam-candidates",
        type=int,
        nargs="+",
        default=DEFAULT_DET_BEAM_CANDIDATES,
        help="Candidate beam sizes.",
    )
    parser.add_argument(
        "--pqlimit-candidates",
        type=int,
        nargs="+",
        default=DEFAULT_PQLIMIT_CANDIDATES,
        help="Candidate PQ limits.",
    )
    parser.add_argument(
        "--workers-candidates",
        type=int,
        nargs="+",
        default=DEFAULT_WORKERS_CANDIDATES,
        help="Candidate worker counts.",
    )
    parser.add_argument(
        "--threads-candidates",
        type=int,
        nargs="+",
        default=DEFAULT_THREADS_CANDIDATES,
        help="Candidate thread counts.",
    )
    parser.add_argument(
        "--sparsify-base-degree-candidates",
        type=int,
        nargs="+",
        default=DEFAULT_SPARSIFY_BASE_DEGREE_CANDIDATES,
        help="Candidate sparsify base degrees.",
    )
    parser.add_argument(
        "--sparsify-max-degree-candidates",
        type=int,
        nargs="+",
        default=DEFAULT_SPARSIFY_MAX_DEGREE_CANDIDATES,
        help="Candidate sparsify max degrees.",
    )
    parser.add_argument(
        "--sparsify-reactivate-limit-candidates",
        type=int,
        nargs="+",
        default=DEFAULT_SPARSIFY_REACTIVATE_LIMIT_CANDIDATES,
        help="Candidate sparsify reactivate limits.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.study_name is None:
        p_tag = _format_p_value(args.p_value)
        d_tag = f"d{args.distance}"
        args.study_name = f"{_slugify(args.basis)}_{args.objective}_{d_tag}_{p_tag}"

    args.storage_dir.mkdir(parents=True, exist_ok=True)
    storage_path = args.storage_dir / f"{_slugify(args.study_name)}.db"
    storage = f"sqlite:///{storage_path}"

    if args.objective == "pareto":
        sampler = optuna.samplers.NSGAIISampler(seed=args.seed)
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            directions=["minimize", "minimize"],
            sampler=sampler,
            load_if_exists=True,
        )
        study.optimize(lambda trial: _objective_pareto(trial, args), n_trials=args.n_trials)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: _objective_scalar(trial, args, args.objective),
            n_trials=args.n_trials,
        )

    _print_summary(study, args.objective)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
