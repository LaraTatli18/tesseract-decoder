from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)

def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_first_row(path: Path) -> dict[str, str] | None:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return next(reader, None)


def setup_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlepad": 10,
        }
    )


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), dpi=300)

def half_shot_floor(n_shots: int | float) -> float:
    """Return the usual 0.5-shot floor used on log-scale LER plots."""
    return 0.5 / max(int(n_shots), 1)

def format_shot_count(n_shots: int) -> str:
    if n_shots >= 1_000_000:
        return f"{n_shots // 1_000_000}M"
    if n_shots >= 1_000:
        return f"{n_shots // 1_000}k"
    return str(n_shots)

def format_number(value: float) -> str:
    """Format a numeric parameter value compactly for labels."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"

def format_p_value(p_value: float) -> str:
    """Format a physical error rate for use in filenames."""
    return f"p{p_value:g}".replace(".", "p")

def create_plot_run(
    parameter: str,
    basis: str,
    n_shots: int,
    run_dirs: list[Path],
    script_name: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    shots = format_shot_count(n_shots)

    output_dir = Path("experiments/plots") / f"{timestamp}_{parameter}_{shots}_{basis}"
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "timestamp": timestamp,
        "plot_script": script_name,
        "parameter": parameter,
        "basis": basis,
        "n_shots": n_shots,
        "run_dirs": [str(r) for r in run_dirs],
    }

    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=4)

    return output_dir


def make_plot_filename(
    parameter: str,
    basis: str,
    n_shots: int,
    distance: int,
) -> str:
    shots = format_shot_count(n_shots)
    return f"{parameter}_{shots}_{basis}_d{distance}.png"

def safe_filename_component(text: str) -> str:
    """Replace filename-unfriendly characters with underscores."""
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in text
    ).strip("_")


def expand_run_dirs(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        if not p.exists() or p.is_file():
            continue

        if (p / "manifest.json").exists() and (p / "results.csv").exists():
            if p not in seen:
                expanded.append(p)
                seen.add(p)
            continue

        for child in sorted(p.iterdir()):
            if child.is_dir() and (child / "manifest.json").exists() and (child / "results.csv").exists():
                if child not in seen:
                    expanded.append(child)
                    seen.add(child)
    return expanded

@dataclass(frozen=True)
class PlotRun:
    """A benchmark run loaded for plotting."""
    run_dir: Path
    manifest: dict[str, Any]
    rows: list[dict[str, str]]


def collect_plot_runs(run_dirs: Iterable[Path]) -> list[PlotRun]:
    """Load valid manifest/results pairs from benchmark run directories."""
    runs: list[PlotRun] = []

    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        results_path = run_dir / "results.csv"

        if not manifest_path.exists() or not results_path.exists():
            print(
                f"Skipping {run_dir}: "
                "missing manifest.json or results.csv"
            )
            continue

        try:
            manifest = load_manifest(manifest_path)
        except ValueError:
            print(f"Skipping {run_dir}: invalid manifest.json")
            continue

        rows = load_rows(results_path)
        if not rows:
            print(f"Skipping {run_dir}: results.csv is empty")
            continue

        runs.append(
            PlotRun(
                run_dir=run_dir,
                manifest=manifest,
                rows=rows,
            )
        )

    return runs


def value_as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def as_float_list(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values]


def check_manifests_consistent(
    manifests: Iterable[dict[str, Any]],
    varying_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Require all manifest fields to match except those explicitly allowed to vary."""
    manifests = list(manifests)
    if not manifests:
        raise ValueError("No manifests were supplied.")

    varying_fields = set(varying_fields or set())
    ignored_fields = {
        "timestamp",
        "output_csv",
        "command",
        "git_commit",
        "git_branch",
        "working_tree_clean",
    } | varying_fields

    reference = manifests[0]
    reference_keys = set(reference) - ignored_fields

    for manifest in manifests[1:]:
        for key in reference_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(
                    f"Manifest field {key!r} differs across runs.\n"
                    f"Reference: {reference.get(key)!r}\n"
                    f"Current:    {manifest.get(key)!r}"
                )

    return {key: reference.get(key) for key in reference_keys}


def require_manifest_fields(
    manifest: dict[str, Any],
    fields: Iterable[str],
    *,
    source: Path | None = None,
) -> None:
    missing = [field for field in fields if field not in manifest]
    if missing:
        location = f" in {source}" if source is not None else ""
        raise ValueError(f"Missing manifest fields{location}: {missing}")


def get_manifest_value(manifest: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in manifest:
        return manifest[key]
    circuit_metadata = manifest.get("circuit_metadata")
    if isinstance(circuit_metadata, dict) and key in circuit_metadata:
        return circuit_metadata[key]
    return default

def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Could not parse boolean value from {value!r}")


def matches_filters(manifest: dict[str, Any], args: argparse.Namespace) -> bool:
    if getattr(args, "basis", None) is not None and manifest.get("basis") != args.basis:
        return False
    if getattr(args, "decode_mode", None) is not None and manifest.get("decode_mode") != args.decode_mode:
        return False
    if getattr(args, "workers", None) is not None and int(manifest.get("workers", -1)) != args.workers:
        return False
    if getattr(args, "threads", None) is not None and int(manifest.get("threads", -1)) != args.threads:
        return False
    if getattr(args, "n_shots", None) is not None and int(manifest.get("n_shots", -1)) != args.n_shots:
        return False

    if getattr(args, "distances", None) is not None:
        manifest_distances = value_as_list(manifest.get("distances", []))
        if as_float_list(manifest_distances) != [float(d) for d in args.distances]:
            return False

    if getattr(args, "p_values", None) is not None:
        manifest_p_values = value_as_list(manifest.get("p_values", []))
        if as_float_list(manifest_p_values) != [float(p) for p in args.p_values]:
            return False

    if getattr(args, "sparsify_errors", None) is not None and manifest.get("sparsify_errors") != args.sparsify_errors:
        return False
    if getattr(args, "sparsify_base_degree", None) is not None and int(manifest.get("sparsify_base_degree", -999999)) != args.sparsify_base_degree:
        return False
    if getattr(args, "sparsify_max_degree", None) is not None and int(manifest.get("sparsify_max_degree", -999999)) != args.sparsify_max_degree:
        return False
    if getattr(args, "sparsify_reactivate_limit", None) is not None and int(manifest.get("sparsify_reactivate_limit", -999999)) != args.sparsify_reactivate_limit:
        return False

    if getattr(args, "det_beam", None) is not None and int(manifest.get("det_beam", -1)) != args.det_beam:
        return False
    if getattr(args, "beam_climbing", None) is not None and bool(manifest.get("beam_climbing")) != args.beam_climbing:
        return False
    if getattr(args, "merge_errors", None) is not None and bool(manifest.get("merge_errors")) != args.merge_errors:
        return False
    if getattr(args, "pqlimit", None) is not None and int(manifest.get("pqlimit", -1)) != args.pqlimit:
        return False

    return True
