from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys

from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _git_value(args: list[str]) -> str:
    """Return a git value, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _git_is_clean() -> bool:
    """Return True if the git working tree is clean."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        )
        return out.strip() == ""
    except Exception:
        return False


def _build_run_name(args: argparse.Namespace) -> str:
    """Create a short, filesystem-friendly run directory name."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    return (
        f"{timestamp}_"
        f"{args.basis}_"
        f"{args.decode_mode}_"
        f"{args.n_shots}shots_"
        f"{args.workers}workers_"
        f"{args.threads}threads_"
        f"beam{args.det_beam}"
    )


def make_run_directory(args: argparse.Namespace) -> tuple[Path, Path]:
    """Create the run directory and return (run_dir, manifest_path)."""
    run_name = _build_run_name(args)
    run_dir = ROOT / "experiments" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    return run_dir, manifest_path


def build_manifest(args: argparse.Namespace, output_csv: Path) -> dict[str, Any]:
    """Build a JSON-serialisable manifest for this run."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "git_branch": _git_value(["git", "branch", "--show-current"]),
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "working_tree_clean": _git_is_clean(),
        "command": ["bazel", "run", "//src/py:run_tesseract", "--",
                    *sys.argv[1:]],
        "output_csv": str(output_csv),
        "stim_dir": str(args.stim_dir),
        "basis": args.basis,
        "distances": list(args.distances),
        "p_values": list(args.p_values),
        "n_shots": args.n_shots,
        "decode_mode": args.decode_mode,
        "workers": args.workers,
        "threads": args.threads,
        "det_beam": args.det_beam,
        "beam_climbing": args.beam_climbing,
        "merge_errors": args.merge_errors,
        "pqlimit": args.pqlimit,
        "det_penalty": args.det_penalty,
    }


def write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Write the run manifest as pretty JSON."""
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
