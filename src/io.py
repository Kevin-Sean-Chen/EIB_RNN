"""Create run directories and save reproducibility files."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np


def create_run_directory(
    output_root: Path,
    experiment: str,
    run_id: str | None = None,
) -> Path:
    """Create and return one new run directory."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_directory = output_root / experiment / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def git_commit(repository_root: Path) -> str | None:
    """Return the current Git commit when it is available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_metadata(repository_root: Path) -> dict[str, Any]:
    """Return software and system information for one run."""
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(repository_root),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "command": sys.argv,
    }


def save_json(path: Path, content: Mapping[str, Any]) -> None:
    """Save a mapping as formatted JSON."""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)
        stream.write("\n")


def save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Save named arrays in one compressed NumPy file."""
    np.savez_compressed(path, **arrays)


def save_csv(
    path: Path,
    field_names: list[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Save rows in one CSV file."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
