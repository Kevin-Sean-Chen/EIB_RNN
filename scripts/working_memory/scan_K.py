"""Scan spatial working-memory performance across K with RLS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.analysis.working_memory import run_working_memory_K_scan
from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.io import create_run_directory, runtime_metadata, save_csv, save_json, save_npz
from src.tasks.working_memory import WORKING_MEMORY_SECTIONS, WorkingMemoryConfig


METRIC_FIELDS = ["K", "output_mse", "memory_mse", "output_r2", "memory_r2"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/working_memory/K_scan.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    """Load one working-memory K scan."""
    config, document = load_dataclass_sections(
        args.config, WorkingMemoryConfig, tuple(WORKING_MEMORY_SECTIONS),
    )
    run = document.get("run", {})
    scan = document.get("scan", {})
    if not isinstance(run, dict) or not isinstance(scan, dict):
        raise ValueError("The run and scan sections must be mappings.")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    K_values = [float(value) for value in scan.get("K_values", [])]
    if not K_values:
        raise ValueError("K_values must not be empty.")
    return config, run, K_values


def plot_result(result, learning_method: str) -> plt.Figure:
    """Plot post-go MSE and R2 across K."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(result.K_values, result.output_mse, "o-", label="Output")
    axes[0].plot(result.K_values, result.memory_mse, "o-", label="Memory")
    axes[0].set(xscale="log", xlabel="K", ylabel="MSE", title="Post-go error")
    axes[0].legend()
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].plot(result.K_values, result.output_r2, "o-", label="Output")
    axes[1].plot(result.K_values, result.memory_r2, "o-", label="Memory")
    axes[1].set(xscale="log", xlabel="K", ylabel="R2", title="Post-go score")
    axes[1].legend()
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    figure.suptitle(f"{learning_method.upper()} working memory across K")
    return figure


def main() -> None:
    """Run the scan and save all outputs."""
    args = parse_args()
    config, run, K_values = load_config(args)
    result = run_working_memory_K_scan(config, K_values)
    output_root = Path(run.get("output_root", "output/scans"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_directory = create_run_directory(
        output_root, str(run.get("experiment", "working_memory_K")), run.get("run_id"),
    )
    figure = plot_result(result, config.learning_method)
    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, WORKING_MEMORY_SECTIONS))
    resolved["scan"] = {"K_values": K_values}
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(run_directory / "results.npz", result.__dict__)
    rows = [
        {field: (K if field == "K" else getattr(result, field)[index]) for field in METRIC_FIELDS}
        for index, K in enumerate(result.K_values)
    ]
    save_csv(run_directory / "metrics.csv", METRIC_FIELDS, rows)
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))
    figure.savefig(run_directory / "summary.png", dpi=180)
    print(f"Saved scan to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
