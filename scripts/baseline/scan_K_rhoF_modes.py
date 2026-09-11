"""Scan local-to-network mode dominance across K and rho_F."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.analysis.local_network_modes import (
    MODE_SCAN_SECTIONS,
    ModeScanConfig,
    ScanResult,
    run_scan,
    scan_metric_rows,
    scan_result_arrays,
)
from src.config import dataclass_to_sections, load_dataclass_sections, save_yaml
from src.io import (
    create_run_directory,
    runtime_metadata,
    save_csv,
    save_json,
    save_npz,
)


METRIC_FIELDS = [
    "K",
    "strength",
    "relative_strength",
    "spectral_strength",
    "network_advantage",
    "pca_dimension",
    "neighbor_correlation",
    "correlation_length",
    "low_k_fraction",
    "nonlocal_fraction",
    "null_fraction",
    "full_nonlocal_overlap",
    "lowrank_power",
    "lowrank_power_std",
    "total_variance",
    "total_variance_std",
    "lowrank_input_variance",
    "lowrank_input_variance_std",
    "lowrank_output_variance",
    "lowrank_output_variance_std",
    "excitatory_active_fraction",
    "excitatory_active_fraction_std",
    "inhibitory_active_fraction",
    "inhibitory_active_fraction_std",
    "excitatory_current_power",
    "excitatory_current_power_std",
    "inhibitory_current_power",
    "inhibitory_current_power_std",
    "net_current_power",
    "net_current_power_std",
    "ei_cancellation_ratio",
    "ei_cancellation_ratio_std",
    "ei_current_correlation",
    "ei_current_correlation_std",
    "mean_balance_e",
    "mean_balance_e_std",
    "mean_balance_i",
    "mean_balance_i_std",
    "active_local_balance_e",
    "active_local_balance_e_std",
    "active_local_balance_i",
    "active_local_balance_i_std",
    "inactive_local_balance_e",
    "inactive_local_balance_e_std",
    "inactive_local_balance_i",
    "inactive_local_balance_i_std",
    "mean_external_current_e",
    "mean_external_current_e_std",
    "mean_excitatory_current_e",
    "mean_excitatory_current_e_std",
    "mean_inhibitory_current_e",
    "mean_inhibitory_current_e_std",
    "mean_net_current_e",
    "mean_net_current_e_std",
    "mean_external_current_i",
    "mean_external_current_i_std",
    "mean_excitatory_current_i",
    "mean_excitatory_current_i_std",
    "mean_inhibitory_current_i",
    "mean_inhibitory_current_i_std",
    "mean_net_current_i",
    "mean_net_current_i_std",
    "stable",
]


def crossover_strength(
    rho_f: np.ndarray,
    advantage: np.ndarray,
    threshold: float,
) -> float:
    """Return the first interpolated rho_F above a threshold."""
    valid = np.isfinite(rho_f) & np.isfinite(advantage)
    x = rho_f[valid]
    y = advantage[valid]
    if x.size == 0 or np.all(y < threshold):
        return np.nan
    index = int(np.flatnonzero(y >= threshold)[0])
    if index == 0:
        return float(x[0])
    x0, x1 = x[index - 1], x[index]
    y0, y1 = y[index - 1], y[index]
    if np.isclose(y1, y0):
        return float(x1)
    fraction = (threshold - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def scan_all_K(
    config: ModeScanConfig,
    K_values: list[float],
) -> dict[float, ScanResult]:
    """Run matched rho_F scans for all requested K values."""
    results = {}
    for K in K_values:
        print(f"\nStart K={K:g}")
        results[float(K)] = run_scan(replace(config, K=K))
    return results


def plot_results(
    results: dict[float, ScanResult],
    config: ModeScanConfig,
    transition_threshold: float,
) -> plt.Figure:
    """Plot reconstruction and alignment diagnostics across K."""
    _ = transition_threshold
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_curves, ax_advantage, ax_alignment, ax_overlap = axes.ravel()

    reference_K = min(results, key=lambda value: abs(np.log(value / config.K)))
    reference = results[reference_K]
    finite = np.isfinite(reference.transition_index)
    stable_indices = np.flatnonzero(reference.stable & finite)
    if stable_indices.size == 0:
        selected = [0]
    else:
        peak = stable_indices[np.argmax(reference.transition_index[stable_indices])]
        selected = sorted(set([stable_indices[0], int(peak), stable_indices[-1]]))
    mode_limit = min(config.plot_modes, reference.local_curves.shape[1])
    mode_counts = reference.geometric_shell_counts[
        reference.geometric_shell_counts <= mode_limit
    ]
    for index in selected:
        color = ax_curves._get_lines.get_next_color()
        rho_f = reference.relative_strengths[index]
        mode_indices = mode_counts - 1
        ax_curves.plot(
            mode_counts,
            reference.local_curves[index, mode_indices],
            "--",
            color=color,
            label=f"Local, rho_F={rho_f:.2g}",
        )
        ax_curves.plot(
            mode_counts,
            reference.network_curves[index, mode_indices],
            "-",
            color=color,
            label=f"Full, rho_F={rho_f:.2g}",
        )
    ax_curves.set(
        xlabel="Number of modes",
        ylabel="Captured variance",
        ylim=(0.0, 1.02),
        title="Local and full reconstruction",
    )
    ax_curves.legend(fontsize=8, ncol=2)

    for K, result in results.items():
        color = ax_advantage._get_lines.get_next_color()
        x = result.relative_strengths
        y = result.transition_index
        ax_advantage.plot(x, y, "o-", color=color, label=f"K={K:g}")
        ax_advantage.fill_between(
            x,
            y - result.transition_std,
            y + result.transition_std,
            color=color,
            alpha=0.15,
        )
        ax_alignment.plot(
            x,
            result.nonlocal_fraction,
            "o-",
            color=color,
            label=f"Matched, K={K:g}",
        )
        ax_alignment.fill_between(
            x,
            result.nonlocal_fraction - result.nonlocal_fraction_std,
            result.nonlocal_fraction + result.nonlocal_fraction_std,
            color=color,
            alpha=0.15,
        )
        ax_alignment.plot(
            x,
            result.null_fraction,
            "--",
            color=color,
            label=f"Null, K={K:g}",
        )

    ax_advantage.axhline(0.0, color="black", linewidth=0.8)
    ax_advantage.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Full minus local variance",
        title="Full-mode advantage",
    )
    ax_advantage.legend()
    ax_alignment.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Variance in rank-r subspace",
        ylim=(0.0, 1.02),
        title="Non-local activity alignment",
    )
    ax_alignment.legend(fontsize=8, ncol=2)

    x = reference.relative_strengths
    y = reference.full_nonlocal_overlap
    ax_overlap.plot(x, y, "o-")
    ax_overlap.fill_between(
        x,
        y - reference.full_nonlocal_overlap_std,
        y + reference.full_nonlocal_overlap_std,
        alpha=0.15,
    )
    ax_overlap.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Mean squared subspace overlap",
        ylim=(0.0, 1.02),
        title="Full-to-non-local overlap",
    )
    figure.suptitle(
        f"Local and full modes across K and rho_F, N={config.N}, "
        f"rank={config.rank}; curves use K={reference_K:g}"
    )
    return figure


def plot_diagnostics(results: dict[float, ScanResult]) -> plt.Figure:
    """Plot activity power, active fractions, and low-rank I/O variance."""
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    ax_lowrank, ax_total, ax_active, ax_input, ax_output, unused = axes.ravel()
    figure.delaxes(unused)

    power_panels = (
        (ax_lowrank, "lowrank_power", "lowrank_power_std"),
        (ax_total, "total_variance", "total_variance_std"),
        (ax_input, "lowrank_input_variance", "lowrank_input_variance_std"),
        (ax_output, "lowrank_output_variance", "lowrank_output_variance_std"),
    )
    for K, result in results.items():
        color = ax_lowrank._get_lines.get_next_color()
        x = result.relative_strengths
        for axis, value_name, std_name in power_panels:
            values = getattr(result, value_name)
            errors = getattr(result, std_name)
            axis.plot(x, values, "o-", color=color, label=f"K={K:g}")
            axis.fill_between(
                x,
                np.maximum(values - errors, np.finfo(float).tiny),
                values + errors,
                color=color,
                alpha=0.15,
            )
        ax_active.plot(
            x,
            result.excitatory_active_fraction,
            "o-",
            color=color,
            label=f"E, K={K:g}",
        )
        ax_active.fill_between(
            x,
            result.excitatory_active_fraction
            - result.excitatory_active_fraction_std,
            result.excitatory_active_fraction
            + result.excitatory_active_fraction_std,
            color=color,
            alpha=0.15,
        )
        ax_active.plot(
            x,
            result.inhibitory_active_fraction,
            "--",
            color=color,
            label=f"I, K={K:g}",
        )
        ax_active.fill_between(
            x,
            result.inhibitory_active_fraction
            - result.inhibitory_active_fraction_std,
            result.inhibitory_active_fraction
            + result.inhibitory_active_fraction_std,
            color=color,
            alpha=0.15,
        )

    titles = (
        "Low-rank activity power",
        "Total activity variance",
        "Active fractions",
        "Low-rank input variance",
        "Low-rank output variance",
    )
    for axis, title in zip(figure.axes, titles):
        axis.set(xlabel="Relative total strength, rho_F", title=title)
        axis.legend(fontsize=8)
    for axis in (ax_lowrank, ax_total, ax_input, ax_output):
        axis.set_yscale("symlog", linthresh=1e-12)
        axis.set_ylabel("Mean squared fluctuation")
    ax_active.set(ylabel="Active fraction", ylim=(-0.02, 1.02))
    figure.suptitle("Mechanism diagnostics across K and rho_F")
    return figure


def plot_ei_cancellation(results: dict[float, ScanResult]) -> plt.Figure:
    """Plot E/I current cancellation and low-rank susceptibility."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_power, ax_ratio, ax_correlation, ax_response = axes.ravel()
    ax_susceptibility = ax_response.twinx()

    for K, result in results.items():
        color = ax_power._get_lines.get_next_color()
        x = result.relative_strengths
        ax_power.plot(
            x, result.excitatory_current_power, "-", color=color,
            label=f"E, K={K:g}",
        )
        ax_power.plot(
            x, result.inhibitory_current_power, "--", color=color,
            label=f"I, K={K:g}",
        )
        ax_power.plot(
            x, result.net_current_power, ":", color=color,
            label=f"Net, K={K:g}",
        )
        for axis, values, errors in (
            (ax_ratio, result.ei_cancellation_ratio, result.ei_cancellation_ratio_std),
            (
                ax_correlation,
                result.ei_current_correlation,
                result.ei_current_correlation_std,
            ),
        ):
            axis.plot(x, values, "o-", color=color, label=f"K={K:g}")
            axis.fill_between(
                x, values - errors, values + errors, color=color, alpha=0.15
            )
        ax_response.plot(
            x,
            result.excitatory_active_fraction,
            "o-",
            color=color,
            label=f"Active E, K={K:g}",
        )
        susceptibility = np.divide(
            result.lowrank_power,
            result.lowrank_output_variance,
            out=np.full_like(result.lowrank_power, np.nan),
            where=result.lowrank_output_variance > np.finfo(float).eps,
        )
        ax_susceptibility.plot(
            x,
            susceptibility,
            ":",
            color=color,
            label=f"Susceptibility, K={K:g}",
        )

    ax_power.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Mean squared current fluctuation",
        title="E, I, and net current power",
        yscale="symlog",
    )
    ax_ratio.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Net power / (E power + I power)",
        title="E/I cancellation ratio",
    )
    ax_correlation.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Correlation",
        ylim=(-1.02, 1.02),
        title="E/I current correlation",
    )
    ax_response.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Excitatory active fraction",
        ylim=(-0.02, 1.02),
        title="Gating and low-rank susceptibility",
    )
    ax_susceptibility.set_ylabel("Low-rank susceptibility")
    ax_susceptibility.set_yscale("log")
    for axis in (ax_power, ax_ratio, ax_correlation):
        axis.legend(fontsize=8, ncol=2)
    response_lines = ax_response.get_lines() + ax_susceptibility.get_lines()
    ax_response.legend(
        response_lines,
        [line.get_label() for line in response_lines],
        fontsize=8,
        ncol=2,
    )
    figure.suptitle("E/I cancellation across K and rho_F")
    return figure


def plot_balance(
    results: dict[float, ScanResult],
    config: ModeScanConfig,
) -> plt.Figure:
    """Plot global, local, and signed-current balance diagnostics."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_mean, ax_active, ax_inactive, ax_currents = axes.ravel()
    balance_panels = (
        (ax_mean, "mean_balance", "Global mean balance"),
        (ax_active, "active_local_balance", "Active-site local balance"),
        (ax_inactive, "inactive_local_balance", "Inactive-site local balance"),
    )
    for K, result in results.items():
        color = ax_mean._get_lines.get_next_color()
        x = result.relative_strengths
        for axis, prefix, _ in balance_panels:
            for population, linestyle in (("e", "-"), ("i", "--")):
                values = getattr(result, f"{prefix}_{population}")
                errors = getattr(result, f"{prefix}_{population}_std")
                axis.plot(
                    x,
                    values,
                    linestyle,
                    color=color,
                    label=f"{population.upper()}, K={K:g}",
                )
                axis.fill_between(
                    x,
                    np.maximum(values - errors, 0.0),
                    np.minimum(values + errors, 1.0),
                    color=color,
                    alpha=0.12,
                )

    for axis, _, title in balance_panels:
        axis.set(
            xlabel="Relative total strength, rho_F",
            ylabel="Residual / gross current",
            ylim=(-0.02, 1.02),
            title=title,
        )
        axis.legend(fontsize=8, ncol=2)

    reference_K = min(results, key=lambda value: abs(np.log(value / config.K)))
    reference = results[reference_K]
    x = reference.relative_strengths
    components = (
        ("external", "External", "black"),
        ("excitatory", "Excitatory", "tab:green"),
        ("inhibitory", "Inhibitory", "tab:blue"),
        ("net", "Net", "tab:red"),
    )
    for component, label, color in components:
        ax_currents.plot(
            x,
            getattr(reference, f"mean_{component}_current_e"),
            "-",
            color=color,
            label=f"{label}, E population",
        )
        ax_currents.plot(
            x,
            getattr(reference, f"mean_{component}_current_i"),
            "--",
            color=color,
            label=f"{label}, I population",
        )
    ax_currents.axhline(0.0, color="gray", linewidth=0.8)
    ax_currents.set(
        xlabel="Relative total strength, rho_F",
        ylabel="Signed mean current",
        yscale="symlog",
        title=f"Signed mean currents at K={reference_K:g}",
    )
    ax_currents.legend(fontsize=8, ncol=2)
    figure.suptitle("Mean and local E/I balance across K and rho_F")
    return figure


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline/K_rhoF_modes.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--N", type=int)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_config(
    args: argparse.Namespace,
) -> tuple[ModeScanConfig, dict, list[float], float]:
    """Load the base scan and K-scan settings."""
    config, document = load_dataclass_sections(
        args.config,
        ModeScanConfig,
        tuple(MODE_SCAN_SECTIONS),
    )
    if args.seed is not None:
        config.seed = args.seed
    if args.N is not None:
        config.N = args.N
        config.__post_init__()

    run = document.get("run", {})
    k_scan = document.get("k_scan", {})
    if not isinstance(run, dict) or not isinstance(k_scan, dict):
        raise ValueError("The run and k_scan sections must be mappings.")
    if args.output_root is not None:
        run["output_root"] = str(args.output_root)
    if args.run_id is not None:
        run["run_id"] = args.run_id
    K_values = [float(value) for value in k_scan.get("K_values", [])]
    if not K_values or any(value <= 0 for value in K_values):
        raise ValueError("k_scan.K_values must contain positive values.")
    transition_threshold = float(k_scan.get("transition_threshold", 0.02))
    return config, run, K_values, transition_threshold


def save_run_files(
    run_directory: Path,
    config: ModeScanConfig,
    results: dict[float, ScanResult],
    K_values: list[float],
    transition_threshold: float,
) -> None:
    """Save one complete K-scan result."""
    resolved = {"run": {"run_directory": str(run_directory)}}
    resolved.update(dataclass_to_sections(config, MODE_SCAN_SECTIONS))
    resolved["k_scan"] = {
        "K_values": K_values,
        "transition_threshold": transition_threshold,
    }
    arrays = {}
    rows = []
    for K, result in results.items():
        prefix = f"K_{K:g}_"
        arrays.update(
            {prefix + name: values for name, values in scan_result_arrays(result).items()}
        )
        rows.extend({"K": K, **row} for row in scan_metric_rows(result))
    save_yaml(run_directory / "config.yaml", resolved)
    save_npz(run_directory / "results.npz", arrays)
    save_csv(run_directory / "metrics.csv", METRIC_FIELDS, rows)
    save_json(run_directory / "metadata.json", runtime_metadata(repo_root))


def main() -> None:
    """Run the scan and save all output files."""
    args = parse_args()
    config, run, K_values, transition_threshold = load_config(args)
    output_root = Path(run.get("output_root", "output/scans"))
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    experiment = str(run.get("experiment", "K_rhoF_modes"))
    run_directory = create_run_directory(output_root, experiment, run.get("run_id"))
    results = scan_all_K(config, K_values)
    figure = plot_results(results, config, transition_threshold)
    diagnostic_figure = plot_diagnostics(results)
    cancellation_figure = plot_ei_cancellation(results)
    balance_figure = plot_balance(results, config)
    save_run_files(run_directory, config, results, K_values, transition_threshold)
    figure.savefig(run_directory / "summary.png", dpi=180)
    diagnostic_figure.savefig(run_directory / "mechanism_diagnostics.png", dpi=180)
    cancellation_figure.savefig(run_directory / "ei_cancellation.png", dpi=180)
    balance_figure.savefig(run_directory / "balance_diagnostics.png", dpi=180)
    print(f"Saved run to {run_directory}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)
        plt.close(diagnostic_figure)
        plt.close(cancellation_figure)
        plt.close(balance_figure)


if __name__ == "__main__":
    main()
