"""Create the three task panels for Figure 2."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.analysis.working_memory import output_readout_window, run_working_memory_K_scan
from src.config import load_dataclass_sections, load_yaml, save_yaml
from src.driven import DRIVEN_DOT_SECTIONS, DrivenDotConfig
from src.io import runtime_metadata, save_csv, save_json, save_npz
from src.learning.rigid_reconstruction import train_rigid_reconstruction
from src.models.rigid_reconstruction import RigidReconstructionReservoir
from src.stimuli import rigid_shift_movie
from src.tasks.driven_dot import run_tracking_scan
from src.tasks.rigid_reconstruction import (
    RIGID_RECONSTRUCTION_SECTIONS,
    RigidReconstructionConfig,
)
from src.tasks.working_memory import WORKING_MEMORY_SECTIONS, WorkingMemoryConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/figures/figure2_tasks.yaml"),
    )
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def save_panel(
    figure: plt.Figure,
    output_directory: Path,
    stem: str,
) -> tuple[Path, Path]:
    """Save one panel as PDF and PNG files."""
    output_directory.mkdir(parents=True, exist_ok=True)
    pdf_path = output_directory / f"{stem}.pdf"
    png_path = output_directory / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    return pdf_path, png_path


def select_best_worst(
    performance: np.ndarray,
    higher_is_better: bool = True,
) -> tuple[int, int]:
    """Return indices for the largest and smallest mean performance."""
    values = np.asarray(performance, dtype=float)
    if values.ndim == 1:
        means = values
    elif values.ndim == 2:
        means = np.nanmean(values, axis=1)
    else:
        raise ValueError("Performance must have one or two dimensions.")
    if means.size == 0 or not np.isfinite(means).any():
        raise ValueError("Performance must contain a finite value.")
    if higher_is_better:
        return int(np.nanargmax(means)), int(np.nanargmin(means))
    return int(np.nanargmin(means)), int(np.nanargmax(means))


def circular_trace(values: np.ndarray) -> np.ndarray:
    """Wrap angles and break false lines across the branch cut."""
    wrapped = np.angle(np.exp(1j * np.asarray(values, dtype=float)))
    jumps = np.abs(np.diff(wrapped)) > np.pi
    wrapped[1:][jumps] = np.nan
    return wrapped


def _plot_scan(
    axis: plt.Axes,
    K_values: np.ndarray,
    performance: np.ndarray,
    ylabel: str,
    best: int,
    worst: int,
) -> None:
    """Plot one performance scan and mark its selected examples."""
    values = np.asarray(performance, dtype=float)
    if values.ndim == 1:
        mean = values
        spread = np.zeros_like(values)
    else:
        mean = np.nanmean(values, axis=1)
        spread = np.nanstd(values, axis=1)
    axis.errorbar(K_values, mean, yerr=spread, color="0.25", marker="o", capsize=3)
    axis.scatter(K_values[best], mean[best], color="tab:blue", s=55, zorder=4, label="Best")
    axis.scatter(K_values[worst], mean[worst], color="tab:red", s=55, zorder=4, label="Worst")
    axis.set(xscale="log", xlabel="K", ylabel=ylabel)
    axis.legend(frameon=False, ncol=2)


def _zoom_r2_scan(axis: plt.Axes, performance: np.ndarray) -> None:
    """Zoom the decoding axis and label values below its visible range."""
    lower, upper = -1.0, 1.0
    mean = np.nanmean(np.asarray(performance, dtype=float), axis=1)
    for index in np.flatnonzero(mean < lower):
        axis.scatter(
            axis.lines[0].get_xdata()[index],
            lower + 0.04,
            marker="v",
            color="0.25",
            clip_on=False,
            zorder=5,
        )
        axis.annotate(
            f"{mean[index]:.1f}",
            (axis.lines[0].get_xdata()[index], lower + 0.08),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axis.set_ylim(lower, upper)


def _example_axes(figure: plt.Figure) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    """Return one scan axis and two comparison axes."""
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15), hspace=0.42)
    return (
        figure.add_subplot(grid[0, :]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
    )


def _title(axis: plt.Axes, role: str, K: float) -> None:
    """Set one direct example label."""
    axis.set_title(f"{role}: K={K:g}")


def plot_panels(arrays: dict[str, np.ndarray]) -> dict[str, plt.Figure]:
    """Return separate panels for decoding, prediction, and memory."""
    K_values = arrays["K_values"]
    figures: dict[str, plt.Figure] = {}

    best, worst = select_best_worst(arrays["rigid_r2"])
    figure = plt.figure(figsize=(7.2, 6.0))
    scan_axis, best_axis, worst_axis = _example_axes(figure)
    _plot_scan(scan_axis, K_values, arrays["rigid_r2"], r"Decoding $R^2$", best, worst)
    _zoom_r2_scan(scan_axis, arrays["rigid_r2"])
    for axis, index, role, color in (
        (best_axis, best, "Best", "tab:blue"),
        (worst_axis, worst, "Worst", "tab:red"),
    ):
        axis.plot(
            arrays["rigid_time"],
            circular_trace(arrays["rigid_target"][index]),
            color="black",
            label="True",
        )
        axis.plot(
            arrays["rigid_time"],
            circular_trace(arrays["rigid_prediction"][index]),
            color=color,
            linestyle="--",
            label="Decoded",
        )
        axis.set(xlabel="Time (s)", ylabel="Angle (rad)")
        _title(axis, role, K_values[index])
        axis.legend(frameon=False, fontsize=8)
    figures["panel_A_decoding"] = figure

    best, worst = select_best_worst(
        arrays["dot_peak_delay_seconds"], higher_is_better=False,
    )
    figure = plt.figure(figsize=(7.2, 6.0))
    scan_axis, best_axis, worst_axis = _example_axes(figure)
    delay_ms = 1000.0 * arrays["dot_peak_delay_seconds"]
    delay_mean = np.nanmean(delay_ms, axis=1)
    delay_std = np.nanstd(delay_ms, axis=1)
    scan_axis.errorbar(
        K_values,
        delay_mean,
        yerr=delay_std,
        color="0.25",
        marker="o",
        capsize=3,
    )
    scan_axis.scatter(K_values[best], delay_mean[best], color="tab:blue", s=55, zorder=4, label="Best")
    scan_axis.scatter(K_values[worst], delay_mean[worst], color="tab:red", s=55, zorder=4, label="Worst")
    scan_axis.set(xscale="log", xlabel="K", ylabel="Peak delay (ms)")
    scan_axis.legend(frameon=False, ncol=2, loc="upper left")
    amplitude_axis = scan_axis.twinx()
    amplitude_mean = np.nanmean(arrays["dot_peak_height"], axis=1)
    amplitude_std = np.nanstd(arrays["dot_peak_height"], axis=1)
    amplitude_axis.errorbar(
        K_values,
        amplitude_mean,
        yerr=amplitude_std,
        color="tab:orange",
        marker="o",
        linestyle="--",
        alpha=0.35,
        capsize=3,
    )
    amplitude_axis.set_ylabel("Peak amplitude", color="tab:orange", alpha=0.6)
    amplitude_axis.tick_params(axis="y", colors="tab:orange")
    for axis, index, role, color in (
        (best_axis, best, "Best", "tab:blue"),
        (worst_axis, worst, "Worst", "tab:red"),
    ):
        curve = arrays["dot_cross_correlation"][index]
        max_lag = float(np.asarray(arrays["dot_max_lag_seconds"]).item())
        lag_window = np.flatnonzero(
            np.abs(arrays["dot_lag_seconds"]) <= max_lag + np.finfo(float).eps
        )
        peak = int(lag_window[np.nanargmax(curve[lag_window])])
        axis.plot(
            arrays["dot_lag_seconds"][lag_window],
            curve[lag_window],
            color=color,
        )
        axis.scatter(arrays["dot_lag_seconds"][peak], curve[peak], color=color, s=35)
        axis.axvline(0.0, color="0.6", linewidth=0.8)
        axis.set(xlabel="Lag (s)", ylabel="Cross-correlation", xlim=(-max_lag, max_lag))
        _title(axis, role, K_values[index])
    figures["panel_B_prediction"] = figure

    best, worst = select_best_worst(arrays["memory_r2"])
    figure = plt.figure(figsize=(7.2, 6.0))
    scan_axis, best_axis, worst_axis = _example_axes(figure)
    _plot_scan(
        scan_axis,
        K_values,
        arrays["memory_r2"],
        r"Cue readout $R^2$",
        best,
        worst,
    )
    _zoom_r2_scan(scan_axis, arrays["memory_r2"])
    cue_end = int(np.asarray(arrays["memory_cue_steps"]).item())
    go_start = cue_end + int(np.asarray(arrays["memory_delay_steps"]).item())
    go_end = go_start + cue_end
    output_window_steps = int(
        np.asarray(arrays.get("memory_output_window_steps", -1)).item()
    )
    output_start_step = int(
        np.asarray(arrays.get("memory_output_start_step", go_start)).item()
    )
    for axis, index, role, color in (
        (best_axis, best, "Best", "tab:blue"),
        (worst_axis, worst, "Worst", "tab:red"),
    ):
        time = arrays["memory_time"]
        axis.axvspan(time[0], time[max(0, cue_end - 1)], color="tab:green", alpha=0.12, label="Stimulus")
        axis.axvspan(time[cue_end], time[max(cue_end, go_start - 1)], color="0.5", alpha=0.10, label="Delay")
        axis.axvspan(
            time[go_start],
            time[min(go_end - 1, len(time) - 1)],
            color="tab:purple",
            alpha=0.12,
            label="Common cue",
        )
        if output_window_steps > 0:
            output_stop = min(output_start_step + output_window_steps, len(time))
            axis.axvspan(
                time[output_start_step],
                time[max(output_start_step, output_stop - 1)],
                color="tab:blue",
                alpha=0.08,
                label="Scored response",
            )
        axis.plot(time, arrays["memory_target"][index], color="black", label="Target")
        readout = np.asarray(arrays["memory_output"][index], dtype=float)
        axis.plot(time, readout, color=color, linestyle="--", label="Readout")
        axis.axvline(time[-1], color="0.3", linestyle=":", linewidth=1.0)
        axis.set(xlabel="Time (s)", ylabel="Readout")
        _title(axis, role, K_values[index])
        axis.legend(frameon=False, fontsize=7, ncol=2)
    figures["panel_C_memory"] = figure
    return figures


def plot_assembled_figure(arrays: dict[str, np.ndarray]) -> plt.Figure:
    """Return Figure 2 with one task per row."""
    K_values = arrays["K_values"]
    figure = plt.figure(figsize=(14.0, 10.4))
    grid = figure.add_gridspec(
        3,
        3,
        width_ratios=(1.05, 1.0, 1.0),
        hspace=0.44,
        wspace=0.43,
    )
    axes = [
        tuple(figure.add_subplot(grid[row, column]) for column in range(3))
        for row in range(3)
    ]

    scan_axis, best_axis, worst_axis = axes[0]
    best, worst = select_best_worst(arrays["rigid_r2"])
    _plot_scan(scan_axis, K_values, arrays["rigid_r2"], r"Decoding $R^2$", best, worst)
    scan_axis.set_title("Decoding", loc="left", fontsize=13, fontweight="bold", pad=10)
    scan_axis.legend(frameon=False, ncol=2, fontsize=10)
    _zoom_r2_scan(scan_axis, arrays["rigid_r2"])
    for axis, index, role, color in (
        (best_axis, best, "Best", "tab:blue"),
        (worst_axis, worst, "Worst", "tab:red"),
    ):
        axis.plot(
            arrays["rigid_time"],
            circular_trace(arrays["rigid_target"][index]),
            color="black",
            label="True",
        )
        axis.plot(
            arrays["rigid_time"],
            circular_trace(arrays["rigid_prediction"][index]),
            color=color,
            linestyle="--",
            label="Decoded",
        )
        axis.set(xlabel="Time (s)", ylabel="Angle (rad)")
        _title(axis, role, K_values[index])
        axis.legend(frameon=False, fontsize=10)

    scan_axis, best_axis, worst_axis = axes[1]
    best, worst = select_best_worst(
        arrays["dot_peak_delay_seconds"], higher_is_better=False,
    )
    delay_ms = 1000.0 * arrays["dot_peak_delay_seconds"]
    delay_mean = np.nanmean(delay_ms, axis=1)
    delay_std = np.nanstd(delay_ms, axis=1)
    scan_axis.errorbar(
        K_values,
        delay_mean,
        yerr=delay_std,
        color="0.25",
        marker="o",
        capsize=3,
    )
    scan_axis.scatter(
        K_values[best], delay_mean[best], color="tab:blue", s=55, zorder=4, label="Best",
    )
    scan_axis.scatter(
        K_values[worst], delay_mean[worst], color="tab:red", s=55, zorder=4, label="Worst",
    )
    scan_axis.set(xscale="log", xlabel="K", ylabel="Peak delay (ms)")
    scan_axis.set_title("Prediction", loc="left", fontsize=13, fontweight="bold", pad=10)
    scan_axis.legend(frameon=False, ncol=2, loc="upper left", fontsize=10)
    amplitude_axis = scan_axis.twinx()
    amplitude_axis.errorbar(
        K_values,
        np.nanmean(arrays["dot_peak_height"], axis=1),
        yerr=np.nanstd(arrays["dot_peak_height"], axis=1),
        color="tab:orange",
        marker="o",
        linestyle="--",
        alpha=0.35,
        capsize=3,
    )
    amplitude_axis.text(
        0.98,
        0.96,
        "Peak amplitude",
        transform=amplitude_axis.transAxes,
        color="tab:orange",
        alpha=0.7,
        fontsize=10,
        ha="right",
        va="top",
    )
    amplitude_axis.tick_params(axis="y", colors="tab:orange")
    max_lag = float(np.asarray(arrays["dot_max_lag_seconds"]).item())
    lag_window = np.flatnonzero(
        np.abs(arrays["dot_lag_seconds"]) <= max_lag + np.finfo(float).eps
    )
    for axis, index, role, color in (
        (best_axis, best, "Best", "tab:blue"),
        (worst_axis, worst, "Worst", "tab:red"),
    ):
        curve = arrays["dot_cross_correlation"][index]
        peak = int(lag_window[np.nanargmax(curve[lag_window])])
        axis.plot(arrays["dot_lag_seconds"][lag_window], curve[lag_window], color=color)
        axis.scatter(arrays["dot_lag_seconds"][peak], curve[peak], color=color, s=35)
        axis.axvline(0.0, color="0.6", linewidth=0.8)
        axis.set(xlabel="Lag (s)", ylabel="Cross-correlation", xlim=(-max_lag, max_lag))
        _title(axis, role, K_values[index])

    scan_axis, best_axis, worst_axis = axes[2]
    best, worst = select_best_worst(arrays["memory_r2"])
    _plot_scan(scan_axis, K_values, arrays["memory_r2"], r"Cue readout $R^2$", best, worst)
    scan_axis.set_title("Working memory", loc="left", fontsize=13, fontweight="bold", pad=10)
    scan_axis.legend(frameon=False, ncol=2, fontsize=10)
    _zoom_r2_scan(scan_axis, arrays["memory_r2"])
    cue_end = int(np.asarray(arrays["memory_cue_steps"]).item())
    go_start = cue_end + int(np.asarray(arrays["memory_delay_steps"]).item())
    go_end = go_start + cue_end
    output_window_steps = int(
        np.asarray(arrays.get("memory_output_window_steps", -1)).item()
    )
    output_start_step = int(
        np.asarray(arrays.get("memory_output_start_step", go_start)).item()
    )
    time = arrays["memory_time"]
    for axis, index, role, color in (
        (best_axis, best, "Best", "tab:blue"),
        (worst_axis, worst, "Worst", "tab:red"),
    ):
        axis.axvspan(
            time[0], time[max(0, cue_end - 1)], color="tab:green", alpha=0.12, label="Stimulus",
        )
        axis.axvspan(
            time[cue_end], time[max(cue_end, go_start - 1)], color="0.5", alpha=0.10, label="Delay",
        )
        axis.axvspan(
            time[go_start],
            time[min(go_end - 1, len(time) - 1)],
            color="tab:purple",
            alpha=0.12,
            label="Common cue",
        )
        if output_window_steps > 0:
            output_stop = min(output_start_step + output_window_steps, len(time))
            axis.axvspan(
                time[output_start_step],
                time[max(output_start_step, output_stop - 1)],
                color="tab:blue",
                alpha=0.08,
                label="Scored response",
            )
        axis.plot(time, arrays["memory_target"][index], color="black", label="Target")
        axis.plot(
            time,
            np.asarray(arrays["memory_output"][index], dtype=float),
            color=color,
            linestyle="--",
            label="Readout",
        )
        axis.axvline(time[-1], color="0.3", linestyle=":", linewidth=1.0)
        axis.set(xlabel="Time (s)", ylabel="Readout")
        _title(axis, role, K_values[index])
        axis.legend(frameon=False, fontsize=10, ncol=2)

    figure.subplots_adjust(left=0.065, right=0.94, top=0.95, bottom=0.07)
    for label, row_axis in zip(("A", "B", "C"), (axes[0][0], axes[1][0], axes[2][0])):
        position = row_axis.get_position()
        figure.text(
            0.012,
            position.y1 + 0.014,
            label,
            fontsize=20,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    return figure


def _required_mapping(document: dict, name: str) -> dict:
    """Return one required configuration mapping."""
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section must be a mapping: {name}")
    return value


def _source_path(value: object) -> Path:
    """Return one repository-relative source path."""
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def run_analysis(document: dict) -> dict[str, np.ndarray]:
    """Run the three matched task scans."""
    common = _required_mapping(document, "common")
    scan = _required_mapping(document, "scan")
    sources = _required_mapping(document, "sources")
    K_values = [float(value) for value in scan.get("K_values", [])]
    seeds = [int(value) for value in scan.get("seeds", [])]
    if not K_values or any(value <= 0 for value in K_values):
        raise ValueError("K_values must contain positive values.")
    if not seeds:
        raise ValueError("seeds must not be empty.")
    u_e = float(common.get("u_e", 10.0))
    u_i = float(common.get("u_i", 0.0))
    tau_e = float(common.get("tau_e", 0.01))
    tau_i = float(common.get("tau_i", 0.02))
    init_duration = float(common.get("init_duration_seconds", 0.5))

    rigid_base, _ = load_dataclass_sections(
        _source_path(sources["rigid"]),
        RigidReconstructionConfig,
        tuple(RIGID_RECONSTRUCTION_SECTIONS),
    )
    rigid_scores = np.empty((len(K_values), len(seeds)))
    rigid_targets = []
    rigid_predictions = []
    for K_index, K in enumerate(K_values):
        for seed_index, seed in enumerate(seeds):
            config = replace(
                rigid_base,
                K=K,
                seed=seed,
                u_e=u_e,
                u_i=u_i,
                tau_e=tau_e,
                tau_i=tau_i,
                init_steps=round(init_duration / rigid_base.dt),
            )
            stimulus, target = rigid_shift_movie(
                config.N,
                config.steps,
                config.dt,
                config.smoothing_width,
                config.shift_distance,
                config.seed,
                config.device,
            )
            result = train_rigid_reconstruction(
                RigidReconstructionReservoir(config), stimulus, target,
            )
            rigid_scores[K_index, seed_index] = result.evaluation_r2
            if seed_index == 0:
                rigid_targets.append(result.target)
                rigid_predictions.append(result.prediction)

    dot_base, dot_document = load_dataclass_sections(
        _source_path(sources["dot"]),
        DrivenDotConfig,
        tuple(DRIVEN_DOT_SECTIONS),
    )
    dot_task = _required_mapping(dot_document, "task")
    dot_config = replace(
        dot_base,
        u_e=u_e,
        u_i=u_i,
        tau_e=tau_e,
        tau_i=tau_i,
        init_steps=round(init_duration / dot_base.dt),
    )
    dot_result = run_tracking_scan(
        dot_config,
        K_values,
        int(dot_task.get("repetitions", 1)),
        int(dot_task.get("max_lag", 20)),
    )

    memory_base, _ = load_dataclass_sections(
        _source_path(sources["memory"]),
        WorkingMemoryConfig,
        tuple(WORKING_MEMORY_SECTIONS),
    )
    memory_scores = np.empty((len(K_values), len(seeds)))
    memory_training_scores = np.empty((len(K_values), len(seeds)))
    memory_outputs = []
    memory_targets = []
    memory_output_start_step, _ = output_readout_window(memory_base)
    for seed_index, seed in enumerate(seeds):
        config = replace(
            memory_base,
            seed=seed,
            u_e=u_e,
            u_i=u_i,
            tau_e=tau_e,
            tau_i=tau_i,
            init_steps=round(init_duration / (memory_base.dt * memory_base.microsteps)),
        )
        result = run_working_memory_K_scan(config, K_values)
        memory_scores[:, seed_index] = result.output_r2
        memory_training_scores[:, seed_index] = result.training_output_r2
        if seed_index == 0:
            memory_outputs = list(result.example_output)
            memory_targets = list(result.example_output_target)

    return {
        "K_values": np.asarray(K_values),
        "seeds": np.asarray(seeds),
        "rigid_r2": rigid_scores,
        "rigid_time": np.arange(rigid_base.steps) * rigid_base.dt,
        "rigid_target": np.asarray(rigid_targets),
        "rigid_prediction": np.asarray(rigid_predictions),
        "dot_peak_height": dot_result.peak_heights,
        "dot_peak_lag_steps": dot_result.peak_lags,
        "dot_peak_delay_seconds": -dot_result.peak_lags * dot_base.dt * dot_base.steps_per_record,
        "dot_lag_seconds": dot_result.lags * dot_base.dt * dot_base.steps_per_record,
        "dot_max_lag_seconds": np.asarray(
            int(dot_task.get("max_lag", 20)) * dot_base.dt * dot_base.steps_per_record
        ),
        "dot_cross_correlation": dot_result.cross_correlation,
        "memory_r2": memory_scores,
        "memory_training_r2": memory_training_scores,
        "memory_time": np.arange(memory_base.steps) * memory_base.dt * memory_base.microsteps,
        "memory_output": np.asarray(memory_outputs),
        "memory_target": np.asarray(memory_targets),
        "memory_cue_steps": np.asarray(memory_base.cue_steps),
        "memory_delay_steps": np.asarray(memory_base.delay_steps),
        "memory_output_window_steps": np.asarray(
            memory_base.output_window_steps
            if memory_base.output_window_steps is not None else -1
        ),
        "memory_output_start_step": np.asarray(memory_output_start_step),
    }


def metric_rows(arrays: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """Return one summary row for each K value."""
    rows = []
    for index, K in enumerate(arrays["K_values"]):
        rows.append(
            {
                "K": float(K),
                "decoding_r2_mean": float(np.nanmean(arrays["rigid_r2"][index])),
                "decoding_r2_std": float(np.nanstd(arrays["rigid_r2"][index])),
                "prediction_delay_ms_mean": float(
                    1000.0 * np.nanmean(arrays["dot_peak_delay_seconds"][index])
                ),
                "prediction_delay_ms_std": float(
                    1000.0 * np.nanstd(arrays["dot_peak_delay_seconds"][index])
                ),
                "prediction_peak_amplitude_mean": float(
                    np.nanmean(arrays["dot_peak_height"][index])
                ),
                "prediction_peak_amplitude_std": float(
                    np.nanstd(arrays["dot_peak_height"][index])
                ),
                "memory_r2_mean": float(np.nanmean(arrays["memory_r2"][index])),
                "memory_r2_std": float(np.nanstd(arrays["memory_r2"][index])),
                "memory_training_r2_mean": float(
                    np.nanmean(arrays["memory_training_r2"][index])
                ),
                "memory_training_r2_std": float(
                    np.nanstd(arrays["memory_training_r2"][index])
                ),
            }
        )
    return rows


def main() -> None:
    """Run the scans and save all Figure 2 panel files."""
    args = parse_args()
    document = load_yaml(args.config)
    run = _required_mapping(document, "run")
    output_directory = args.output_directory or Path(
        run.get("output_directory", "output/figures/figure2")
    )
    if not output_directory.is_absolute():
        output_directory = repo_root / output_directory
    data_directory = output_directory / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    arrays = run_analysis(document)
    save_npz(data_directory / "results.npz", arrays)
    resolved = dict(document)
    resolved["run"] = {"output_directory": str(output_directory)}
    save_yaml(data_directory / "config.yaml", resolved)
    save_json(data_directory / "metadata.json", runtime_metadata(repo_root))
    rows = metric_rows(arrays)
    save_csv(data_directory / "metrics.csv", list(rows[0]), rows)
    figures = plot_panels(arrays)
    for stem, figure in figures.items():
        save_panel(figure, output_directory, stem)
    print(f"Saved Figure 2 panels to {output_directory}")
    if args.show:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)


if __name__ == "__main__":
    main()
