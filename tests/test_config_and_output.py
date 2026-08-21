"""Tests for YAML configuration and reproducible run output."""

from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from scripts.scan_local_network_modes import save_run_files
from src.analysis.local_network_modes import (
    MODE_SCAN_SECTIONS,
    ModeScanConfig,
    plot_result,
    run_scan,
)
from src.config import load_dataclass_sections
from src.io import create_run_directory


repo_root = Path(__file__).resolve().parents[1]


class ConfigurationAndOutputTests(unittest.TestCase):
    """Check configuration loading and the standard output set."""

    def test_default_yaml_loads(self) -> None:
        config, document = load_dataclass_sections(
            repo_root / "configs/scans/local_network_modes.yaml",
            ModeScanConfig,
            tuple(MODE_SCAN_SECTIONS),
        )

        self.assertEqual(config.N, 15)
        self.assertEqual(config.rank, 4)
        self.assertEqual(document["run"]["experiment"], "local_network_modes")

    def test_unknown_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("network:\n  unknown_value: 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown_value"):
                load_dataclass_sections(
                    path,
                    ModeScanConfig,
                    tuple(MODE_SCAN_SECTIONS),
                )

    def test_small_run_writes_complete_output(self) -> None:
        config = ModeScanConfig(
            N=5,
            K=10.0,
            rho_f_values=[0.0, 0.5],
            rank=2,
            seed=3,
            n_seeds=1,
            init_steps=5,
            record_steps=10,
            sample_every=2,
        )
        result = run_scan(config)

        with tempfile.TemporaryDirectory() as directory:
            run_directory = create_run_directory(
                Path(directory),
                "local_network_modes",
                "test-run",
            )
            save_run_files(run_directory, config, result)
            figure = plot_result(result, config)
            figure.savefig(run_directory / "summary.png", dpi=50)
            plt.close(figure)

            expected = {
                "config.yaml",
                "metadata.json",
                "metrics.csv",
                "results.npz",
                "summary.png",
            }
            self.assertEqual({path.name for path in run_directory.iterdir()}, expected)
            with np.load(run_directory / "results.npz") as arrays:
                self.assertEqual(arrays["example_rates"].shape[:2], (2, 25))


if __name__ == "__main__":
    unittest.main()
