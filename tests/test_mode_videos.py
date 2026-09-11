"""Tests for local-to-network activity videos."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.baseline.render_K_rhoF_videos import (
    activity_frames,
    mean_reconstruction_score,
    render_condition_video,
    select_condition_indices,
    select_frame_indices,
    shared_raw_limit,
)


repo_root = Path(__file__).resolve().parents[1]


class ModeVideoTests(unittest.TestCase):
    """Check condition selection and video output."""

    def test_video_script_help_succeeds(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts/baseline/render_K_rhoF_videos.py"),
                "--help",
            ],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_conditions_select_zero_peak_and_largest_stable_rho(self) -> None:
        indices = select_condition_indices(
            rho_f=np.array([0.0, 0.2, 0.5, 1.0, 8.0]),
            advantage=np.array([0.0, 0.1, 0.4, 0.3, 0.05]),
            stable=np.array([True, True, True, False, True]),
        )

        self.assertEqual(indices, (0, 2, 4))

    def test_frame_indices_include_recording_endpoints(self) -> None:
        indices = select_frame_indices(sample_count=10, frame_count=4)

        np.testing.assert_array_equal(indices, np.array([0, 3, 6, 9]))

    def test_activity_frames_center_and_scale_each_condition(self) -> None:
        activity = np.array(
            [
                [1.0, 3.0],
                [2.0, 4.0],
                [3.0, 5.0],
                [4.0, 6.0],
            ]
        )

        raw, normalized = activity_frames(activity)

        self.assertEqual(raw.shape, (2, 2, 2))
        self.assertEqual(normalized.shape, (2, 2, 2))
        self.assertAlmostEqual(float(normalized.mean()), 0.0)
        self.assertAlmostEqual(float(np.sqrt(np.mean(normalized**2))), 1.0)

    def test_shared_raw_limit_resists_one_pooled_outlier(self) -> None:
        limit = shared_raw_limit(
            [
                np.ones((10, 10)),
                np.concatenate([np.ones(100), np.array([1000.0])]),
            ]
        )

        self.assertEqual(limit, 1.0)

    def test_reconstruction_score_uses_complete_shells_within_limit(self) -> None:
        score = mean_reconstruction_score(
            curve=np.array([0.1, 0.2, 0.4, 0.8]),
            shell_counts=np.array([1, 2, 4]),
            mode_limit=2,
        )

        self.assertAlmostEqual(score, 0.15)

    def test_renderer_writes_each_selected_frame(self) -> None:
        activity = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition.gif"

            render_condition_video(
                path=path,
                activity=activity,
                frame_indices=np.array([0, 1, 2]),
                raw_limit=6.0,
                K=10.0,
                rho_f=0.5,
                sample_interval=0.1,
                fps=2,
                metrics={
                    "local": 0.2,
                    "full": 0.5,
                    "advantage": 0.3,
                    "nonlocal": 0.4,
                },
            )

            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 3)

    def test_script_renders_zero_peak_and_high_conditions(self) -> None:
        activities = np.array(
            [
                [[1.0, 2.0, 3.0]] * 4,
                [[2.0, 3.0, 4.0]] * 4,
                [[3.0, 4.0, 5.0]] * 4,
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            (run_directory / "config.yaml").write_text(
                "simulation:\n  dt: 0.1\n  sample_every: 1\n"
                "analysis:\n  auc_modes: 2\n",
                encoding="utf-8",
            )
            np.savez_compressed(
                run_directory / "results.npz",
                K_10_relative_strengths=np.array([0.0, 0.5, 8.0]),
                K_10_transition_index=np.array([0.0, 0.4, 0.1]),
                K_10_stable=np.array([True, True, True]),
                K_10_example_rates=activities,
                K_10_local_curves=np.array([[0.1, 0.2, 0.4, 0.8]] * 3),
                K_10_network_curves=np.array([[0.2, 0.4, 0.6, 0.9]] * 3),
                K_10_geometric_shell_counts=np.array([1, 2, 4]),
                K_10_nonlocal_fraction=np.array([0.1, 0.5, 0.2]),
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "scripts/baseline/render_K_rhoF_videos.py"),
                    "--run-directory",
                    str(run_directory),
                    "--K",
                    "10",
                    "--frame-count",
                    "3",
                    "--fps",
                    "2",
                ],
                cwd=repo_root,
                capture_output=True,
                check=False,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {path.name for path in run_directory.glob("rhoF_*.gif")},
                {"rhoF_0.gif", "rhoF_0.5.gif", "rhoF_8.gif"},
            )


if __name__ == "__main__":
    unittest.main()
