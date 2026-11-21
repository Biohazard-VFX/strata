"""The shared-GPU reserve must prevent job launch, not merely warn afterward."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from strata_train.benchmark import measured


class AdmissionTests(unittest.TestCase):
    def test_insufficient_headroom_does_not_launch(self) -> None:
        with (
            patch("strata_train.benchmark.gpu_memory", return_value=(10000, 9047, {})),
            patch("strata_train.benchmark.subprocess.Popen") as launch,
        ):
            result = measured(["must-not-run"], Path("unused"), 10, 7000, 2048)
        launch.assert_not_called()
        self.assertEqual(result["status"], "not_measurable")
        self.assertEqual(result["device_free_mib"], 9047)
        self.assertEqual(result["expected_peak_mib"], 7000)
        self.assertEqual(result["reserve_mib"], 2048)


if __name__ == "__main__":
    unittest.main()
