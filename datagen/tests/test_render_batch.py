"""A local batch render runs every task and exits non-zero when any of them fails."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_datagen import cli
from strata_datagen.render import husk

PASSED: dict[str, object] = {"exit_code": 0, "post_passed": True, "device": "cpu", "mean_frame_seconds": 1.0}
OUTCOMES: dict[str, dict[str, object]] = {
    "ok": PASSED,
    "crashed": {**PASSED, "exit_code": 3, "post_passed": False},
    "rejected": {**PASSED, "post_passed": False},
}


class RenderBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def render(self, shots: list[str]) -> tuple[int, list[str], str]:
        """Run `render` over one task per shot, each task's outcome named by its shot."""
        jobs = [job for shot in shots for job in husk.jobs(self.root / "shots" / shot, self.root, (1,), 60)]
        husk.write_job_list(self.root, jobs)
        ran: list[str] = []

        def task(shot: Path, frames: range, husk_path: str) -> dict[str, object]:
            del frames, husk_path
            ran.append(shot.name)
            return OUTCOMES[shot.name.split("_")[0]]

        output = io.StringIO()
        with mock.patch.object(cli, "render_task", task), contextlib.redirect_stdout(output):
            try:
                cli.main(["render", "--root", str(self.root)])
            except SystemExit as stop:
                return int(stop.code or 0), ran, output.getvalue()
        return 0, ran, output.getvalue()

    def test_every_task_runs_and_any_failure_exits_one(self) -> None:
        for failing in ("crashed", "rejected"):
            with self.subTest(failing):
                code, ran, output = self.render(["ok_a", failing, "ok_b"])
                self.assertEqual(code, 1)
                self.assertEqual(ran, ["ok_a", failing, "ok_b"])
                self.assertIn(f"failed {failing}:1-1", output)

    def test_a_clean_batch_exits_zero(self) -> None:
        code, ran, output = self.render(["ok_a", "ok_b"])
        self.assertEqual((code, ran), (0, ["ok_a", "ok_b"]))
        self.assertNotIn("failed", output)


if __name__ == "__main__":
    unittest.main()
