import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from strata_datagen.render import husk
from tools import check_decisions, check_generators, check_identifiers

ASSET_WORD = "ja" + "guar"
PROJECT_WORD = "EN" + "BD"


class ToolTest(unittest.TestCase):
    def test_identifiers_are_banned_outside_presets_and_projects_everywhere(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "presets").mkdir()
            (root / "presets" / "a.py").write_text(f"HERO = '{ASSET_WORD}'\n")
            self.assertEqual(check_identifiers.violations([root]), [])
            (root / "gen.py").write_text(f"HERO = '{ASSET_WORD}'\n")
            (root / "presets" / "b.py").write_text(f"# {PROJECT_WORD}_110\n")
            found = check_identifiers.violations([root])
            self.assertEqual(len(found), 2)

    def test_generators_may_not_import_each_other(self) -> None:
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "a.py").write_text("import numpy\n")
            (folder / "b.py").write_text("from strata_datagen.generators import a\n")
            (folder / "c.py").write_text("from . import b\n")
            self.assertEqual(len(check_generators.violations(folder)), 2)

    def test_decisions_are_short_single_line_bullets(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "DECISIONS.md"
            path.write_text("# D\n\n- fine\n- " + "x" * 199 + "\nwrapped text\n")
            self.assertEqual(len(check_decisions.violations(path)), 2)


class _Monitor:
    def __init__(self) -> None:
        self.samples = [10000, 12000]
        self.done = mock.Mock()

    def start(self) -> None:
        return None


def _frame_seconds(_shot: Path) -> list[float]:
    return [2.0, 4.0]


class HuskTest(unittest.TestCase):
    def test_gpu_failure_retries_with_the_cpu_device_and_says_so(self) -> None:
        job = husk.Job("s", "shots/s", ["husk", "scene/shot.usda"], ["beauty", "deep"])
        calls: list[dict[str, str]] = []

        def launch(_root: Path, _job: husk.Job, _husk: str, env: dict[str, str], _log: Path) -> int:
            calls.append(env)
            return 1 if env else 0

        with (
            mock.patch.object(husk, "_husk", launch),
            mock.patch.object(husk, "_VramMonitor", _Monitor),
            mock.patch.object(husk, "_frame_seconds", _frame_seconds),
            mock.patch.object(husk.time, "sleep"),
        ):
            record = husk.run(Path("/nonexistent"), job, "husk")
        self.assertEqual(calls, [husk.GPU_ONLY, {}])
        self.assertEqual(
            (record["device"], record["exit_code"], record["mean_frame_seconds"]), ("gpu+cpu", 0, 3.0)
        )


if __name__ == "__main__":
    unittest.main()
