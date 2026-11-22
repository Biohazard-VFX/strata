import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from strata_datagen.render import husk
from tools import check_generators, check_identifiers

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


class _Monitor:
    def __init__(self) -> None:
        self.samples = [10000, 12000]
        self.done = mock.Mock()

    def start(self) -> None:
        return None


def _frame_seconds(_shot: Path) -> list[float]:
    return [2.0, 4.0]


class HuskTest(unittest.TestCase):
    def test_renders_only_on_the_gpu_and_never_retries_with_the_cpu_device(self) -> None:
        calls: list[dict[str, str]] = []

        def launch(_shot: Path, _frames: range, _husk: str, env: dict[str, str], _lock: int) -> int:
            calls.append(env)
            return 1

        with (
            mock.patch.object(husk, "_husk", launch),
            mock.patch.object(husk, "_VramMonitor", _Monitor),
            mock.patch.object(husk, "_frame_seconds", _frame_seconds),
            mock.patch.object(husk.time, "sleep"),
        ):
            record = husk.run(Path("/nonexistent"), range(1, 3), "husk", -1)
        self.assertEqual(calls, [{"KARMA_XPU_DISABLE_EMBREE_DEVICE": "1"}])
        self.assertEqual((record["device"], record["exit_code"]), ("gpu", 1))

if __name__ == "__main__":
    unittest.main()
