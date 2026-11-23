"""Every CLI command that changes a shot directory holds the shot's writer lock while it does."""

import argparse
import contextlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_datagen import cli, files, shotlock
from strata_datagen.render import husk
from tests.concurrency import free
from tests.fixtures import SKY, Sample, write_deep, write_shot
from tests.shotwrites import watching

SHOT_WRITERS = {"build", "render", "render-task", "ingest", "pack", "accept", "remove-samples", "qa"}
SHOT_FREE = {"fetch", "prepare", "humans", "morphs", "creatures"}
PIXELS: list[Sample | list[Sample]] = [[Sample(0.5, 1.0, 2), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]


def commands() -> set[str]:
    usage = cli.parser().format_usage()
    match = re.search(r"\{([^}]*)\}", usage)
    assert match is not None
    return set(match.group(1).split(","))


def frame_args(name: str, shot: Path) -> list[str]:
    return [name, "--shot", str(shot), "--first", "1", "--count", "1"]


def fake_render(shot: Path, _frames: range, _husk: str, lock: int) -> dict[str, object]:
    """Stand in for husk: the descriptor it would inherit must be the shot's held lock."""
    held, passed = os.fstat(lock), (shot / shotlock.LOCK).stat()
    assert (held.st_dev, held.st_ino) == (passed.st_dev, passed.st_ino)
    files.write_text(shot / "render.log", "rendered\n")
    return {"exit_code": 0, "device": "gpu", "mean_frame_seconds": 1.0}


def build_fixture(root: Path) -> None:
    def plan(spec: SimpleNamespace, *_: object) -> SimpleNamespace:
        return SimpleNamespace(
            directory=root / "shots" / spec.shot_id,
            frames=(1,),
            track=SimpleNamespace(move="locked"),
            light=SimpleNamespace(hdri=SimpleNamespace(key="sky")),
        )

    def compose_write(shot: SimpleNamespace, *_: object) -> None:
        (shot.directory / "scene").mkdir(parents=True, exist_ok=True)
        files.write_text(shot.directory / "scene" / "shot.usda", "#usda 1.0\n")

    def export_write(shot: SimpleNamespace, *_: object) -> None:
        files.write_text(shot.directory / "manifest.json", "{}\n")

    def specs(_args: argparse.Namespace) -> list[SimpleNamespace]:
        return [SimpleNamespace(shot_id="built")]

    def load(_root: Path) -> None:
        return None

    with (
        mock.patch.object(cli, "_specs", specs),
        mock.patch.object(cli.catalogue, "load", load),
        mock.patch.object(cli.compose, "plan", plan),
        mock.patch.object(cli.compose, "write", compose_write),
        mock.patch.object(cli.export, "write", export_write),
    ):
        cli.main(["build", "--root", str(root)])


class ShotWriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_every_command_is_classified_as_writing_shots_or_not(self) -> None:
        self.assertEqual(SHOT_WRITERS | SHOT_FREE, commands())
        self.assertEqual(SHOT_WRITERS & SHOT_FREE, set())

    def run_command(self, argv: list[str]) -> None:
        with contextlib.suppress(SystemExit):
            cli.main(argv)

    def test_frame_commands_write_only_under_the_lock(self) -> None:
        shot = self.root / "shots" / "s"
        write_shot(shot, [PIXELS])
        with watching(self.root) as watch:
            for name in ("pack", "accept", "remove-samples"):
                self.run_command(frame_args(name, shot))
        self.assertGreater(watch.writes, 0)
        self.assertEqual((watch.unlocked, watch.unexplained), ([], []))

    def test_build_writes_only_under_the_lock(self) -> None:
        with watching(self.root) as watch:
            build_fixture(self.root)
        self.assertGreater(watch.writes, 0)
        self.assertEqual((watch.unlocked, watch.unexplained), ([], []))

    def test_render_writes_only_under_one_hold_that_husk_inherits(self) -> None:
        shot = self.root / "shots" / "r"
        write_shot(shot, [PIXELS])
        husk.write_job_list(self.root, husk.jobs(shot, self.root, (1,), husk.CHUNK_FRAMES))
        with watching(self.root) as watch, mock.patch.object(cli.husk, "run", fake_render):
            cli.main(["render", "--root", str(self.root)])
        self.assertGreater(watch.writes, 0)
        self.assertEqual((watch.unlocked, watch.unexplained), ([], []))
        self.assertTrue(json.loads((shot / "manifest.json").read_text())["timings"]["post_passed"])
        self.assertTrue((shot / "layers" / "0001.exr").exists())
        with watching(self.root) as watch, mock.patch.object(cli.husk, "run", fake_render):
            cli.main(["render-task", "--shot", str(shot), "--first", "1", "--count", "1"])
        self.assertGreater(watch.writes, 0)
        self.assertEqual((watch.unlocked, watch.unexplained), ([], []))

    def test_qa_writes_and_moves_only_under_the_lock(self) -> None:
        write_shot(self.root / "shots" / "good", [PIXELS])
        cli.main(frame_args("pack", self.root / "shots" / "good"))
        write_shot(self.root / "shots" / "bad", [PIXELS])
        (self.root / "shots" / "bad" / "deep" / "0001.exr").unlink()
        with watching(self.root) as watch:
            cli.main(["qa", "--root", str(self.root)])
        self.assertGreater(watch.writes, 0)
        self.assertEqual((watch.unlocked, watch.unexplained), ([], []))
        self.assertTrue((self.root / "rejected" / "bad" / "qa.json").exists())
        self.assertFalse((self.root / "shots" / "bad").exists())
        self.assertTrue(free(self.root / "shots" / "good"))


class AuditTest(unittest.TestCase):
    """The audit itself: writes it used to miss are reported, and locked Python writes are not."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.shot = self.root / "shots" / "s"
        write_shot(self.shot, [PIXELS])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_around_python(self) -> None:
        folder = os.open(self.shot, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.close(os.open("at.txt", os.O_WRONLY | os.O_CREAT, 0o644, dir_fd=folder))
            os.symlink("camera.json", "symlink.json", dir_fd=folder)
            os.link("camera.json", "hardlink.json", src_dir_fd=folder, dst_dir_fd=folder)
            os.rename("hardlink.json", "renamed.json", src_dir_fd=folder, dst_dir_fd=folder)
        finally:
            os.close(folder)

    def test_descriptor_relative_writes_and_links_without_the_lock_are_reported(self) -> None:
        with watching(self.root) as watch:
            self.write_around_python()
        self.assertEqual(
            [entry.split()[-1] for entry in watch.unlocked],
            [
                f"shots/s/{name}"
                for name in ("at.txt", "symlink.json", "hardlink.json", "hardlink.json", "renamed.json")
            ],
        )
        self.assertEqual(watch.unexplained, [])

    def test_the_same_writes_under_the_lock_are_not(self) -> None:
        with watching(self.root) as watch, shotlock.held(self.shot):
            self.write_around_python()
        self.assertEqual((watch.unlocked, watch.unexplained), ([], []))
        self.assertGreater(watch.writes, 0)

    def test_a_native_write_python_never_sees_is_reported_by_the_snapshot(self) -> None:
        with watching(self.root) as watch, shotlock.held(self.shot):
            write_deep(self.shot / "deep" / "0001.exr", [SKY, SKY, SKY])
            write_deep(self.shot / "deep" / "0002.exr", [SKY, SKY, SKY])
        self.assertEqual(watch.unexplained, ["shots/s/deep/0001.exr", "shots/s/deep/0002.exr"])


if __name__ == "__main__":
    unittest.main()
