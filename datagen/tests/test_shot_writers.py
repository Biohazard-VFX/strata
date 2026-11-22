"""Every CLI command that changes a shot directory holds the shot's writer lock while it does."""

import argparse
import contextlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_datagen import cli, files
from strata_datagen.render import husk
from tests.concurrency import free
from tests.fixtures import SKY, Sample, write_shot
from tests.shotwrites import watching

SHOT_WRITERS = {"build", "render", "pack", "accept", "prune", "remove-samples", "qa"}
SHOT_FREE = {"fetch", "prepare", "humans", "morphs", "creatures"}
PIXELS: list[Sample | list[Sample]] = [[Sample(0.5, 1.0, 2), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]


def commands() -> set[str]:
    usage = cli.parser().format_usage()
    match = re.search(r"\{([^}]*)\}", usage)
    assert match is not None
    return set(match.group(1).split(","))


def frame_args(name: str, shot: Path) -> list[str]:
    return [name, "--shot", str(shot), "--first", "1", "--count", "1"]


def fake_render(root: Path, job: husk.Job, _husk: str) -> dict[str, object]:
    files.write_text(root / job.workdir / "render.log", "rendered\n")
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
        write_shot(shot, [PIXELS], stride=1)
        with watching(self.root) as watch:
            for name in ("pack", "accept", "prune", "remove-samples"):
                self.run_command(frame_args(name, shot))
        self.assertGreater(watch.writes, 0)
        self.assertEqual(watch.unlocked, [])

    def test_build_writes_only_under_the_lock(self) -> None:
        with watching(self.root) as watch:
            build_fixture(self.root)
        self.assertGreater(watch.writes, 0)
        self.assertEqual(watch.unlocked, [])

    def test_render_writes_only_under_the_lock_and_its_post_steps_take_it_themselves(self) -> None:
        shot = self.root / "shots" / "r"
        write_shot(shot, [PIXELS], stride=1)
        post = [["strata-datagen", "pack", "--shot", ".", "--first", "1", "--count", "1"]]
        job = husk.Job("r", "shots/r", ["husk"], [], post)
        files.write_text(self.root / "jobs.json", json.dumps([job.__dict__]))
        with watching(self.root) as watch, mock.patch.object(cli.husk, "run", fake_render):
            cli.main(["render", "--root", str(self.root)])
        self.assertGreater(watch.writes, 0)
        self.assertEqual(watch.unlocked, [])
        self.assertTrue(json.loads((shot / "manifest.json").read_text())["timings"]["post_passed"])
        self.assertTrue((shot / "layers" / "0001.exr").exists())

    def test_qa_writes_and_moves_only_under_the_lock(self) -> None:
        write_shot(self.root / "shots" / "good", [PIXELS], stride=1)
        cli.main(frame_args("pack", self.root / "shots" / "good"))
        write_shot(self.root / "shots" / "bad", [PIXELS], stride=1)
        (self.root / "shots" / "bad" / "deep" / "0001.exr").unlink()
        with watching(self.root) as watch:
            cli.main(["qa", "--root", str(self.root)])
        self.assertGreater(watch.writes, 0)
        self.assertEqual(watch.unlocked, [])
        self.assertTrue((self.root / "rejected" / "bad" / "qa.json").exists())
        self.assertFalse((self.root / "shots" / "bad").exists())
        self.assertTrue(free(self.root / "shots" / "good"))


if __name__ == "__main__":
    unittest.main()
