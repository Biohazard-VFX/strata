"""Downloaded farm output enters a shot only through ingest, under the shot's writer lock."""

import shutil
import tempfile
import unittest
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from strata_datagen import cli, shotlock
from strata_datagen.render import husk, ingest, pack
from tests.concurrency import FIRST, FRAMES, Paused, free, pack_paused
from tests.fixtures import SKY, Sample, write_shot
from tests.shotwrites import watching

OTHER: list[Sample | list[Sample]] = [[Sample(0.4, 1.0, 2), SKY], [Sample(0.1, 0.5, 1), SKY], SKY]


def contents(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in folder.rglob("*")
        if p.is_file() and not p.name.startswith(shotlock.LOCK)
    }


class IngestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        base = Path(self.directory.name)
        self.root = base / "build"
        self.shot = self.root / "shots" / "s"
        write_shot(self.shot, [FIRST], stride=1)
        self.download = base / "download"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def farm(self, pixels: list[Sample | list[Sample]]) -> None:
        """What a farm node renders and packs in its own copy of the shot, downloaded output folders only."""
        remote = Path(self.directory.name) / "remote" / "s"
        shutil.rmtree(remote.parent, ignore_errors=True)
        shutil.rmtree(self.download, ignore_errors=True)
        write_shot(remote, [pixels], stride=1)
        pack.pack(remote, FRAMES)
        for folder in husk.OUTPUTS:
            if (remote / folder).is_dir():
                shutil.copytree(remote / folder, self.download / folder)

    def ingest(self) -> None:
        cli.main(["ingest", "--shot", str(self.shot), "--source", str(self.download)])

    def test_ingest_moves_output_in_only_under_the_lock_and_leaves_no_staging(self) -> None:
        self.farm(OTHER)
        with watching(self.root) as watch:
            self.ingest()
        self.assertGreater(watch.writes, 0)
        self.assertEqual(watch.unlocked, [])
        for folder in ("beauty", "deep", "layers", "samples", "merged"):
            self.assertEqual(contents(self.shot / folder), contents(self.download / folder), folder)
        self.assertEqual(list((self.shot / pack.STAGING).iterdir()), [])
        self.assertTrue(free(self.shot))

    def test_ingest_is_refused_while_another_process_holds_the_shot(self) -> None:
        self.farm(OTHER)
        rendered = {folder: contents(self.shot / folder) for folder in ("beauty", "deep")}
        writer = Paused(pack_paused, str(self.shot))
        try:
            with self.assertRaisesRegex(shotlock.ShotLockedError, "locked by"):
                self.ingest()
        finally:
            self.assertEqual(writer.finish(), "ok")
        self.assertEqual({folder: contents(self.shot / folder) for folder in rendered}, rendered)

    def test_published_frames_stay_write_once(self) -> None:
        pack.pack(self.shot, FRAMES)
        self.farm(FIRST)
        moved = ingest.ingest(self.shot, self.download)
        self.assertNotIn("samples/s_0001", moved)
        self.assertNotIn("merged/s_0001", moved)
        merged_path = self.download / "merged" / "s_0001" / "deep.npz"
        merged_path.write_bytes(b"different merged deep")
        kept = contents(self.shot)
        with self.assertRaisesRegex(ValueError, "different published frame"):
            self.ingest()
        self.assertEqual(contents(self.shot), kept)
        self.assertIn("beauty/0001.exr", moved)
        self.farm(OTHER)
        kept = contents(self.shot)
        with self.assertRaisesRegex(ValueError, "different published frame"):
            self.ingest()
        self.assertEqual(contents(self.shot), kept)

    def test_a_failed_commit_leaves_whole_units_and_a_retry_completes_it(self) -> None:
        self.farm(OTHER)
        downloaded = contents(self.download)
        units = {
            Path(*Path(key).parts[:2]) if Path(key).parts[0] in ingest.SIDECARS else Path(key)
            for key in downloaded
        }
        self.assertGreater(len(units), 2)
        for failing_move in range(1, len(downloaded) + 2):
            shot = self.root / "shots" / f"retry{failing_move}"
            write_shot(shot, [FIRST], stride=1)
            before = contents(shot)
            try:
                with self.failing_move(failing_move):
                    ingest.ingest(shot, self.download)
            except OSError as error:
                self.assertEqual(str(error), "disk full")
            else:
                self.assertGreater(failing_move, len(units))
                break
            after = contents(shot)
            for unit in units:
                keys = {key for key in downloaded if Path(key).is_relative_to(unit)}
                landed = {key: after.get(key) for key in keys} == {key: downloaded[key] for key in keys}
                kept = {key: after.get(key) for key in keys} == {key: before.get(key) for key in keys}
                self.assertTrue(landed or kept, (failing_move, unit))
            self.assertEqual(list((shot / pack.STAGING).iterdir()), [])
            self.assertTrue(free(shot))
            ingest.ingest(shot, self.download)
            final = contents(shot)
            self.assertEqual({key: final.get(key) for key in downloaded}, downloaded)
        else:
            self.fail("ingest never completed without a failing move")

    @staticmethod
    @contextmanager
    def failing_move(failing: int) -> Generator[None]:
        """Fail the `failing`-th rename out of staging, as a full disk or vanished mount would."""
        replace = Path.replace
        moves: list[Path] = []

        def move(path: Path, target: Path) -> Path:
            if pack.STAGING in path.parts:
                moves.append(path)
                if len(moves) == failing:
                    raise OSError("disk full")
            return replace(path, target)

        with mock.patch.object(Path, "replace", move):
            yield

    def test_anything_but_output_folders_of_regular_files_is_refused(self) -> None:
        self.farm(OTHER)
        kept = contents(self.shot)
        cases = {
            "unknown folder": (self.download / "scene").mkdir,
            "symlink": lambda: (self.download / "beauty" / "0002.exr").symlink_to(self.shot / "camera.json"),
        }
        for name, spoil in cases.items():
            with self.subTest(name):
                self.farm(OTHER)
                spoil()
                with self.assertRaises(ValueError):
                    self.ingest()
                self.assertEqual(contents(self.shot), kept)


if __name__ == "__main__":
    unittest.main()
