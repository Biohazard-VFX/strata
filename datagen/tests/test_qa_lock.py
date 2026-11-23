"""qa never moves or rewrites a shot another process is writing: it refuses, and the shot stays whole."""

import tempfile
import unittest
from pathlib import Path

from strata_datagen import cli, shotlock
from tests.concurrency import FIRST, FRAMES, Paused, free, pack_paused, pack_shot, run
from tests.fixtures import write_shot


class QaLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.shot = self.root / "shots" / "s"
        write_shot(self.shot, [FIRST])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_qa_refuses_a_shot_that_is_being_packed_and_rejects_it_afterwards(self) -> None:
        writer = Paused(pack_paused, str(self.shot))
        try:
            with self.assertRaisesRegex(shotlock.ShotLockedError, "locked by"):
                cli.main(["qa", "--root", str(self.root)])
            self.assertFalse(free(self.shot))
            self.assertFalse((self.root / "rejected").exists())
            self.assertFalse((self.shot / "qa.json").exists())
        finally:
            outcome = writer.finish()
        self.assertEqual(outcome, "ok")
        self.assertTrue((self.shot / "layers" / f"{FRAMES.start:04d}.exr").exists())
        self.assertTrue((self.shot / "samples" / "s_0001").is_dir())

        (self.shot / "deep" / "0001.exr").unlink()
        cli.main(["qa", "--root", str(self.root)])
        rejected = self.root / "rejected" / "s"
        self.assertFalse(self.shot.exists())
        self.assertTrue((rejected / "qa.json").exists())
        self.assertTrue(free(rejected))
        self.assertTrue((rejected / "samples" / "s_0001").is_dir())

    def test_a_shot_moved_while_held_releases_its_lock_at_the_new_place(self) -> None:
        destination = self.root / "moved"
        with shotlock.held(self.shot) as hold:
            hold.move(destination)
            self.assertTrue(run(pack_shot, str(destination)).startswith("ShotLockedError"))
        self.assertTrue(free(destination))

    def test_a_refused_move_releases_the_lock_where_the_shot_stays(self) -> None:
        destination = self.root / "taken"
        (destination / "x").mkdir(parents=True)
        with self.assertRaises(OSError), shotlock.held(self.shot) as hold:
            hold.move(destination)
        self.assertTrue(free(self.shot))
        self.assertFalse((destination / shotlock.LOCK).exists())


if __name__ == "__main__":
    unittest.main()
