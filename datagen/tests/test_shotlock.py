"""The shot writer lock: one writer at a time, and only a verifiably dead owner loses the lock."""

import fcntl
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

from strata_datagen import shotlock
from strata_datagen.qa import checks
from strata_datagen.render import pack
from tests.concurrency import (
    CONTEXT,
    FIRST,
    FRAMES,
    TIMEOUT_S,
    Paused,
    pack_paused,
    pack_shot,
    pausing,
    run,
)
from tests.fixtures import write_shot

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event


def hold_after_guard(
    shot: str, release: "Event", entered: "Event", proceed: "Event", results: "Queue[str]"
) -> None:
    """Take the shot lock, stopping before the guard that serializes breaking a stale lock."""
    with pausing(fcntl, "flock", entered, proceed):
        try:
            with shotlock.held(Path(shot)):
                results.put("acquired")
                release.wait(TIMEOUT_S)
        except shotlock.ShotLockedError:
            results.put("locked")


def dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    return process.pid


class ShotLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.shot = Path(self.directory.name) / "s"
        write_shot(self.shot, [FIRST], stride=1)
        pack.pack(self.shot, FRAMES)
        self.lock = self.shot / shotlock.LOCK

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_lock(self, **changes: object) -> str:
        text = json.dumps({**shotlock.owner(), **changes}, sort_keys=True) + "\n"
        self.lock.write_text(text)
        return text

    def test_the_lock_excludes_every_other_writer_of_the_shot(self) -> None:
        writer = Paused(pack_paused, str(self.shot))
        try:
            owner = json.loads(self.lock.read_text())
            self.assertEqual((owner["pid"], owner["host"]), (writer.process.pid, socket.gethostname()))
            self.assertTrue(run(pack_shot, str(self.shot)).startswith("ShotLockedError"))
            writers: list[Callable[[Path, range], object]] = [
                pack.pack,
                checks.accept,
                checks.prune,
                pack.remove_samples,
            ]
            for write in writers:
                with (
                    self.subTest(write.__name__),
                    self.assertRaisesRegex(shotlock.ShotLockedError, "locked by"),
                ):
                    write(self.shot, FRAMES)
            self.assertTrue((self.shot / "samples" / "s_0001").is_dir())
        finally:
            outcome = writer.finish()
        self.assertEqual(outcome, "ok")
        self.assertFalse(self.lock.exists())

    def test_offline_removal_refuses_while_a_live_owner_holds_the_shot(self) -> None:
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            text = self.write_lock(pid=sleeper.pid, start=shotlock.started(sleeper.pid))
            with self.assertRaises(shotlock.ShotLockedError):
                pack.remove_samples(self.shot, FRAMES)
            self.assertTrue((self.shot / "samples" / "s_0001").is_dir())
            self.assertEqual(self.lock.read_text(), text)
        finally:
            sleeper.kill()
            sleeper.wait()
        self.assertEqual(pack.remove_samples(self.shot, FRAMES), [1])
        self.assertFalse(self.lock.exists())

    def test_only_a_verifiably_dead_owner_loses_the_lock(self) -> None:
        cases: list[tuple[str, dict[str, object], bool]] = [
            ("exited", {"pid": dead_pid(), "start": "0"}, True),
            ("pid reused", {"start": "0"}, True),
            ("rebooted", {"boot": "another boot"}, True),
            ("alive", {}, False),
            ("other host", {"host": "another-host", "pid": dead_pid(), "start": "0"}, False),
        ]
        for name, changes, broken in cases:
            with self.subTest(name):
                text = self.write_lock(**changes)
                if broken:
                    pack.pack(self.shot, FRAMES)
                    self.assertFalse(self.lock.exists())
                else:
                    with self.assertRaises(shotlock.ShotLockedError):
                        pack.pack(self.shot, FRAMES)
                    self.assertEqual(self.lock.read_text(), text)
                    self.lock.unlink()

    def test_two_processes_breaking_one_stale_lock_never_both_take_it(self) -> None:
        self.write_lock(pid=dead_pid(), start="0")
        release = CONTEXT.Event()
        first = Paused(hold_after_guard, str(self.shot), release)
        second = Paused(hold_after_guard, str(self.shot), release)
        try:
            outcomes = [first.resume(), second.resume()]
        finally:
            release.set()
            first.finish()
            second.finish()
        self.assertEqual(outcomes, ["acquired", "locked"])
        self.assertFalse(self.lock.exists())


if __name__ == "__main__":
    unittest.main()
