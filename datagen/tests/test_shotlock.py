"""The shot writer lock: one holder at a time, and the kernel, not a pid, decides when it is free."""

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
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
    free,
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

SLEEPER = "import time; time.sleep(600)"


def hold_at_flock(
    shot: str, release: "Event", entered: "Event", proceed: "Event", results: "Queue[str]"
) -> None:
    """Take the shot lock, stopping after its file is open and before it is locked."""
    with pausing(fcntl, "flock", entered, proceed):
        try:
            with shotlock.held(Path(shot)):
                results.put("acquired")
                release.wait(TIMEOUT_S)
        except shotlock.ShotLockedError:
            results.put("locked")


def hand_to_renderer(shot: str, results: "Queue[str]") -> None:
    """Hold the shot, start a renderer that inherits the lock descriptor, report its pid and wait."""
    with shotlock.held(Path(shot)) as hold:
        child = subprocess.Popen([sys.executable, "-c", SLEEPER], pass_fds=(hold.descriptor,))
        results.put(str(child.pid))
        time.sleep(TIMEOUT_S)


def gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return True
    return state in {"Z", "X"}


def wait_gone(pid: int) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not gone(pid):
        if time.monotonic() > deadline:
            raise TimeoutError(pid)
        time.sleep(0.05)


class ShotLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.shot = Path(self.directory.name) / "s"
        write_shot(self.shot, [FIRST], stride=1)
        pack.pack(self.shot, FRAMES)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_the_lock_excludes_every_other_writer_of_the_shot(self) -> None:
        writer = Paused(pack_paused, str(self.shot))
        try:
            owner = json.loads((self.shot / shotlock.OWNER).read_text())
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
        self.assertTrue(free(self.shot))

    def test_a_lock_file_without_a_holder_never_blocks_whatever_its_record_says(self) -> None:
        record = {**shotlock.owner(), "host": "another-host", "pid": 1, "start": "0", "boot": "b"}
        for name in (shotlock.LOCK, shotlock.OWNER):
            (self.shot / name).write_text(json.dumps(record) + "\n")
        self.assertEqual(pack.remove_samples(self.shot, FRAMES), [1])
        self.assertTrue((self.shot / shotlock.LOCK).exists())

    def test_a_renderer_holding_the_inherited_descriptor_keeps_the_shot_after_its_parent_dies(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(sig.name):
                results: Queue[str] = CONTEXT.Queue()
                parent = CONTEXT.Process(target=hand_to_renderer, args=(str(self.shot), results))
                parent.start()
                renderer = int(results.get(timeout=TIMEOUT_S))
                try:
                    os.kill(parent.pid or 0, sig)
                    parent.join(TIMEOUT_S)
                    self.assertEqual(parent.exitcode, -sig)
                    self.assertFalse(gone(renderer))
                    with self.assertRaisesRegex(shotlock.ShotLockedError, "locked by"):
                        pack.remove_samples(self.shot, FRAMES)
                    self.assertTrue((self.shot / "samples" / "s_0001").is_dir())
                finally:
                    os.kill(renderer, signal.SIGKILL)
                wait_gone(renderer)
                self.assertTrue(free(self.shot))

    def test_holds_nest_within_a_process_and_release_with_the_outermost(self) -> None:
        with shotlock.held(self.shot) as outer:
            with shotlock.held(self.shot) as inner:
                self.assertIs(inner, outer)
                self.assertTrue(run(pack_shot, str(self.shot)).startswith("ShotLockedError"))
            self.assertTrue(run(pack_shot, str(self.shot)).startswith("ShotLockedError"))
        self.assertEqual(run(pack_shot, str(self.shot)), "ok")

    def test_two_processes_racing_for_the_lock_never_both_take_it(self) -> None:
        release = CONTEXT.Event()
        first = Paused(hold_at_flock, str(self.shot), release)
        second = Paused(hold_at_flock, str(self.shot), release)
        try:
            outcomes = [first.resume(), second.resume()]
        finally:
            release.set()
            first.finish()
            second.finish()
        self.assertEqual(outcomes, ["acquired", "locked"])
        self.assertTrue(free(self.shot))

    def test_a_contender_that_opened_the_lock_before_the_shot_moved_refuses(self) -> None:
        release = CONTEXT.Event()
        contender = Paused(hold_at_flock, str(self.shot), release)
        destination = self.shot.with_name("moved")
        try:
            with shotlock.held(self.shot) as hold:
                hold.move(destination)
            outcome = contender.resume()
        finally:
            release.set()
            contender.finish()
        self.assertEqual(outcome, "locked")
        self.assertFalse(self.shot.exists())
        self.assertTrue(free(destination))


if __name__ == "__main__":
    unittest.main()
