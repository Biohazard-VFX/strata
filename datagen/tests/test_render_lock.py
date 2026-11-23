"""Generated render jobs and their renderer hold the shot lock; competing writers are refused meanwhile."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from strata_datagen import cli, shotlock
from strata_datagen.qa import checks
from strata_datagen.render import husk, pack
from tests.concurrency import FIRST, TIMEOUT_S, free
from tests.fixtures import write_shot

FRAMES = range(1, 3)
STARTED, GO = "started", "go"
FAKE_HUSK = f"""#!{sys.executable}
import os, sys, time
from pathlib import Path
barrier = Path(os.environ["FAKE_HUSK_BARRIER"])
(barrier / "{STARTED}").write_text(str(os.getpid()))
deadline = time.monotonic() + {TIMEOUT_S}
while not (barrier / "{GO}").exists() and time.monotonic() < deadline:
    time.sleep(0.05)
"""
ENTRY_POINT = f"""#!/usr/bin/env sh
exec "{sys.executable}" -c "import sys; from strata_datagen.cli import main; main()" "$@"
"""


def executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def gone(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return True
    return state in {"Z", "X"}


class RenderLockTest(unittest.TestCase):
    """A fake husk stops at a barrier, so competitors run while the renderer is writing."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        tools = Path(self.directory.name) / "bin"
        tools.mkdir()
        executable(tools / "husk", FAKE_HUSK)
        executable(tools / "nvidia-smi", "#!/usr/bin/env sh\necho 100\n")
        executable(tools / "strata-datagen", ENTRY_POINT)
        self.husk = str(tools / "husk")
        self.path = f"{tools}{os.pathsep}{os.environ['PATH']}"

    def tearDown(self) -> None:
        for barrier in Path(self.directory.name).glob("*/barrier"):
            (barrier / GO).touch()
        self.directory.cleanup()

    def build(self, name: str) -> dict[str, str]:
        """A packed, accepted two-frame shot with its job list; returns the environment for its job."""
        self.root = Path(self.directory.name) / name
        self.shot = self.root / "shots" / "s"
        write_shot(self.shot, [FIRST, FIRST])
        pack.pack(self.shot, FRAMES)
        self.assertEqual([r.failures for r in checks.accept(self.shot, FRAMES)], [[], []])
        husk.write_job_list(self.root, husk.jobs(self.shot, self.root, tuple(FRAMES), husk.CHUNK_FRAMES))
        self.barrier = self.root / "barrier"
        self.barrier.mkdir()
        return {
            **os.environ,
            "PATH": self.path,
            "FAKE_HUSK_BARRIER": str(self.barrier),
        }

    def renderer(self, process: subprocess.Popen[bytes]) -> int:
        deadline = time.monotonic() + TIMEOUT_S
        while not (self.barrier / STARTED).exists():
            if process.poll() is not None or time.monotonic() > deadline:
                self.fail(f"renderer never started; exit {process.returncode}")
            time.sleep(0.05)
        return int((self.barrier / STARTED).read_text())

    def assert_competitors_refused(self) -> None:
        with self.assertRaisesRegex(shotlock.ShotLockedError, "locked by"):
            checks.accept(self.shot, FRAMES)
        self.assertTrue((self.shot / "deep" / "0002.exr").exists())
        with self.assertRaisesRegex(shotlock.ShotLockedError, "locked by"):
            cli.main(["qa", "--root", str(self.root)])
        self.assertTrue(self.shot.is_dir())
        self.assertFalse((self.shot / "qa.json").exists())

    def test_a_generated_job_holds_the_shot_while_its_renderer_writes(self) -> None:
        for name in ("jobs.sh", "jobs.json"):
            with self.subTest(name):
                env = self.build(name)
                (task,) = json.loads((self.root / "jobs.json").read_text())
                launch = {
                    "jobs.sh": (["sh", str(self.root / "jobs.sh")], self.root),
                    "jobs.json": (task["command"], self.root / task["workdir"]),
                }
                command, cwd = launch[name]
                job = subprocess.Popen(command, cwd=cwd, env=env)
                try:
                    self.renderer(job)
                    self.assert_competitors_refused()
                finally:
                    (self.barrier / GO).touch()
                    job.wait(TIMEOUT_S)
                self.assertEqual(job.returncode, 0)
                self.assertTrue((self.shot / "deep" / "0002.exr").exists())
                self.assertTrue(free(self.shot))

    def test_the_shot_stays_locked_while_the_renderer_outlives_a_killed_render(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(sig.name):
                env = self.build(sig.name)
                command = ["strata-datagen", "render", "--root", str(self.root), "--husk", self.husk]
                render = subprocess.Popen(command, env=env)
                renderer = self.renderer(render)
                try:
                    render.send_signal(sig)
                    render.wait(TIMEOUT_S)
                    self.assertEqual(render.returncode, -sig)
                    self.assertFalse(gone(renderer))
                    self.assert_competitors_refused()
                finally:
                    (self.barrier / GO).touch()
                deadline = time.monotonic() + TIMEOUT_S
                while not gone(renderer) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(free(self.shot))


if __name__ == "__main__":
    unittest.main()
