"""Write-once sidecar publication under concurrency: separate processes, interleaved by injected barriers."""

import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from strata_datagen.exr import read_deep
from strata_datagen.render import pack
from strata_dataset import samples as contract_samples
from tests.concurrency import FIRST, FRAMES, Paused, at_call, outcome, pack_paused, pack_shot, pausing, run
from tests.fixtures import SKY, Sample, write_beauty, write_deep, write_shot

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event

SECOND: list[Sample | list[Sample]] = [SKY, [Sample(0.4, 0.25, 1), SKY], SKY]


def publish_paused(
    deep: str, published: str, entered: "Event", proceed: "Event", results: "Queue[str]"
) -> None:
    """Publish sidecars directly, without the shot lock, stopping just before the publishing rename."""
    frame = pack.samples(read_deep(Path(deep)))
    beauty = Path(deep).with_suffix(".beauty.exr")
    with pausing(Path, "rename", entered, proceed):
        results.put(outcome(lambda: pack.write_samples(frame, beauty, Path(published))))


def rerender(shot: Path, pixels: list[Sample | list[Sample]]) -> None:
    write_deep(shot / "deep" / "0001.exr", pixels)
    write_beauty(shot / "beauty" / "0001.exr", read_deep(shot / "deep" / "0001.exr"))


def published(shot: Path) -> Path:
    return shot / "samples" / "s_0001"


def files(shot: Path) -> dict[str, tuple[int, int]]:
    """Every published file with its inode and modification time."""
    root = published(shot)
    found = [root, *sorted(root.rglob("*"))]
    return {p.relative_to(shot).as_posix(): (p.stat().st_ino, p.stat().st_mtime_ns) for p in found}


def arrays_equal(found: dict[str, NDArray[np.generic]], expected: dict[str, NDArray[np.generic]]) -> bool:
    return found.keys() == expected.keys() and all(np.array_equal(found[k], v) for k, v in expected.items())


def sidecars(deep: Path) -> dict[str, NDArray[np.generic]]:
    return {**pack.images(deep.with_suffix(".beauty.exr")), **pack.samples(read_deep(deep)).arrays()}


def staging_left(shot: Path) -> list[str]:
    staging = shot / pack.STAGING
    return sorted(p.name for p in staging.iterdir()) if staging.is_dir() else []


class PublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.shot = Path(self.directory.name) / "s"
        write_shot(self.shot, [FIRST], stride=1)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def load_while_packing(self) -> tuple[str, dict[str, NDArray[np.generic]]]:
        """Load the published frame, running a whole pack in another process after its first array."""
        outcomes: list[str] = []
        with at_call(np, "load", lambda: outcomes.append(run(pack_shot, str(self.shot))), number=1):
            found = contract_samples.load(published(self.shot))
        return outcomes[0], found

    def test_an_identical_repack_is_a_no_op(self) -> None:
        pack.pack(self.shot, FRAMES)
        before = files(self.shot)
        pack.pack(self.shot, FRAMES)
        self.assertFalse(published(self.shot).is_symlink())
        self.assertEqual(files(self.shot), before)
        self.assertEqual(staging_left(self.shot), [])

    def test_a_different_repack_is_refused_until_removed_offline(self) -> None:
        pack.pack(self.shot, FRAMES)
        first = contract_samples.read(published(self.shot))
        rerender(self.shot, SECOND)
        with self.assertRaisesRegex(ValueError, "immutable"):
            pack.pack(self.shot, FRAMES)
        kept = contract_samples.read(published(self.shot))
        self.assertEqual(kept.identity, first.identity)
        self.assertTrue(arrays_equal(kept.arrays, first.arrays))
        self.assertEqual(pack.remove_samples(self.shot, FRAMES), [1])
        self.assertFalse(published(self.shot).exists())
        pack.pack(self.shot, FRAMES)
        self.assertEqual(contract_samples.load(published(self.shot))["ids"].tolist(), [0, 1, 0, 0])

    def test_concurrent_packers_never_replace_or_delete_a_published_frame(self) -> None:
        pack.pack(self.shot, FRAMES)
        first = contract_samples.read(published(self.shot))
        rerender(self.shot, SECOND)
        writer = Paused(pack_paused, str(self.shot))
        try:
            second = run(pack_shot, str(self.shot))
        finally:
            outcome = writer.finish()
        self.assertTrue(second.startswith("ShotLockedError"), second)
        self.assertTrue(outcome.startswith("ValueError") and "immutable" in outcome, outcome)
        kept = contract_samples.read(published(self.shot))
        self.assertEqual(kept.identity, first.identity)
        self.assertTrue(arrays_equal(kept.arrays, first.arrays))
        self.assertEqual(staging_left(self.shot), [])

    def test_a_reader_never_mixes_or_loses_a_frame_while_another_pack_runs(self) -> None:
        pack.pack(self.shot, FRAMES)
        expected = contract_samples.load(published(self.shot))
        for pixels, wanted in ((FIRST, "ok"), (SECOND, "ValueError")):
            with self.subTest(wanted):
                rerender(self.shot, pixels)
                outcome, found = self.load_while_packing()
                mixed = sorted(k for k, v in expected.items() if not np.array_equal(found[k], v))
                self.assertEqual(mixed, [])
                self.assertTrue(outcome.startswith(wanted), outcome)

    def test_racing_writers_without_the_lock_still_cannot_replace_a_published_frame(self) -> None:
        for pixels, wanted in ((FIRST, "ok"), (SECOND, "ValueError")):
            with self.subTest(wanted), tempfile.TemporaryDirectory() as tmp:
                deeps = [Path(tmp) / f"{name}.exr" for name in ("a", "b")]
                for deep, frame in zip(deeps, (FIRST, pixels), strict=True):
                    write_deep(deep, frame)
                    write_beauty(deep.with_suffix(".beauty.exr"), read_deep(deep))
                target = Path(tmp) / "samples" / "s_0001"
                first = Paused(publish_paused, str(deeps[0]), str(target))
                second = Paused(publish_paused, str(deeps[1]), str(target))
                self.assertEqual(first.finish(), "ok")
                outcome = second.finish()
                self.assertTrue(outcome.startswith(wanted), outcome)
                self.assertTrue(arrays_equal(contract_samples.load(target), sidecars(deeps[0])))
                self.assertEqual(staging_left(Path(tmp)), [])


if __name__ == "__main__":
    unittest.main()
