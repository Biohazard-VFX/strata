"""Writers in separate spawned processes, stopped and resumed at injected barriers."""

import multiprocessing
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import numpy as np

from strata_datagen import shotlock
from strata_datagen.render import pack
from tests.fixtures import SKY, Sample

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event

CONTEXT = multiprocessing.get_context("spawn")
TIMEOUT_S = 60.0
FIRST: list[Sample | list[Sample]] = [[Sample(0.5, 1.0, 2), SKY], [Sample(0.2, 0.5, 1), SKY], SKY]
FRAMES = range(1, 2)


def outcome(run: Callable[[], object]) -> str:
    try:
        run()
    except (OSError, ValueError, RuntimeError) as error:
        return f"{type(error).__name__}: {error}"
    return "ok"


@contextmanager
def at_call(owner: object, name: str, action: Callable[[], object], number: int = 0) -> Generator[None]:
    """Run `action` once, just before call `number` (from 0) of `owner.name` proceeds."""
    original: Callable[..., object] = getattr(owner, name)
    calls: list[None] = []

    def counted(*args: object, **kwargs: object) -> object:
        if len(calls) == number:
            action()
        calls.append(None)
        return original(*args, **kwargs)

    with mock.patch.object(owner, name, counted):
        yield


def pausing(owner: object, name: str, entered: "Event", proceed: "Event") -> AbstractContextManager[None]:
    """Stop at the first call of `owner.name` until `proceed`, after signalling `entered`."""
    return at_call(owner, name, lambda: (entered.set(), proceed.wait(TIMEOUT_S)))


def free(shot: Path) -> bool:
    """True when this process, holding nothing, can take the shot's lock now."""
    try:
        with shotlock.held(shot):
            return True
    except shotlock.ShotLockedError:
        return False


def pack_shot(shot: str, results: "Queue[str]") -> None:
    results.put(outcome(lambda: pack.pack(Path(shot), FRAMES)))


def pack_paused(shot: str, entered: "Event", proceed: "Event", results: "Queue[str]") -> None:
    """Pack, stopping inside the shot's writer section when the sidecars are first encoded."""
    with pausing(np, "save", entered, proceed):
        results.put(outcome(lambda: pack.pack(Path(shot), FRAMES)))


def run(target: Callable[..., None], *args: object) -> str:
    results: Queue[str] = CONTEXT.Queue()
    process = CONTEXT.Process(target=target, args=(*args, results))
    process.start()
    outcome = results.get(timeout=TIMEOUT_S)
    process.join(TIMEOUT_S)
    return outcome


class Paused:
    """A process started with (*args, entered, proceed, results) and stopped at its barrier."""

    def __init__(self, target: Callable[..., None], *args: object) -> None:
        self.entered, self.proceed = CONTEXT.Event(), CONTEXT.Event()
        self.results: Queue[str] = CONTEXT.Queue()
        self.outcome: str | None = None
        self.process = CONTEXT.Process(target=target, args=(*args, self.entered, self.proceed, self.results))
        self.process.start()
        deadline = time.monotonic() + TIMEOUT_S
        while not self.entered.wait(0.1):
            if not self.process.is_alive():
                raise RuntimeError(f"stopped before its barrier: {self.results.get(timeout=TIMEOUT_S)}")
            if time.monotonic() > deadline:
                self.process.kill()
                raise RuntimeError("never reached its barrier")

    def resume(self) -> str:
        """Let the process pass its barrier and return what it reported."""
        if self.outcome is None:
            self.proceed.set()
            self.outcome = self.results.get(timeout=TIMEOUT_S)
        return self.outcome

    def finish(self) -> str:
        outcome = self.resume()
        self.process.join(TIMEOUT_S)
        return outcome
