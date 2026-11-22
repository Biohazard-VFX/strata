"""Step-keyed native CPU batches with bounded, persistent-worker prefetch."""

from collections import deque
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from strata_training.data.dataset import Frame, batch, input_batch
from strata_training.data.online import OnlineLoader, PreparedBatch
from strata_training.data.torch_online import finalize, prepare_raw


@dataclass
class LoadedBatch:
    """Own CPU or device tensors until consumed; mappings never cross worker queues."""

    selections: list[tuple[int, int, int]]
    inputs: dict[str, torch.Tensor]
    target: dict[str, torch.Tensor]
    pyramid: list[torch.Tensor]
    prepared: PreparedBatch | None

    def map(
        self, function: Callable[[torch.Tensor], torch.Tensor], pin: bool = False
    ) -> "LoadedBatch":
        target = {k: function(v) for k, v in self.target.items()}
        prepared = self.prepared
        if prepared is not None:
            prepared = replace(
                prepared,
                images=function(prepared.images),
                jobs=[job.map(function, pin) for job in prepared.jobs],
                target=target,
                scales=function(prepared.scales),
                normalized=function(prepared.normalized)
                if prepared.normalized is not None
                else None,
                cached_inputs={k: function(v) for k, v in (prepared.cached_inputs or {}).items()},
                cached_pyramid=[function(v) for v in (prepared.cached_pyramid or [])],
            )
        return LoadedBatch(
            self.selections,
            {k: function(v) for k, v in self.inputs.items()},
            target,
            [function(v) for v in self.pyramid],
            prepared,
        )

    def pin_memory(self) -> "LoadedBatch":
        return self.map(lambda value: value.pin_memory(), pin=True)

    def to(self, device: str) -> "LoadedBatch":
        result = self.map(lambda value: value.to(device, non_blocking=True))
        if result.prepared is not None and result.prepared.device_math:
            result.prepared = finalize(result.prepared, result.prepared.crop, result.prepared.halo)
            result.target = result.prepared.target
        return result


class NativeBatches(Dataset[LoadedBatch]):
    """Borrow metadata, own per-job RNG; step identity is independent of worker scheduling."""

    def __init__(
        self,
        frames: list[Frame],
        online: OnlineLoader | None,
        clips: list[list[int]],
        config: dict[str, Any],
        start_step: int = 0,
    ) -> None:
        self.frames, self.online, self.clips = frames, online, clips
        self.config, self.start_step = config, start_step

    def __len__(self) -> int:
        return self.config["steps"] - self.start_step

    def __getitem__(self, index: int) -> LoadedBatch:
        config = self.config
        rng = np.random.default_rng(
            np.random.SeedSequence([config["seed"], self.start_step + index + 1])
        )
        crop, halo = config["crop"], config["halo"]
        decoder = config["mode"] == "decoder"
        selections: list[tuple[int, int, int]] = []
        while len(selections) < config["batch_size"]:
            clip = self.clips[int(rng.integers(len(self.clips)))]
            coordinates = self.frames[clip[0]].crop(rng, crop + 2 * halo)
            selections.extend(
                (i, *coordinates) for i in clip[: config["batch_size"] - len(selections)]
            )
        if self.online is None:
            inputs, target, pyramid = batch(self.frames, selections, crop, halo, "cpu", decoder)
            return LoadedBatch(selections, inputs, target, pyramid, None)
        if config.get("loader_math", "numpy") == "torch":
            prepared = prepare_raw(self.online, selections, rng, crop, halo)
        else:
            prepared = self.online.prepare(selections, rng, crop, halo, "cpu")
        if prepared.clean:
            prepared.cached_inputs, prepared.cached_pyramid = input_batch(
                self.frames, [selections[i] for i in prepared.clean], crop, halo, "cpu", decoder
            )
        return LoadedBatch(selections, {}, prepared.target, [], prepared)


@contextmanager
def device_batches(dataset: NativeBatches, device: str) -> Generator[Iterator[LoadedBatch]]:
    """Own four lookahead batches; join preparation on exit, including early stop or errors.

    A single consumer thread owns iteration. CUDA preparation uses a private stream
    and worker thread; returned tensors are ordered and retained on the consumer stream.
    """
    source = iter(batches(dataset))
    if torch.device(device).type != "cuda":
        consumer = (row.to(device) for row in source)
        try:
            yield consumer
        finally:
            consumer.close()
        return
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))

    def prepare() -> tuple[LoadedBatch, torch.cuda.Event] | None:
        raw = next(source, None)
        if raw is None:
            return None
        with torch.cuda.device(device), torch.cuda.stream(stream), torch.no_grad():
            loaded = raw.to(device)
            ready = torch.cuda.Event()
            ready.record(stream)
        return loaded, ready

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="strata-prepare") as executor:
        futures = deque(executor.submit(prepare) for _ in range(4))

        def consume() -> Generator[LoadedBatch]:
            while futures:
                pending = futures.popleft().result()
                if pending is None:
                    break
                loaded, ready = pending
                current = torch.cuda.current_stream(device)
                current.wait_event(ready)

                def retain(
                    value: torch.Tensor, consumer: torch.cuda.Stream = current
                ) -> torch.Tensor:
                    value.record_stream(consumer)
                    return value

                loaded.map(retain)
                futures.append(executor.submit(prepare))
                yield loaded

        consumer = consume()
        try:
            yield consumer
        finally:
            consumer.close()
            try:
                for future in futures:
                    future.result()
            finally:
                executor.shutdown(wait=True)
                stream.synchronize()


def batches(dataset: NativeBatches) -> DataLoader[LoadedBatch]:
    """Own a bounded worker queue; caller releases the loader when training stops."""
    workers = int(dataset.config["loader_workers"])
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=int(dataset.config["loader_prefetch"]) if workers else None,
        multiprocessing_context="spawn" if workers else None,
        pin_memory=dataset.config.get("device", "cuda") == "cuda",
        generator=torch.Generator().manual_seed(dataset.config["seed"]),
    )
