from collections import defaultdict
from typing import Any


def clip_indices(entries: list[dict[str, Any]], length: int) -> list[list[int]]:
    if length < 1:
        raise ValueError("Clip length must be positive")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        grouped[entry["shot"]].append(index)
    clips: list[list[int]] = []
    for indices in grouped.values():
        indices.sort(key=lambda i: entries[i]["frame"])
        splits = {entries[i]["split"] for i in indices}
        if len(splits) != 1:
            raise ValueError("A sequence must not cross the train/holdout boundary")
        current: list[int] = []
        for index in indices:
            if current and (
                len(current) == length
                or entries[index]["frame"] != entries[current[-1]]["frame"] + 1
            ):
                clips.append(current)
                current = []
            current.append(index)
        if current:
            clips.append(current)
    return clips
