import hashlib
import json
from pathlib import Path
from typing import Any


def freeze(index: Path, output: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = json.loads(index.read_text())
    split: dict[str, Any] = {
        "version": 1,
        "train": [e["id"] for e in entries if e["split"] == "train"],
        "test": [e["id"] for e in entries if e["split"] == "test"],
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "seed": 4004,
        "mixed": "legacy v004 masks for reference comparison; native K-layer masks also reported",
        "selection": "final fixed-step checkpoints, no best-holdout selection",
        "scale": "oracle per-frame global alignment on eroded single surfaces, including holdout",
    }
    if not split["train"] or not split["test"] or set(split["train"]) & set(split["test"]):
        raise ValueError("A disjoint nonempty train/test split is required")
    if output.exists():
        if json.loads(output.read_text()) != split:
            raise ValueError("Frozen split differs; choose a new versioned protocol path")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(split, indent=2) + "\n")
    return split


def entries_for(index: Path, protocol: Path, split: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(index.read_text())
    frozen: dict[str, Any] = json.loads(protocol.read_text())
    if hashlib.sha256(index.read_bytes()).hexdigest() != frozen["index_sha256"]:
        raise ValueError("Dataset index changed after protocol freeze")
    ids = set(frozen[split])
    selected = [e for e in data if e["id"] in ids]
    if len(selected) != len(ids):
        raise ValueError("Frozen split contains missing or duplicate frames")
    return selected
