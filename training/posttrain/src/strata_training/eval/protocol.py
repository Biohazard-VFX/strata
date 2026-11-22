import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from strata_training.data.policy import require_supervision
from strata_training.io import write_json


def freeze(index: Path, output: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = json.loads(index.read_text())
    split: dict[str, Any] = {
        "version": 1,
        "train": [e["id"] for e in entries if e["split"] == "train"],
        "test": [e["id"] for e in entries if e["split"] == "test"],
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "seed": 4004,
        "mixed": "native K-layer mixed-coverage masks",
        "selection": "best OOD-validation mixed-alpha checkpoint; step budget is a ceiling",
        "scale": "oracle per-frame global alignment on eroded single surfaces, including holdout",
    }
    if not split["train"] or not split["test"] or set(split["train"]) & set(split["test"]):
        raise ValueError("A disjoint nonempty train/test split is required")
    if output.exists():
        if json.loads(output.read_text()) != split:
            raise ValueError("Frozen split differs; choose a new versioned protocol path")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, split)
    return split


def entries_for(
    index: Path,
    protocol: Path,
    split: str,
    *,
    purpose: Literal["training", "validation", "test"] | None = None,
) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(index.read_text())
    frozen: dict[str, Any] = json.loads(protocol.read_text())
    if hashlib.sha256(index.read_bytes()).hexdigest() != frozen["index_sha256"]:
        raise ValueError("Dataset index changed after protocol freeze")
    ids = set(frozen[split])
    selected = [e for e in data if e["id"] in ids]
    if len(selected) != len(ids):
        raise ValueError("Frozen split contains missing or duplicate frames")
    use = purpose or ("training" if split == "train" else "test")
    if use != "test":
        require_supervision(selected, purpose=use, root=index.parent)
    return selected
