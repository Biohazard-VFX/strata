import json
from pathlib import Path
from typing import Any

from strata_training.data.cache import prepare
from strata_training.eval.protocol import freeze
from strata_training.eval.runner import evaluate
from strata_training.model.features import extract
from strata_training.train import train


def pipeline(config_path: Path) -> None:
    config: dict[str, Any] = json.loads(config_path.read_text())
    index = Path(config["index"])
    protocol = Path(config["protocol"])
    prepare(Path(config["dataset"]), index.parent, config["layers"])
    freeze(index, protocol)
    extract(index, Path(config["features"]), config["preset"], int(config.get("clip_length", 1)))
    train(config_path)
    evaluate(
        index,
        protocol,
        Path(config["output"]) / "evaluation",
        [Path(config["output"]) / "final.pt"],
    )
