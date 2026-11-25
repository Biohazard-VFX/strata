import json
from pathlib import Path

from strata_training.config import training_defaults, validate_config


def test_shipped_diagnostic():
    path = Path(__file__).resolve().parents[1] / "configs/diagnostic-400.json"
    shipped = json.loads(path.read_text())
    config = training_defaults(shipped)
    validate_config(config)
    assert (config["preset"], config["mode"], config["layers"]) == ("depthpro2-h", "head", 4)
    keys = ("steps", "warmup_steps", "batch_size", "crop", "halo")
    assert tuple(config[key] for key in keys) == (400, 20, 1, 64, 48)
    assert config["input_mode"] == "cached"
    assert not config["early_stopping"]
    assert not any(key.startswith("validation_") for key in shipped)
    paths = ("index", "protocol", "features", "output")
    assert all(not Path(config[key]).is_absolute() for key in paths)
