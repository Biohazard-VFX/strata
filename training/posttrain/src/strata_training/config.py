from typing import Any


def training_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Return a new mapping; nested settings are borrowed and never mutated."""
    result = {
        "input_mode": "hybrid",
        "clean_fraction": 0.4,
        "crop": 512,
        "halo": 0,
        "compile_encoder": False,
        "loader_workers": 0,
        "loader_prefetch": 2,
        "early_stopping": True,
        "validation_every": 250,
        "validation_patience": 4,
        **config,
    }
    return result


def validate_config(config: dict[str, Any]) -> None:
    for key, minimum, default in (("loader_workers", 0, 0), ("loader_prefetch", 1, 2)):
        value = config.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if config.get("early_stopping", True):
        for key, default in (("validation_every", 250), ("validation_patience", 4)):
            value = config.get(key, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
    mode = config.get("input_mode", "hybrid")
    if mode not in {"cached", "hybrid"}:
        raise ValueError("input_mode must be cached or hybrid")
    if mode == "hybrid":
        if not config.get("samples"):
            raise ValueError("Hybrid training requires a denoised raw-sample cache: samples")
        if config.get("scale_mode", "raw") != "raw" or config.get("clip_length", 1) != 1:
            raise ValueError("Hybrid native crops require raw scale and independent single frames")
        if not 0 <= config.get("clean_fraction", 0.4) <= 1:
            raise ValueError("clean_fraction must be in [0, 1]")
    if config["mode"] not in {"head", "decoder"}:
        raise ValueError("mode must be head or decoder")
    if config.get("device", "cuda") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config["learning_rate"] <= 0:
        raise ValueError("Head learning rate must be positive")
    if config["mode"] == "decoder" and config["decoder_learning_rate"] <= 0:
        raise ValueError("Decoder learning rate must be positive")
    if not 0 <= config.get("warmup_steps", 0) < config["steps"]:
        raise ValueError("Warmup must be nonnegative and shorter than training")
    length = config.get("clip_length", 1)
    if length < 1 or length > config["batch_size"]:
        raise ValueError("Clip length must be positive and not exceed frame batch size")
    crop, halo = config["crop"], config["halo"]
    if crop < 1 or halo < 0 or (crop + 2 * halo) % 32:
        raise ValueError("Native crop plus halo must be positive and aligned to 32 pixels")
