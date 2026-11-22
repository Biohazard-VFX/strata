from typing import Any


def training_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Return a new mapping; nested settings are borrowed and never mutated."""
    result = {
        "input_mode": "hybrid",
        "clean_fraction": 0.4,
        "crop": 512,
        "halo": 0,
        "compile_encoder": False,
        **config,
    }
    return result


def validate_config(config: dict[str, Any]) -> None:
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
