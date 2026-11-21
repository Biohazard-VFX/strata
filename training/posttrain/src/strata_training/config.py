from typing import Any


def validate_config(config: dict[str, Any]) -> None:
    if config["mode"] not in {"head", "decoder"}:
        raise ValueError("mode must be head or decoder")
    if config.get("device", "cuda") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    length = config.get("clip_length", 1)
    if length < 1 or length > config["batch_size"]:
        raise ValueError("Clip length must be positive and not exceed frame batch size")
    crop, halo = config["crop"], config["halo"]
    if crop < 1 or halo < 0 or (crop + 2 * halo) % 32:
        raise ValueError("Native crop plus halo must be positive and aligned to 32 pixels")
