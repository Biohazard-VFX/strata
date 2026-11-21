from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from strata_training.data.dataset import Frame
from strata_training.model.dp2 import DeepDepthPro
from strata_training.model.head import DeepHead
from strata_training.types import Layers


def load_model(path: Path) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    config: dict[str, Any] = checkpoint["config"]
    if config["mode"] == "decoder":
        model: nn.Module = DeepDepthPro(config["preset"], config["layers"], decoder_trainable=False)
    else:
        channels = 32 if config["preset"] == "depthpro2-s" else 128
        model = DeepHead(channels, nn.Identity(), config["layers"])
    model.load_state_dict(checkpoint["model"])
    if isinstance(model, DeepDepthPro):
        model.base.decoder.cuda()
        model.base.head.cuda()
        model.deep_head.cuda()
    else:
        model.cuda()
    return model.eval(), config


def predict(model: nn.Module, config: dict[str, Any], frame: Frame) -> Layers:
    image = torch.as_tensor(frame.arrays["beauty"].astype(np.float32), device="cuda")[None]
    height, width = image.shape[-2:]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if config["mode"] == "decoder":
            pyramid = [
                torch.as_tensor(np.array(p), device="cuda").float()[None] for p in frame.pyramid
            ]
            padded = torch.nn.functional.pad(
                image, (0, -width % 32, 0, -height % 32), mode="replicate"
            )
            scale = torch.tensor(frame.scale, device="cuda")
            output = cast(DeepDepthPro, model).from_pyramid(pyramid, padded, scale)
        else:
            features = torch.as_tensor(np.array(frame.features), device="cuda").float()[None]
            base = torch.as_tensor(frame.base, device="cuda").float()[None, None]
            output = cast(dict[str, torch.Tensor], model(features, image, base))
    values = {k: v[0, ..., :height, :width].float().cpu().numpy() for k, v in output.items()}
    return Layers(values["inverse_depth"], values["alpha"], values["rgb"])
