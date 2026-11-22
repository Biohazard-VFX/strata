import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from strata_dataset.roles import membership


def authored_roles(alpha: Any, ids: Any) -> Any:
    names = {0: "sky", 1: "terrain", 2: "hero_skin", 3: "hero_groom", 4: "hero_hair"}
    values = ids.cpu().numpy() if isinstance(ids, torch.Tensor) else ids
    weights = np.array([membership(names.get(int(i), "layer")) for i in values]).reshape(-1, 3)
    if isinstance(alpha, torch.Tensor):
        return torch.tensor(weights, device=alpha.device, dtype=alpha.dtype) * alpha[:, None]
    return weights.astype(np.float32) * alpha[:, None]


@pytest.fixture
def packed_shot(tmp_path: Path) -> Path:
    shot = tmp_path / "dataset/shot"
    shutil.copytree(Path(os.environ["STRATA_DATASET_FIXTURE"]), shot)
    return shot


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def math_device(request: pytest.FixtureRequest) -> str:
    return str(request.param)
