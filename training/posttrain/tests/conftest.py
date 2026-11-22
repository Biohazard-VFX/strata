import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def math_device(request: pytest.FixtureRequest) -> str:
    return str(request.param)
