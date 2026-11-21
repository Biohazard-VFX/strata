import json
from pathlib import Path

import pytest
import torch

from strata_training.cli import main
from strata_training.io import write_json
from strata_training.torch_api import api


def test_cli_thread_policy_and_frozen_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index, protocol = tmp_path / "index.json", tmp_path / "split.json"
    write_json(index, [{"id": "a", "split": "train"}, {"id": "b", "split": "test"}])
    previous = torch.get_num_threads()
    monkeypatch.setattr(
        "sys.argv", ["strata-train", "--threads", "1", "freeze", str(index), str(protocol)]
    )
    try:
        main()
        assert torch.get_num_threads() == 1
        assert json.loads(capsys.readouterr().out)["test"] == ["b"]
        assert json.loads(protocol.read_text())["train"] == ["a"]
        monkeypatch.setattr(
            "sys.argv", ["strata-train", "--threads", "0", "freeze", str(index), str(protocol)]
        )
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 2
        assert "--threads must be positive" in capsys.readouterr().err
    finally:
        api.set_num_threads(previous)
