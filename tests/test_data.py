from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from nanotabpfn import data
from nanotabpfn.data import PriorDumpDataLoader


@pytest.fixture
def prior_dump_file(tmp_path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    filename = tmp_path / "prior_dump.h5"
    features = np.arange(5 * 4 * 3, dtype=np.float32).reshape(5, 4, 3)
    targets = np.arange(5 * 4, dtype=np.int64).reshape(5, 4)

    with h5py.File(filename, "w") as dump:
        dump["max_num_classes"] = np.array([3])
        dump["num_features"] = np.array([2, 3, 1, 2, 3])
        dump["num_datapoints"] = np.array([3, 2, 4, 3, 2])
        dump["single_eval_pos"] = np.array([1, 1, 2, 2, 1])
        dump["X"] = features
        dump["y"] = targets

    return str(filename), features, targets


def test_prior_dump_data_loader_initializes_from_dump_and_defaults_device(
    prior_dump_file: tuple[str, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filename, _, _ = prior_dump_file
    monkeypatch.setattr(data, "get_default_device", lambda: "cpu")

    loader = PriorDumpDataLoader(filename, num_steps=4, batch_size=2)

    assert loader.device == "cpu"
    assert loader.pointer == 0
    assert loader.max_num_classes == 3
    assert len(loader) == 4


def test_prior_dump_data_loader_batches_data_and_resets_pointer(
    prior_dump_file: tuple[str, np.ndarray, np.ndarray],
) -> None:
    filename, features, targets = prior_dump_file
    loader = PriorDumpDataLoader(filename, num_steps=4, batch_size=2, device="cpu")

    batches = list(loader)

    assert len(batches) == 4
    expected: list[tuple[np.ndarray, np.ndarray, int]] = [
        (features[0:2, :3, :3], targets[0:2, :3], 1),
        (features[2:4, :4, :2], targets[2:4, :4], 2),
        (features[4:5, :2, :3], targets[4:5, :2], 1),
        (features[0:2, :3, :3], targets[0:2, :3], 1),
    ]

    for batch, (expected_features, expected_targets, split_index) in zip(
        batches, expected, strict=True
    ):
        torch.testing.assert_close(batch["x"], torch.from_numpy(expected_features))
        torch.testing.assert_close(batch["y"], torch.from_numpy(expected_targets))
        assert batch["train_test_split_index"] == split_index

    assert loader.pointer == 2
