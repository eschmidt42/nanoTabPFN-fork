import random

import numpy as np
import pytest
import torch

from nanotabpfn.utils import get_default_device, set_randomness_seed


def test_set_randomness_seed_repeats_python_numpy_and_torch():
    seed = 123

    set_randomness_seed(seed)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    set_randomness_seed(seed)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert second == first


@pytest.mark.parametrize(
    ("mps_available", "cuda_available", "expected"),
    [
        (False, False, "cpu"),
        (True, False, "mps"),
        (False, True, "cuda"),
        (True, True, "cuda"),
    ],
)
def test_get_default_device(
    mps_available: bool,
    cuda_available: bool,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    assert get_default_device() == expected
