import random

import numpy as np
import torch
from numpy.typing import NDArray


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def preprocess_numpy_array(x: NDArray, device: torch.device):
    return torch.from_numpy(x).unsqueeze(0).to(torch.float).to(device)


def optional_unsqueeze(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Labels should be like (batches, num_train_datapoints, 1), adding the last dimension if it is missing."""

    if len(y.shape) < len(x.shape):
        y = y.unsqueeze(-1)

    return y
