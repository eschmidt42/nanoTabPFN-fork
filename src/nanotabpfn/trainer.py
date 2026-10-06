import time

import schedulefree
import torch
from torch import nn
from torch.utils.data import DataLoader

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.utils import get_default_device


def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    lr: float = 1e-4,
    device: torch.device | None = None,
    steps_per_eval: int = 10,
):
    """
    Trains our model on the given prior using the given criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        lr: (float) learning rate
        device: (torch.device) the device we are using
        steps_per_eval: (int) how many steps we wait before running evaluation again
        eval_func: a function that takes in a classifier and returns a dict containing the average scores
                   for some metrics and datasets

    Returns:
        (model) our trained numpy model
        (list) a list containing our eval history, each entry is the real time used for training so far together
               with a dict mapping metric names to their average values accross a list of datasets
    """
    if not device:
        device = get_default_device()
    model.to(device)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    train_time = 0
    eval_history = []
    try:
        for step, full_data in enumerate(prior):
            step_start_time = time.time()
            train_test_split_index = full_data["train_test_split_index"]
            # if (torch.isnan(data[0]).any() or torch.isnan(data[1]).any()):
            #    continue
            data = (
                full_data["x"].to(device),
                full_data["y"][:, :train_test_split_index].to(device),
            )
            targets = full_data["y"].to(device)

            output = model(data, train_test_split_index=train_test_split_index)
            targets = targets[:, train_test_split_index:]

            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])

            loss = criterion(output, targets).mean()
            loss.backward()
            total_loss = loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step_train_duration = time.time() - step_start_time
            train_time += step_train_duration

            # evaluate
            if step % steps_per_eval == steps_per_eval - 1:
                print(f"time {train_time:7.1f}s | loss {total_loss:7.4f}")

    except KeyboardInterrupt:
        pass

    return model, eval_history
