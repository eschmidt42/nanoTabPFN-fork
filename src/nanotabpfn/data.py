import h5py
import torch
from torch.utils.data import DataLoader

from nanotabpfn.utils import get_default_device


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    filename: str
    num_steps: int
    pointer: int
    batch_size: int
    device: str
    max_num_classes: int

    def __init__(self, filename: str, num_steps: int, batch_size: int, device: str | None = None):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        if device is None:
            device = get_default_device()
        self.device = device
        self.pointer = 0

        with h5py.File(self.filename, "r") as f:
            self.max_num_classes = f["max_num_classes"][0]

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                end = self.pointer + self.batch_size
                num_features = f["num_features"][self.pointer : end].max()
                num_datapoints_batch = f["num_datapoints"][self.pointer : end]
                max_seq_in_batch = int(num_datapoints_batch.max())
                x = torch.from_numpy(f["X"][self.pointer : end, :max_seq_in_batch, :num_features])
                y = torch.from_numpy(f["y"][self.pointer : end, :max_seq_in_batch])
                train_test_split_index = f["single_eval_pos"][self.pointer : end]

                self.pointer += self.batch_size
                if self.pointer >= f["X"].shape[0]:
                    print("""Finished iteration over all stored datasets! """)
                    self.pointer = 0

                yield {
                    "x": x.to(self.device),
                    "y": y.to(self.device),
                    "train_test_split_index": train_test_split_index[0].item(),
                }

    def __len__(self):
        return self.num_steps
