import torch
from torch.utils.data import Dataset
import numpy as np


class MSGIDSequence(Dataset):
    def __init__(self, data_arr: np.ndarray, label_arr: np.ndarray, device: torch.device):
        self.sequences = data_arr
        self.labels = label_arr
        assert self.sequences.shape[0] == self.labels.shape[0]

        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.Tensor(self.sequences[idx]).type(torch.LongTensor).to(self.device), \
               torch.Tensor(self.labels[idx]).type(torch.LongTensor).to(self.device)
