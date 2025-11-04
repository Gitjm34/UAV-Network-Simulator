import numpy as np
import torch
from torch.utils.data import Dataset

X_LEN = 32
Y_LEN = 96


class MSGIDSequence(Dataset):
    def __init__(self, data_arr: np.ndarray, label_arr: np.ndarray, device: torch.device):
        self.sequences = data_arr
        assert self.sequences.shape[1] == (X_LEN + Y_LEN)
        self.labels = label_arr
        assert self.sequences.shape[0] == self.labels.shape[0]
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        x = self.sequences[idx][:X_LEN]
        x = torch.Tensor(x).type(torch.LongTensor).to(self.device)

        y = self.sequences[idx][X_LEN:]
        y = torch.Tensor(y).type(torch.LongTensor).to(self.device)

        label = torch.Tensor(self.labels[idx]).to(self.device)

        return x, y, label


if __name__ == '__main__':
    pass
