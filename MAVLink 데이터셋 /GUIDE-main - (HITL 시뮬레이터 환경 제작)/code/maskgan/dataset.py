from maskgan.mask import MASKING, StochasticMask

import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy


class MSGIDSequence(Dataset):
    def __init__(self, data_arr: np.ndarray, label_arr: np.ndarray,
                 mask_builder, can_id_dict, device: torch.device):
        self.sequences = data_arr
        self.labels = label_arr
        assert self.sequences.shape[0] == self.labels.shape[0]
        self.mask_builder = mask_builder
        self.can_id_dict = can_id_dict
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        sequence = self.sequences[idx]
        seq_len = len(sequence)
        mask_indices = self.mask_builder(seq_len)
        src_sequence, tgt_sequence, mask_sequence = self.__prepare_data(sequence, mask_indices)
        label = torch.Tensor(self.labels[idx]).to(self.device)

        return src_sequence, tgt_sequence, mask_sequence, label

    def __prepare_data(self, sequence: np.ndarray, mask_indices: list) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor):
        src_sequence = deepcopy(list(sequence))
        # src_sequence.append(self.can_id_dict.get_eos_idx())

        tgt_sequence = deepcopy(list(sequence))
        # tgt_sequence.insert(0, self.can_id_dict.get_eos_idx())

        # srcs(masked):     I <mask> a <mask>
        # tgts(unmasked):   I am a boy
        # mask:             0, 1, 0, 1
        mask_sequence = torch.zeros(len(tgt_sequence)).to(self.device)   # 0은 마스킹x, 1은 마스킹o
        mask_sym_idx = self.can_id_dict.get_mask_idx()
        for mask_idx in mask_indices:
            if mask_idx == 0:
                continue
            # mask_sequence[mask_idx + 1] = MASKING
            mask_sequence[mask_idx] = MASKING
            src_sequence[mask_idx] = mask_sym_idx

        src_sequence = torch.tensor(src_sequence).to(self.device)
        tgt_sequence = torch.tensor(tgt_sequence).to(self.device)

        return src_sequence, tgt_sequence, mask_sequence


if __name__ == '__main__':
    pass
