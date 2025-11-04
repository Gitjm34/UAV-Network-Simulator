import torch
from torch import nn
from torch.distributions.categorical import Categorical

import numpy as np


class Generator(nn.Module):
    def __init__(self, n_vocabs: int, max_seq_len: int, hidden_size: int,
                 lr: float, criterion, optimizer,
                 msg_set: np.ndarray, msg_id_prob_dist: np.ndarray, device):
        super().__init__()
        self.n_vocabs = n_vocabs
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(self.n_vocabs, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, self.n_vocabs)

        self.criterion = criterion
        self.pre_criterion = nn.CrossEntropyLoss()   # Logit, Target
        self.optimizer = optimizer(self.parameters(), lr=lr)

        self.occurred_msg_id_list = msg_set
        self.msg_id_prob_dist = msg_id_prob_dist

        self.device = device

    def forward(self, input_: torch.Tensor, hidden: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # (batch_size,)
        _t = self.embedding(input_)
        # (batch_size, hidden_size)
        emb = _t.unsqueeze(1)
        # (batch_size, 1, hidden_size)
        emb = emb.permute(1, 0, 2)
        # (1, batch_size, hidden_size)

        """
        # (batch_size, label_size)
        labels_ = labels.unsqueeze(0)
        # (1, batch_size, label_size)
        emb_label = torch.concat([emb, labels_], dim=2)
        """

        output, _hidden = self.gru(emb, hidden)
        # (1, batch_size, hidden_size), (1, batch_size, hidden_size)
        output = self.out(output.squeeze(0))
        # (batch_size, n_vocab)
        return output, _hidden      # (batch_size, n_vocab), (1, batch_size, hidden_size)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

    def choose_start_letters(self, batch_size: int):
        start_letter_npy \
            = np.random.choice(self.occurred_msg_id_list, size=batch_size, p=self.msg_id_prob_dist)
        return torch.Tensor(start_letter_npy).type(torch.LongTensor).to(self.device)

    def generate_samples(self, batch_size: int) -> tuple:
        next_can_id = self.choose_start_letters(batch_size)
        # (batch_size,)
        gen_outputs = torch.zeros(self.max_seq_len, batch_size, device=self.device)
        # (seq_len, batch_size)
        gen_states = torch.zeros(self.max_seq_len, batch_size, self.hidden_size, device=self.device)
        # (seq_len, batch_size, hidden_size)

        gen_hidden = self.init_hidden(batch_size)
        # (1, B, hidden_size)
        for seq_elem_idx in range(self.max_seq_len):
            gen_logit, gen_hidden = self.forward(next_can_id, gen_hidden)
            # (batch_size, n_vocab), (1, batch_size, hidden_size)

            distribution = Categorical(logits=gen_logit)
            generated_id_tensor = distribution.sample()
            # (batch_size, )

            gen_outputs[seq_elem_idx, :] = generated_id_tensor
            gen_states[seq_elem_idx, :, :] = gen_hidden[0]

            next_can_id = generated_id_tensor

        gen_outputs = gen_outputs.type(torch.LongTensor).to(self.device)
        return gen_outputs, gen_states  # (seq_len, batch_size), (seq_len, batch_size, hidden_size)

    def pre_train_model(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):
        batch_size, seq_len = input_tensor.shape
        assert seq_len == (self.max_seq_len - 1)

        self.optimizer.zero_grad()

        nll_loss = 0
        gen_hidden = self.init_hidden(batch_size)
        # (1, B, hidden_size)
        for seq_elem_idx in range(self.max_seq_len - 1):
            inp = input_tensor[:, seq_elem_idx]
            # (B,)
            gen_logit, gen_hidden = self.forward(inp, gen_hidden)
            # (B, n_vocab), (1, B, hidden_size)
            nll_loss += self.pre_criterion(gen_logit, target_tensor[:, seq_elem_idx])

        # (B,)
        # nll_loss = nll_loss.sum()
        # (1,)
        nll_loss /= batch_size
        nll_loss.backward()
        self.optimizer.step()

        return nll_loss

    def train_model(self, samples: torch.Tensor, rewards: torch.Tensor):
        # (seq_len, batch_size), (batch_size, seq_len)
        samples_ = samples.permute(1, 0)
        batch_size, seq_len = samples_.size()

        self.train()
        self.optimizer.zero_grad()

        # Prediction 계산
        gen_hidden = self.init_hidden(batch_size)
        # (1, batch_size, hidden_size)
        next_can_id = self.choose_start_letters(batch_size)
        # (batch_size,)

        predictions = []
        for seq_elem_idx in range(seq_len):
            gen_logit, gen_hidden = self.forward(next_can_id, gen_hidden)
            # (batch_size, n_vocab), (1, batch_size, hidden_size)

            # Sparse Softmax Crossentropy
            cross_entropy = self.criterion(gen_logit, samples_[:, seq_elem_idx])
            # (batch_size, )
            predictions.append(cross_entropy)
            # seq_len * (batch_size, )
            next_can_id = samples_[:, seq_elem_idx]
        predictions = torch.stack(predictions)
        # (seq_len, batch_size)
        predictions = predictions.permute(1, 0)
        # (batch_size, seq_len)

        loss = - (predictions * rewards).sum(dim=1).mean()
        # (1,)

        # Loss 계산
        loss.backward()
        self.optimizer.step()

        return loss
