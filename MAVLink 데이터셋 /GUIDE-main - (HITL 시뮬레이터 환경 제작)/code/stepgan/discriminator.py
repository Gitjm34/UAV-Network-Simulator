from stepgan.seq2seq import Seq2Seq

import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, n_vocabs: int, decoder_output_size: int, hidden_size: int, label_size: int,
                 encoder_seq_len: int, decoder_seq_len: int,
                 lr: float, criterion, encoder_optimizer, decoder_optimizer,
                 device: torch.device, is_discriminator: bool = True):
        super().__init__()
        self.model = Seq2Seq.build_model(n_vocabs, decoder_output_size, hidden_size, label_size,
                                         encoder_seq_len, decoder_seq_len, device)
        self.criterion = criterion
        self.encoder_optimizer = encoder_optimizer(self.model.encoder.parameters(), lr=lr)
        self.decoder_optimizer = decoder_optimizer(self.model.decoder.parameters(), lr=lr)
        self.device = device

        # self.max_seq_len = max_seq_len
        self.is_discriminator = is_discriminator

    def forward(self, x_tensor: torch.Tensor, label_batch: torch.Tensor, y_tensor: torch.Tensor = None):
        # decoder_outputs, decoded_words
        return self.model(x_tensor, label_batch, y_tensor)

    def train_discriminator_model(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor,
                                  label_batch: torch.Tensor, is_real: bool):
        if not self.is_discriminator:
            print(f'Warning: This model is a discriminator')

        self.model.train()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        pred_logits, _ = self.forward(x_tensor, label_batch, y_tensor)
        # (seq_len, batch_size, 1), (seq_len, batch_size, 1)
        pred_logits = pred_logits.squeeze(2)
        # (seq_len, batch_size)

        q_values = nn.Sigmoid()(pred_logits)    # Score 계산
        # (seq_len, batch_size)
        discriminator_score = q_values.mean(dim=0)
        # (batch_size,)

        batch_size = x_tensor.size()[0]

        if is_real:
            truths = torch.ones(batch_size)   # 1로 레이블링
        else:
            truths = torch.zeros(batch_size)  # 0으로 레이블링
        truths = truths.to(self.device)
        # Loss 계산

        loss = self.criterion(discriminator_score, truths)
        loss = loss / batch_size    # Loss 평균
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        n_answers = torch.sum((discriminator_score > 0.5) == (truths > 0.5)).data.item()

        return loss.data.item(), n_answers

    def train_critic_model(self, values: torch.Tensor, q_values: torch.Tensor):
        if self.is_discriminator:
            print(f'Warning: This model is a critic')

        # self.model.train()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        loss = 0
        batch_size = values.size()[0]
        for batch_idx in range(batch_size):
            loss += self.criterion(values[batch_idx], q_values[batch_idx])
        loss = loss / batch_size
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data.item()
