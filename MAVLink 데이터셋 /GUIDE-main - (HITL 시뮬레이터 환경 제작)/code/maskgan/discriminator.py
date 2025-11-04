from maskgan.seq2seq import Seq2Seq

import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, n_vocabs: int, decoder_output_size: int, hidden_size: int, label_size: int, max_seq_len: int,
                 lr: float, criterion, encoder_optimizer, decoder_optimizer,
                 device: torch.device, is_discriminator: bool = True):
        super().__init__()
        self.model = Seq2Seq.build_model(n_vocabs, decoder_output_size, hidden_size, label_size, max_seq_len, device)
        self.criterion = criterion
        self.encoder_optimizer = encoder_optimizer(self.model.encoder.parameters(), lr=lr)
        self.decoder_optimizer = decoder_optimizer(self.model.decoder.parameters(), lr=lr)

        self.max_seq_len = max_seq_len
        self.is_discriminator = is_discriminator

    def forward(self, input_batch: torch.Tensor, label_batch: torch.Tensor, target_batch: torch.Tensor = None):
        # decoder_outputs, decoded_words
        return self.model(input_batch, label_batch, target_batch)

    def train_discriminator_model(self, input_batch: torch.Tensor, target_batch: torch.Tensor,
                                  mask_batch: torch.Tensor, label_batch: torch.Tensor, is_real: bool):
        if not self.is_discriminator:
            print(f'Warning: This model is a discriminator')

        self.model.train()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        pred_logits, _ = self.forward(input_batch, label_batch, target_batch)
        # (seq_len, batch_size, 1), (seq_len, batch_size, 1)
        pred_logits = pred_logits.squeeze(2)
        # (seq_len, batch_size)
        if is_real:
            truths = torch.ones_like(pred_logits)
        else:
            truths = torch.ones_like(pred_logits) - mask_batch.permute(1, 0)    # Truth가 1이면 real

        loss = 0
        for elem_idx in range(self.max_seq_len):
            loss += self.criterion(pred_logits[elem_idx], truths[elem_idx])

        batch_size = input_batch.size()[0]
        loss = loss / batch_size    # Loss 평균
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        n_answers = torch.sum((pred_logits > 0.0) == (truths > 0.5)).data.item()

        return loss.data.item(), n_answers

    def train_critic_model(self, baselines: torch.Tensor, cumulative_rewards: torch.Tensor):
        if self.is_discriminator:
            print(f'Warning: This model is a critic')

        self.model.train()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        loss = self.criterion(baselines, cumulative_rewards)

        batch_size = baselines.size()[0]
        loss = loss / batch_size
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.data.item()
