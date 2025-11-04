from stepgan.seq2seq import Seq2Seq

from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, n_vocabs: int, decoder_output_size: int, hidden_size: int, label_size: int,
                 encoder_seq_len: int, decoder_seq_len: int,
                 lr: float, pre_criterion, encoder_optimizer, decoder_optimizer,
                 device: torch.device):
        super().__init__()
        self.model = Seq2Seq.build_model(n_vocabs, decoder_output_size, hidden_size, label_size,
                                         encoder_seq_len, decoder_seq_len, device)
        self.pre_criterion = pre_criterion
        self.encoder_optimizer = encoder_optimizer(self.model.encoder.parameters(), lr=lr)
        self.decoder_optimizer = decoder_optimizer(self.model.decoder.parameters(), lr=lr)

        self.decoder_seq_len = decoder_seq_len
        self.device = device

    def __sample_from_gen_logits(self, gen_logits: torch.Tensor):
        # generated_tensor = input_tensor.detach().clone()  # (batch_size, seq_len)
        generated_tensor = []
        log_probs = []
        for elem_idx in range(self.decoder_seq_len):
            gen_logit = gen_logits[elem_idx]  # (batch_size, output_size)
            distribution = Categorical(logits=gen_logit)
            generated_id_tensor = distribution.sample()  # (batch_size)

            generated_tensor.append(generated_id_tensor)
            log_probs.append(distribution.log_prob(generated_id_tensor))

        generated_tensor = torch.stack(generated_tensor, dim=1)     # (B, seq_len)
        log_probs = torch.stack(log_probs, dim=1)   # (B, seq_len)

        return generated_tensor, log_probs

    def forward(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, label_tensor: torch.Tensor):
        # (seq_len, batch_size, output_size), (seq_len, batch_size, 1)
        gen_logits, _ = self.model(x_tensor, label_tensor, y_tensor)
        return self.__sample_from_gen_logits(gen_logits)

    def pre_train_model(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, label_tensor: torch.Tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        gen_logits, _ = self.model(x_tensor, label_tensor, y_tensor)
        nll_loss = 0
        _target_tensor = y_tensor.permute(1, 0).type(torch.LongTensor).to(self.device)
        for elem_idx in range(self.decoder_seq_len):
            nll_loss += self.pre_criterion(gen_logits[elem_idx], _target_tensor[elem_idx])
        nll_loss /= x_tensor.shape[0]
        nll_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return nll_loss

    def train_model(self, log_probs: torch.Tensor, q_values: torch.Tensor, baselines: torch.Tensor):
        # (B, seq_len)

        batch_size = log_probs.shape[0]

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        loss = 0
        for batch_idx in range(batch_size):
            weighting_coef = 1.0
            reward = (q_values[batch_idx] - baselines[batch_idx]) * weighting_coef
            loss -= (reward * log_probs[batch_idx]).sum()   # For gradient descent
        loss = loss / batch_size
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss

    def sample_data(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, label_tensor: torch.Tensor):
        with torch.no_grad():
            generated_tensor, log_probs = self.forward(x_tensor, y_tensor, label_tensor)
        return generated_tensor, log_probs
