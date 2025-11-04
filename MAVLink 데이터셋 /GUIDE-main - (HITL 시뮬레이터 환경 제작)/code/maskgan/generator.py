from maskgan.seq2seq import Seq2Seq

from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, n_vocabs: int, decoder_output_size: int, hidden_size: int, label_size: int, max_seq_len: int,
                 lr: float, pre_criterion, criterion, encoder_optimizer, decoder_optimizer,
                 device: torch.device):
        super().__init__()
        self.model = Seq2Seq.build_model(n_vocabs, decoder_output_size, hidden_size, label_size, max_seq_len, device)
        self.pre_criterion = pre_criterion
        self.criterion = criterion
        self.encoder_optimizer = encoder_optimizer(self.model.encoder.parameters(), lr=lr)
        self.decoder_optimizer = decoder_optimizer(self.model.decoder.parameters(), lr=lr)

        self.max_seq_len = max_seq_len
        self.device = device

    def __sample_from_gen_logits(self, input_tensor: torch.Tensor, mask_tensor: torch.Tensor, gen_logits: torch.Tensor):
        generated_tensor = input_tensor.detach().clone()  # (batch_size, seq_len)
        is_masked = mask_tensor.bool()

        log_probs = []
        for elem_idx in range(self.max_seq_len):
            gen_logit = gen_logits[elem_idx]  # (batch_size, output_size)
            distribution = Categorical(logits=gen_logit)
            generated_id_tensor = distribution.sample()  # (batch_size)
            f_sample = torch.where(is_masked[:, elem_idx], generated_id_tensor, generated_tensor[:, elem_idx])
            generated_tensor[:, elem_idx] = f_sample
            log_probs.append(distribution.log_prob(f_sample))
        log_probs = torch.stack(log_probs, dim=1)

        return generated_tensor, log_probs

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, mask_tensor: torch.Tensor,
                label_tensor: torch.Tensor):
        # (seq_len, batch_size, output_size), (seq_len, batch_size, 1)
        gen_logits, _ = self.model(input_tensor, label_tensor, target_tensor)
        return self.__sample_from_gen_logits(input_tensor, mask_tensor, gen_logits)

    def pre_train_model(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, label_tensor: torch.Tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        gen_logits, _ = self.model(input_tensor, label_tensor, target_tensor)
        nll_loss = 0
        _target_tensor = target_tensor.permute(1, 0).type(torch.LongTensor).to(self.device)
        for elem_idx in range(self.max_seq_len):
            nll_loss += self.pre_criterion(gen_logits[elem_idx], _target_tensor[elem_idx])
        nll_loss /= input_tensor.shape[0]
        nll_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return nll_loss

    def train_model(self, log_probs: torch.Tensor, dis_logits: torch.Tensor,
                    mask_tensor: torch.Tensor, baselines: torch.Tensor):
        # self.model.train()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        reward, cumulative_rewards = self.calculate_criterion(log_probs, dis_logits, mask_tensor, baselines)
        loss = -1 * reward
        batch_size = mask_tensor.size()[0]
        loss = loss.sum() / batch_size
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss, cumulative_rewards

    def calculate_criterion(self, log_probs, dis_logits, mask_tensor, baselines):
        return self.criterion(log_probs, dis_logits, mask_tensor, baselines)

    def sample_data(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, mask_tensor: torch.Tensor,
                    label_tensor: torch.Tensor):
        with torch.no_grad():
            generated_tensor, log_probs = self.forward(input_tensor, target_tensor, mask_tensor, label_tensor)
        return generated_tensor, log_probs
