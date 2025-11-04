import torch
from torch import nn


# Ranker
class Discriminator(nn.Module):
    def __init__(self, n_vocab: int, max_seq_len: int, embedding_size: int, lr: float, optimizer, device):
        super().__init__()
        self.embedding_size = embedding_size

        self.max_seq_len = max_seq_len
        self.device = device

        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.num_filters = [10, 20, 20, 20, 20, 10, 10, 10, 10, 10, 16, 16]
        assert len(self.filter_sizes) == len(self.num_filters)
        total_n_filters = sum(self.num_filters)

        self.embedding = nn.Embedding(n_vocab, embedding_size)

        # in_channels = [1] + self.num_filters[:-1]
        self.conv2d_list = []
        self.pool2d_list = []
        in_channel = 1
        for filter_size, n_filters in zip(self.filter_sizes, self.num_filters):
            assert self.max_seq_len - filter_size + 1 > 0

            self.conv2d_list.append(nn.Conv2d(in_channel, n_filters,
                                              kernel_size=(filter_size, embedding_size),
                                              stride=(1, 1), padding='valid').to(self.device))
            self.pool2d_list.append(nn.MaxPool2d(kernel_size=(self.max_seq_len-filter_size+1, 1),
                                                 stride=(1, 1)).to(self.device))
        self.highway = Highway(total_n_filters)

        self.gamma = 0.8

        self.optimizer = optimizer(self.parameters(), lr=lr)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # seq_len = input_.shape[1]

        # (batch_size, seq_len)
        embedded = self.embedding(input_)
        # (batch_size, seq_len, embedding_size)
        emb = embedded.unsqueeze(1)
        # (batch_size, 1, seq_len, embedding_size)

        """
        # (B, label_size)
        labels_ = labels.repeat(seq_len, 1, 1)
        # (seq_len, B, label_size)
        labels_ = labels_.permute(1, 0, 2)
        # (B, seq_len, label_size)
        labels_ = labels_.unsqueeze(1)
        # (B, 1, seq_len, label_size)
        emb_label = torch.concat([emb, labels_], dim=3)
        # (B, 1, seq_len, embedding_size + label_size)
        """

        pool_list = []
        for idx in range(len(self.conv2d_list)):
            conv_out = self.conv2d_list[idx](emb)
            pool_out = self.pool2d_list[idx](conv_out)
            pool_list.append(pool_out)

        concat_pool = torch.cat(pool_list, dim=1)
        # (batch_size, total_n_filters, 1, 1)
        concat_pool = concat_pool.squeeze()
        # (batch_size, total_n_filters)

        output = self.highway(concat_pool)
        # (batch_size, n_filters)
        output = nn.Dropout(p=0.25)(output)     # TODO: Dropout ratio
        # (batch_size, n_filters)

        return output

    def train_model(self, input_batch: torch.Tensor, label_tensor: torch.Tensor, ref_batch: torch.Tensor):
        self.train()

        self.optimizer.zero_grad()

        target_output = self.forward(input_batch)
        ref_output = self.forward(ref_batch)
        log_rank_scores = self.calculate_rank_score(target_output, ref_output)
        # (B, )

        pos_ind = label_tensor     # (B, )
        neg_ind = (~(label_tensor.type(torch.bool))).type(torch.int)   # (B, )

        pos_loss = (log_rank_scores * pos_ind).sum() / pos_ind.sum()
        neg_loss = (log_rank_scores * neg_ind).sum() / neg_ind.sum()

        loss = - (pos_loss - neg_loss)  # TODO: Ref size로 나누어야 하나? 확인 필요
        loss.backward()
        self.optimizer.step()

        return loss

    def calculate_rank_score(self, target_output: torch.Tensor, ref_output: torch.Tensor) -> torch.Tensor:
        target_output_ = torch.nn.functional.normalize(target_output)
        ref_output_ = torch.nn.functional.normalize(ref_output)
        scores = torch.matmul(target_output_, ref_output_.transpose(0, 1))  # (B, ref_batch)
        scores = self.gamma * scores    # (B, ref_batch)
        scores = torch.nn.LogSoftmax(dim=0)(scores)   # (B, ref_batch)
        scores = torch.mean(scores, dim=1)  # (B, )
        return scores   # (B, )

    """
    def calculate_rank_score__(self, target_output: torch.Tensor, ref_output: torch.Tensor) -> torch.Tensor:
        scores = self.calculate_score(target_output, ref_output)
        log_rank_scores = torch.nn.LogSoftmax(dim=0)(scores)
        return log_rank_scores  # (target_batch, )
    """

    def calculate_rollout_rank_score(self, target_tensor: torch.Tensor, ref_tensor: torch.Tensor,
                                     n_rollouts: int, seq_len: int):
        batch_size = target_tensor.shape[2]
        ref_batch_size = ref_tensor.shape[0]
        # target_tensor: (n_rollouts, seq_len, batch_size, seq_len)
        # ref_tensor: (r_batch_size, seq_len)
        target_tensor_ = target_tensor.reshape(-1, seq_len)
        # (n_rollouts * seq_len * batch_size, seq_len)

        target_output = self.forward(target_tensor_)
        # (n_rollouts * seq_len * batch_size, n_filters)
        ref_output = self.forward(ref_tensor)
        # (r_batch_size, n_filters)

        # scores = self.calculate_score(target_output, ref_output)
        target_output_ = torch.nn.functional.normalize(target_output)
        ref_output_ = torch.nn.functional.normalize(ref_output)
        scores = torch.matmul(target_output_, ref_output_.transpose(0, 1))
        # (n_rollouts * seq_len * batch_size, ref_batch_size)
        scores = self.gamma * scores
        # (n_rollouts * seq_len * batch_size, ref_batch_size)
        scores = scores.reshape(n_rollouts * seq_len, batch_size, ref_batch_size)
        # (n_rollouts * seq_len, batch_size, ref_batch_size)
        scores = torch.nn.LogSoftmax(dim=1)(scores)
        # (n_rollouts * seq_len, batch_size, ref_batch_size)
        scores = torch.mean(scores, dim=2)
        # (n_rollouts * seq_len, batch_size)
        rollout_rank_scores = scores.reshape(n_rollouts, seq_len, batch_size)
        # (n_rollouts, seq_len, batch_size)
        return rollout_rank_scores  # (n_rollouts, seq_len, batch_size)


class Highway(nn.Module):
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.transform = nn.Linear(input_size, input_size)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input_):
        proj_result = nn.functional.relu(self.proj(input_))
        proj_gate = torch.sigmoid(self.transform(input_))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input_)
        return gated


if __name__ == '__main__':
    pass
