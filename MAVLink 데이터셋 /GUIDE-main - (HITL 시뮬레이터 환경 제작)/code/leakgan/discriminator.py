import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, n_vocab: int, max_seq_len: int, embedding_size: int, n_labels: int,
                 lr: float, criterion, optimizer, device):
        super().__init__()
        self.start_token = n_vocab
        self.embedding_size = embedding_size

        self.max_seq_len = max_seq_len
        self.device = device

        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.num_filters = [10, 20, 20, 20, 20, 10, 10, 10, 10, 10, 16, 16]
        assert len(self.filter_sizes) == len(self.num_filters)
        self.total_n_filters = sum(self.num_filters)

        self.embedding = nn.Embedding(n_vocab + 1, embedding_size)

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
        self.highway = Highway(self.total_n_filters)
        self.classification = nn.Linear(sum(self.num_filters) + n_labels, 1)

        self.optimizer = optimizer(self.parameters(), lr=lr)
        self.criterion = criterion

    def forward(self, input_: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # (batch_size, seq_len)
        embedded = self.embedding(input_)
        # (batch_size, seq_len, embedding_size)
        emb = embedded.unsqueeze(1)
        # (batch_size, 1, seq_len, embedding_size)

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
        # (batch_size, total_n_filters)
        feature = nn.Dropout(p=0.25)(output)
        # (batch_size, total_n_filters)

        feature_labels = torch.concat([feature, labels], dim=1)    # (B, total_n_filters + n_labels)
        output = self.classification(feature_labels)
        # (batch_size, 1)

        return output, feature

    def train_model(self, pos_data: torch.Tensor, pos_data_label: torch.Tensor,
                    neg_data: torch.Tensor, neg_data_label: torch.Tensor) -> (float, float):
        self.train()

        self.optimizer.zero_grad()
        pos_label = torch.ones(pos_data.shape[0]).to(self.device)   # 0
        pos_out, _ = self.forward(pos_data, pos_data_label)
        pos_out = pos_out.squeeze(1)    # (B, )
        pos_loss = self.criterion(pos_out, pos_label)     # + self.l2_loss()
        pos_loss.backward()
        self.optimizer.step()
        pos_n_answers = torch.sum((pos_out > 0.5) == (pos_label > 0.5)).data.item()

        self.optimizer.zero_grad()
        neg_label = torch.zeros(neg_data.shape[0]).to(self.device)  # 1
        neg_out, _ = self.forward(neg_data, neg_data_label)
        neg_out = neg_out.squeeze(1)  # (B, )
        neg_loss = self.criterion(neg_out, neg_label)  # + self.l2_loss()
        neg_loss.backward()
        self.optimizer.step()
        neg_n_answers = torch.sum((neg_out > 0.5) == (neg_label > 0.5)).data.item()

        # (Loss, Accuracy)
        return (pos_loss.item() + neg_loss.item()) / 2, \
               (pos_n_answers + neg_n_answers) / (pos_data.shape[0] + neg_data.shape[0])


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
