from scipy.stats import truncnorm
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn.utils import clip_grad_norm_

import numpy as np


def truncated_normal(shape, lower=-0.2, upper=0.2):
    size = 1
    for dim in shape:
        size *= dim
    w_truncated = truncnorm.rvs(lower, upper, size=size)
    w_truncated = torch.from_numpy(w_truncated).float()
    w_truncated = w_truncated.view(shape)
    return w_truncated


class Manager(nn.Module):
    def __init__(self, n_labels: int, batch_size: int, hidden_dim: int, goal_out_size: int):
        super(Manager, self).__init__()
        self.n_labels = n_labels
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size

        self.recurrent_unit = nn.LSTMCell(
            self.goal_out_size + self.n_labels,     # input size
            self.hidden_dim         # hidden size
        )
        self.fc = nn.Linear(
            self.hidden_dim,    # in_features
            self.goal_out_size  # out_features
        )
        self.goal_init = nn.Parameter(torch.zeros(self.batch_size, self.goal_out_size))
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)
        self.goal_init.data = truncated_normal(self.goal_init.data.shape)

    def forward(self, f_t: torch.Tensor, h_m_t: torch.Tensor, c_m_t: torch.Tensor, labels: torch.Tensor):
        feature_labels = torch.concat([f_t, labels], dim=1)    # (B, total_n_filters + n_labels)
        h_m_tp1, c_m_tp1 = self.recurrent_unit(feature_labels, (h_m_t, c_m_t))
        sub_goal = self.fc(h_m_tp1)
        sub_goal = torch.renorm(sub_goal, 2, 0, 1.0)
        # Returns a tensor where each sub-tensor of input along dimension dim is normalized such that the p-norm
        #   of the sub-tensor is lower than the value maxnorm
        return sub_goal, h_m_tp1, c_m_tp1


class Worker(nn.Module):
    def __init__(self, vocab_size, n_labels, embed_dim, hidden_dim, goal_out_size, goal_size):
        super(Worker, self).__init__()
        self.vocab_size = vocab_size
        self.n_labels = n_labels
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size

        self.emb = nn.Embedding(self.vocab_size + 1, self.embed_dim)
        self.recurrent_unit = nn.LSTMCell(self.embed_dim + self.n_labels, self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, self.goal_size * self.vocab_size)
        self.goal_change = nn.Parameter(torch.zeros(self.goal_out_size, self.goal_size))
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            nn.init.normal_(param, std=0.1)

    def forward(self, x_t, h_w_t, c_w_t, labels):
        x_t_emb = self.emb(x_t)     # (B, embed_dim)
        emb_labels = torch.concat([x_t_emb, labels], dim=1)     # (B, embed_dim + n_labels)
        h_w_tp1, c_w_tp1 = self.recurrent_unit(emb_labels, (h_w_t, c_w_t))
        output_tp1 = self.fc(h_w_tp1)
        output_tp1 = output_tp1.view(-1, self.vocab_size, self.goal_size)
        return output_tp1, h_w_tp1, c_w_tp1


class Generator(nn.Module):
    def __init__(self, n_vocabs: int, n_labels: int, batch_size: int, max_seq_len: int,
                 hidden_size: int, total_n_filters: int, goal_size: int, goal_out_size: int, step_size: int,
                 lr: float, criterion, manager_optimizer, worker_optimizer,
                 msg_set: np.ndarray, msg_id_prob_dist: np.ndarray, device):
        super().__init__()

        self.n_vocabs = n_vocabs
        self.n_labels = n_labels
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.total_n_filters = total_n_filters
        self.goal_size = goal_size
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.device = device

        self.manager = Manager(self.n_labels, batch_size, self.hidden_size, self.goal_out_size).to(self.device)
        self.worker = Worker(self.n_vocabs, self.n_labels, self.hidden_size, self.hidden_size,
                             self.goal_out_size, self.goal_size).to(self.device)

        self.criterion = criterion
        self.pre_criterion = nn.CrossEntropyLoss()   # Logit, Target
        self.manager_optimizer = manager_optimizer(self.manager.parameters(), lr=lr)
        self.worker_optimizer = worker_optimizer(self.worker.parameters(), lr=lr)

        # dd
        self.occurred_msg_id_list = msg_set
        self.msg_id_prob_dist = msg_id_prob_dist

    def forward(self, x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, labels, t, temperature=1.0):
        sub_goal, h_m_tp1, c_m_tp1 = self.manager(f_t, h_m_t, c_m_t, labels)
        output, h_w_tp1, c_w_tp1 = self.worker(x_t, h_w_t, c_w_t, labels)
        last_goal_temp = last_goal + sub_goal
        w_t = torch.matmul(real_goal, self.worker.goal_change)
        # (B, goal_size) = (B, goal_out_size) * (goal_out_size, goal_size)
        w_t = torch.renorm(w_t, 2, 0, 1.0)
        w_t = torch.unsqueeze(w_t, -1)
        # (B, goal_size, 1)
        try:
            logits = torch.squeeze(torch.matmul(output, w_t))
        except:
            print('here')
            raise NotImplementedError
        # (B, vocab_size) <- (B, vocab_size, 1) = (B, vocab_size, self.goal_size) * (B, goal_size, 1)
        probs = nn.functional.softmax(temperature * logits, dim=1)
        x_tp1 = Categorical(probs).sample()
        # (B,)
        return x_tp1, h_m_tp1, c_m_tp1, h_w_tp1, c_w_tp1, last_goal_temp, real_goal, sub_goal, probs, t + 1

    def init_hidden(self, batch_size: int) -> (torch.Tensor, torch.Tensor):
        h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(batch_size, self.hidden_size, device=self.device)
        return h, c

    def choose_start_letters(self, batch_size: int):
        start_letter_npy \
            = np.random.choice(self.occurred_msg_id_list, size=batch_size, p=self.msg_id_prob_dist)
        return torch.Tensor(start_letter_npy).type(torch.LongTensor).to(self.device)

    def pre_train_model(self, input_tensor: torch.Tensor, real_goal: torch.Tensor, prediction: torch.Tensor,
                        delta_feature: torch.Tensor) -> (int, int):
        max_norm = 5.0

        # Manager
        mloss = - torch.mean(1.0 - torch.nn.functional.cosine_similarity(real_goal, delta_feature))
        torch.autograd.grad(mloss, self.manager.parameters())
        # mloss.backward()
        clip_grad_norm_(self.manager.parameters(), max_norm=max_norm)
        self.manager_optimizer.step()
        self.manager_optimizer.zero_grad()

        # Worker
        prediction = torch.clamp(prediction, 1e-20, 1.0)
        hot_one = one_hot(input_tensor, self.n_vocabs)
        wloss = - torch.mean(hot_one * torch.log(prediction))
        torch.autograd.grad(wloss, self.worker.parameters())
        # wloss.backward()
        clip_grad_norm_(self.worker.parameters(), max_norm=max_norm)
        self.worker_optimizer.step()
        self.worker_optimizer.zero_grad()

        return mloss.item(), wloss.item()

    def train_model(self, adv_rets, rewards):
        max_norm = 5.0

        real_goal = adv_rets["real_goal"]
        all_goal = adv_rets["all_goal"]
        prediction = adv_rets["prediction"]
        delta_feature = adv_rets["delta_feature"]
        delta_feature_for_worker = adv_rets["delta_feature_for_worker"]
        gen_token = adv_rets["gen_token"]

        # Manager loss
        m_loss = - torch.mean(rewards * (1.0 - torch.nn.functional.cosine_similarity(delta_feature, real_goal,
                                                                                     dim=2)))

        # Worker loss
        intrinsic_rewards = 1.0 - torch.nn.functional.cosine_similarity(all_goal, delta_feature_for_worker, dim=2)
        prediction = torch.clamp(prediction, 1e-20, 1.0)
        w_loss = - torch.mean(
            intrinsic_rewards * torch.sum(one_hot(gen_token, self.n_vocabs) * torch.log(prediction), dim=2))

        torch.autograd.grad(m_loss, self.manager.parameters())  # based on loss improve the parameters
        torch.autograd.grad(w_loss, self.worker.parameters())
        # m_loss.backward(retrain_graph=True)
        # w_loss.backward()
        clip_grad_norm_(self.manager.parameters(), max_norm)
        clip_grad_norm_(self.worker.parameters(), max_norm)
        self.manager_optimizer.step()
        self.worker_optimizer.step()

        return m_loss.item(), w_loss.item()


def one_hot(x: torch.Tensor, vocab_size: int):
    batch_size, seq_len = x.size()
    out = torch.zeros(batch_size * seq_len, vocab_size, device=x.device)

    x = x.contiguous()
    x = x.view(-1, 1)

    if (x.data < vocab_size).all() == 0:
        for i, d in enumerate(x.data):
            if x[i].item() > vocab_size - 1:
                x[i] = 0

    out = out.scatter_(1, x.data, 1.0)
    out = out.view(batch_size, seq_len, vocab_size)

    return out
