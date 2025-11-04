import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, n_labels, gpu=False, oracle_init=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim + n_labels, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp, hidden, labels):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim

        labels_ = labels.unsqueeze(0)                           # (1, B, n_labels)
        emb_label = torch.concat([emb, labels_], 2)             # (1, B, embedding_dim + n_labels)

        out, hidden = self.gru(emb_label, hidden)               # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, num_samples, start_letters, labels, length=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """
        assert length >= 0

        if length == 0:
            sample_length = self.max_seq_len
        else:
            sample_length = length

        samples = torch.zeros(num_samples, sample_length).type(torch.LongTensor)
        samples_softmax = torch.zeros(num_samples, sample_length, self.vocab_size).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor(start_letters))
        if self.gpu:
            samples = samples.cuda()
            samples_softmax = samples_softmax.cuda()
            inp = inp.cuda()

        for i in range(sample_length):
            out, h = self.forward(inp, h, labels)           # out: num_samples x vocab_size

            samples_softmax[:, i, :] = torch.exp(out).view(-1, self.vocab_size).data
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples, samples_softmax

    def batchNLLLoss(self, inp, target, labels):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h, labels)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, inp, target, reward, labels):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h, labels)
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # -log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size


class Discriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, n_labels, gpu=False, dropout=0.1):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.gpu = gpu

        self.gru_n_layers = 2
        self.gru_bidirectional = True

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.gru = nn.GRU(embedding_dim + n_labels, hidden_dim, num_layers=2, bidirectional=True)
        self.gru = nn.GRU(embedding_dim + n_labels, hidden_dim,
                          num_layers=self.gru_n_layers, bidirectional=self.gru_bidirectional)
        # self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.gru2hidden = nn.Linear(self.gru_n_layers * (1 + self.gru_bidirectional) * hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        # h = autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim))
        h = autograd.Variable(torch.zeros(self.gru_n_layers * (1 + self.gru_bidirectional) * 1,
                                          batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp, hidden, labels, sig):
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(inp)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim

        seq_len = emb.shape[0]
        labels_ = labels.repeat(seq_len, 1, 1)                   # (seq_len, B, n_labels)
        emb_label = torch.concat([emb, labels_], 2)             # (seq_len, B, embedding_dim + n_labels)

        _, hidden = self.gru(emb_label, hidden)                    # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, self.gru_n_layers * (1 + self.gru_bidirectional) * self.hidden_dim))
        # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        if sig:
            out = torch.sigmoid(out)
        return out

    def batchClassify(self, inp, labels, sig):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h, labels, sig)
        return out.view(-1)

    def batchBCELoss(self, inp, target, labels, sig):
        """
        Returns Binary Cross Entropy Loss for discriminator.

         Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size (binary 1/0)
        """

        loss_fn = nn.BCELoss()
        h = self.init_hidden(inp.size()[0])
        out = self.forward(inp, h, labels, sig)
        return loss_fn(out, target)

    def restore(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
