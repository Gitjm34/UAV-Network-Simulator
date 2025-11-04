import torch
import torch.nn as nn

SOS_token = 0
EOS_token = 1


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, label_size: int):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size + label_size, self.hidden_size)

    def forward(self, input_: torch.Tensor, hidden: torch.Tensor, labels: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor):
        # (batch_size, 1)
        _t = self.embedding(input_)
        # (batch_size, hidden_size)
        emb = _t.unsqueeze(1)
        # (batch_size, 1, hidden_size)
        emb = emb.permute(1, 0, 2)
        # (1, batch_size, hidden_size)

        # (batch_size, label_size)
        labels_ = labels.unsqueeze(0)
        # (1, batch_size, label_size)
        emb_label = torch.concat([emb, labels_], dim=2)

        output, hidden = self.gru(emb_label, hidden)
        # (1, batch_size, hidden_size), (1, batch_size, hidden_size)
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, hidden):
        output = self.embedding(input_).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, label_size: int, output_size: int,
                 dropout_p: float, max_length: int):
        super(AttnDecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size + label_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_: torch.Tensor, hidden: torch.Tensor, labels: torch.Tensor, encoder_outputs: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor):
        # (batch_size, 1)
        embedded = self.embedding(input_).unsqueeze(1)
        # (batch_size, 1, hidden_size)
        embedded = self.dropout(embedded)
        # (batch_size, 1, hidden_size)
        embedded = embedded.permute(1, 0, 2)
        # (1, batch_size, hidden_size)

        attn_weights = torch.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # (batch_size, hidden_size * 2) -> (batch_size, max_length)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute(1, 0, 2))
        # (batch_size, 1, hidden_size)
        attn_applied = attn_applied.permute(1, 0, 2)
        # (1, batch_size, hidden_size)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # (batch_size, hidden_size * 2)
        output = self.attn_combine(output).unsqueeze(0)
        # (batch_size, hidden_size) -> (1, batch_size, hidden_size)

        output = torch.relu(output)

        # (batch_size, label_size)
        labels_ = labels.unsqueeze(0)
        # (1, batch_size, label_size)
        concat_label = torch.concat([output, labels_], dim=2)

        output, hidden = self.gru(concat_label, hidden)
        # (1, batch_size, hidden_size), (1, batch_size, hidden_size)

        """
        if self.output_size == 1:
            output = torch.sigmoid(self.out(output[0]))
        else:
            output = torch.softmax(self.out(output[0]), dim=1)
        """
        output = self.out(output.squeeze(0))    # (batch_size, output_size)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderRNN, decoder: AttnDecoderRNN, device: torch.device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    @classmethod
    def build_model(cls, n_vocabs: int, decoder_output_size: int, hidden_size: int, label_size: int, max_seq_len: int,
                    device: torch.device):
        encoder = EncoderRNN(n_vocabs, hidden_size, label_size).to(device)
        decoder = AttnDecoderRNN(n_vocabs, hidden_size, label_size, decoder_output_size, 0.25, max_seq_len).to(device)

        return cls(encoder, decoder, device)

    def forward(self, input_batch: torch.Tensor, label_batch: torch.Tensor, target_batch: torch.Tensor = None):
        max_seq_len = self.decoder.max_length
        batch_size = input_batch.size(0)

        encoder_outputs = torch.zeros(max_seq_len, batch_size, self.encoder.hidden_size, device=self.device)
        encoder_hidden = self.encoder.init_hidden(batch_size, self.device)

        for seq_elem_idx in range(max_seq_len):
            encoder_output, encoder_hidden = self.encoder(input_batch[:, seq_elem_idx], encoder_hidden, label_batch)
            # (1, batch_size, hidden_size), (1, batch_size, hidden_size)
            encoder_outputs[seq_elem_idx, :, :] = encoder_output[0, :, :]

        # decoder_input = torch.tensor([[SOS_token]] * batch_size, device=self.device)
        decoder_input = input_batch[:, 0]
        decoder_hidden = encoder_hidden

        decoder_outputs, decoded_words = [], []
        for seq_elem_idx in range(max_seq_len):
            decoder_output, decoder_hidden, _ \
                = self.decoder(decoder_input, decoder_hidden, label_batch, encoder_outputs)

            # decoder_output: (batch_size, output_size)
            decoder_outputs.append(decoder_output)

            _, top_i = decoder_output.topk(1)
            # top_i: (batch_size, 1)
            decoded_words.append(top_i.cpu().detach())

            if seq_elem_idx >= max_seq_len - 1:  # Index range 에러를 막기 위해 강제로 break
                break

            if target_batch is not None:
                decoder_input = target_batch[:, seq_elem_idx+1]
            else:
                decoder_input = top_i.squeeze().detach()
            # decoder_input = decoder_input.view(batch_size, 1)

        decoder_outputs = torch.stack(decoder_outputs)
        decoded_words = torch.stack(decoded_words)
        # (seq_len, batch_size, output_size), (seq_len, batch_size, 1)
        return decoder_outputs, decoded_words


if __name__ == '__main__':
    pass
