from torch import nn
import torch

# TODO(jerin): Need to fix this as per
# https://github.com/tensorflow/models/blob/master/research/maskgan/model_utils/model_losses.py


class REINFORCE(nn.Module):
    def __init__(self, gamma, clip_value):
        super().__init__()
        self.gamma = gamma
        self.clip_value = clip_value
        self.log_sigmoid = torch.nn.LogSigmoid()

    def forward(self, log_probs, logits, weight, baselines=None):
        # TODO(jerin): How do I assert that this implementation is solid?
        # Is the generator giving the correct rewards?
        batch_size, seqlen = logits.size()
        rewards = self.log_sigmoid(logits)

        cumulative_rewards = []
        for t in range(seqlen):
            cum_value = rewards.new_zeros(batch_size)
            for s in range(t, seqlen):
                exp = float(s-t)
                k = (self.gamma ** exp)
                cum_value += k * weight[:, s] * rewards[:, s]
            cumulative_rewards.append(cum_value)

        cumulative_rewards = torch.stack(cumulative_rewards, dim=1)     # (batch_size, seq_len)

        if baselines is not None:
            advantages = cumulative_rewards - baselines
        else:
            advantages = cumulative_rewards

        # Normalize. Always.  At any stage, this helps prune out bad actions
        # and encourage better ones among the fold.
        advantages = advantages - advantages.mean(dim=0)
        advantages = advantages.clamp(-1*self.clip_value, self.clip_value)

        # advantages = weight*(cumulative_rewards - baselines)

        # Multiply with logprobs
        generator_objective = (advantages * log_probs).sum(dim=1)
        return generator_objective, cumulative_rewards.clone()


class TBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    # def forward(self, pred_logits, truths, weight=None):
    def forward(self, pred_logits, truths):
        # B, T, H = pred_logits.size()
        truths = truths.float()
        # weight = weight.float()
        # _debug(pred_logits, truths, weight)
        loss = self.criterion(pred_logits, truths)
        # missing = weight.sum().item()
        # assert (missing != 0)
        # return ((loss * weight).sum()/missing)
        return loss.sum()


class TCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    # def forward(self, logits, truths, weight=None):
    def forward(self, logits, truths):
        # logits = logits.contiguous()
        # B, T, H = logits.size()
        # logits = logits.view(T*B, H)
        # target = truths.contiguous().view(-1)
        loss = self.criterion(logits, truths)
        return loss.sum()


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    # def forward(self, preds, truths, weights):
    def forward(self, preds, truths):
        truths = truths.float()
        mse_loss = self.criterion(preds, truths)
        # missing = weights.sum()
        # return weights*mse_loss/missing
        return mse_loss.sum()


def _debug(pred_logits, truths, weight):
    B, T, H = pred_logits.size()
    # pred_logits = pred_logits.view(-1, H)
    # truths = truths.view(-1)
    for b in range(B):
        npreds = pred_logits[b, :, :].view(-1)
        ntruths = truths[b, :, :].view(-1)
        nweights = weight[b, :].view(-1)
        weighted = nn.BCEWithLogitsLoss(reduction='none')(npreds, ntruths) * nweights
        outstr = """
sizes: {} {} {}
predns:  {}
truths:  {}
weights: {}
final:   {}
        """.format(npreds.size(), ntruths.size(), nweights.size(),
                torch.sigmoid(npreds).tolist(),
                ntruths.tolist(),
                nweights.tolist(),
                weighted.tolist()
        )
        print(outstr, flush=True)
        break
