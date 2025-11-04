from rankgan.generator import Generator
from rankgan.discriminator import Discriminator

import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

import os
import numpy as np
import datetime
import pickle


class RankGAN:
    def __init__(self, train_dataset: Dataset, n_vocabs: int, batch_size: int, ref_size: int,
                 gen_hidden_size: int, ran_hidden_size: int, n_rollouts: int, max_seq_len: int,
                 gen_lr: float, ran_lr: float, save_path: str,
                 msg_set: np.ndarray, msg_id_prob_dist: np.ndarray, device: torch.device):
        assert max_seq_len > 0

        self.n_vocabs = n_vocabs
        self.n_rollouts = n_rollouts
        self.max_seq_len = max_seq_len

        self.save_path = save_path
        self.device = device

        generator_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        generator_optimizer = torch.optim.Adam
        ranker_optimizer = torch.optim.Adam

        self.generator \
            = Generator(n_vocabs, self.max_seq_len, gen_hidden_size,
                        gen_lr, generator_criterion, generator_optimizer,
                        msg_set, msg_id_prob_dist, self.device).to(self.device)
        self.rollout \
            = Generator(n_vocabs, self.max_seq_len, gen_hidden_size,
                        gen_lr, generator_criterion, generator_optimizer,
                        msg_set, msg_id_prob_dist, self.device).to(self.device)
        self.ranker \
            = Discriminator(n_vocabs, self.max_seq_len, ran_hidden_size,
                            ran_lr, ranker_optimizer, self.device).to(self.device)

        self.train_dataloader = RankGAN.__build_dataloader(train_dataset, batch_size)
        self.ref_dataloader = RankGAN.__build_dataloader(train_dataset, ref_size)
        self.last_generator_save_path = ''

    @staticmethod
    def __build_dataloader(data: Dataset, batch_size: int) -> DataLoader:
        return DataLoader(data, batch_size=batch_size, shuffle=True)

    def rollout_generator(self, gen_outputs: torch.Tensor, gen_states: torch.Tensor, n_rollouts: int) \
            -> torch.Tensor:
        self.rollout.train()

        # Rollout
        rollout_results = []
        for rollout_idx in range(n_rollouts):
            rollout_list = []
            for rollout_step in range(1, self.max_seq_len):
                rollout_seqs = []

                for i in range(rollout_step):
                    rollout_seqs.append(gen_outputs[i, :])
                # rollout_step * (batch_size, )

                gen_input = gen_outputs[rollout_step - 1]     # (batch_size, )
                gen_hidden = gen_states[rollout_step - 1].unsqueeze(0)   # (1, batch_size, hidden_size)
                for i in range(self.max_seq_len - rollout_step):
                    gen_logit, gen_hidden = self.rollout(gen_input, gen_hidden)
                    distribution = Categorical(logits=gen_logit)
                    gen_input = distribution.sample()  # (batch_size, )
                    rollout_seqs.append(gen_input)

                rollout_seqs = torch.stack(rollout_seqs)    # (seq_len, batch_size)
                rollout_list.append(rollout_seqs)
            rollout_list.append(gen_outputs)
            # (seq_len) * (seq_len, batch_size)
            rollout_list = torch.stack(rollout_list)
            # (seq_len, seq_len, batch_size)
            rollout_results.append(rollout_list)
        rollout_results = torch.stack(rollout_results)
        # (n_rollouts, seq_len, seq_len, batch_size)
        rollout_results = rollout_results.permute(0, 1, 3, 2)
        # (n_rollouts, seq_len, batch_size, seq_len)
        return rollout_results

    def get_reward(self, rollout_results: torch.Tensor, ref_tensor: torch.Tensor):
        # rollout_results: (n_rollouts, seq_len, batch_size, seq_len)
        rollout_rank_scores \
            = self.ranker.calculate_rollout_rank_score(rollout_results, ref_tensor, self.n_rollouts, self.max_seq_len)
        # (n_rollouts, seq_len, batch_size)
        rewards = torch.mean(rollout_rank_scores, dim=0)
        # (seq_len, batch_size)
        rewards = rewards.permute(1, 0)
        # (batch_size, seq_len)
        return rewards

    def train(self, gen_pre_epochs: int, dis_pre_epochs: int, adv_epochs: int) -> None:
        print_period = 100
        loss_acc_dict = {'pre_gen_loss': [], 'pre_r_loss': [],
                         'adv_gen_loss': [], 'adv_r_loss': [], 'adv_reward': []}

        # Pretraining
        for epoch in range(gen_pre_epochs):
            batch_counter = 0
            avg_gen_nll_loss = 0
            # Generator
            for input_tensor, _ in self.train_dataloader:
                gen_nll_loss = self.pre_train_generator(input_tensor)

                batch_counter += 1
                avg_gen_nll_loss += gen_nll_loss

                if batch_counter % print_period == 0:
                    print(f'[Pre]Gen NLL loss: {gen_nll_loss}')

            avg_gen_nll_loss /= batch_counter
            self.save_generator(f'pre_trained_gen_all_{epoch}')
            loss_acc_dict['pre_gen_loss'].append(avg_gen_nll_loss)
            print(f'[Pre][{epoch}]Gen NLL loss: {avg_gen_nll_loss} at {datetime.datetime.now()}\n')

        for epoch in range(dis_pre_epochs):
            batch_counter = 0
            avg_r_loss = 0
            # Ranker
            ref_iterator = iter(self.ref_dataloader)
            for input_tensor, label_tensor in self.train_dataloader:
                label_tensor_ = label_tensor.squeeze()
                ref_tensor, _ = next(ref_iterator)
                r_loss = self.train_ranker(input_tensor, label_tensor_, ref_tensor)

                batch_counter += 1
                avg_r_loss += r_loss

                if batch_counter % print_period == 0:
                    print(f'[Pre]Ranker loss: {r_loss}')

            avg_r_loss /= batch_counter
            self.save_ranker(f'pre_trained_ran_all_{epoch}')
            loss_acc_dict['pre_r_loss'].append(avg_r_loss)
            print(f'[Pre][{epoch}]Ranker loss: {avg_r_loss} at {datetime.datetime.now()}\n')

        # Rollout load from generator
        self.sync_rollout_to_generator()

        # Adversarial training
        for epoch in range(adv_epochs):
            batch_counter = 0
            avg_r_loss = 0
            avg_gen_loss = 0
            avg_reward = 0

            rollout_results = None
            gen_samples = None
            ref_iterator = iter(self.ref_dataloader)
            for input_tensor, label_tensor in self.train_dataloader:
                label_tensor_ = label_tensor.squeeze()
                ref_tensor, _ = next(ref_iterator)

                if batch_counter % 30 == 0 or rollout_results is None:
                    self.save_generator(f'gen_tmp')
                    self.sync_rollout_to_generator()
                    g_loss, reward_mean, gen_samples, rollout_results \
                        = self.train_generator(ref_tensor, None, None)
                else:
                    g_loss, reward_mean, gen_samples, rollout_results \
                        = self.train_generator(ref_tensor, gen_samples, rollout_results)
                r_loss = self.train_ranker(input_tensor, label_tensor_, ref_tensor)

                batch_counter += 1
                avg_r_loss += r_loss
                avg_gen_loss += g_loss
                avg_reward += reward_mean

                if batch_counter % print_period == 0:
                    print(f'[Adv]Gen loss: {g_loss}')
                    print(f'[Adv]Ranker loss: {r_loss}')

            avg_r_loss /= batch_counter
            avg_gen_loss /= batch_counter
            avg_reward /= batch_counter

            loss_acc_dict['adv_r_loss'].append(avg_r_loss)
            loss_acc_dict['adv_gen_loss'].append(avg_gen_loss)
            loss_acc_dict['adv_reward'].append(avg_reward)

            self.save_generator(f'adv_trained_gen_all_{epoch}')
            self.save_ranker(f'adv_trained_ran_all_{epoch}')

            print(f'[Adv][{epoch}]Ranker loss: {avg_r_loss}, Gen loss: {avg_gen_loss}, '
                  f'Reward: {avg_reward} at {datetime.datetime.now()}\n')

        loss_acc_path = os.path.join(self.save_path, 'loss_acc_dict(rankgan).pickle')
        with open(loss_acc_path, 'wb') as pickle_file:
            pickle.dump(loss_acc_dict, pickle_file)

    def pre_train_generator(self, input_tensor: torch.Tensor):
        self.generator.train()

        inp = input_tensor[:, :-1]
        target = input_tensor[:, 1:]
        g_nll_loss = self.generator.pre_train_model(inp, target)
        return g_nll_loss

    def train_generator(self, ref_tensor: torch.Tensor,
                        gen_samples: torch.Tensor = None, rollout_results: torch.Tensor = None):
        self.generator.train()

        sample_batch_size = 32

        if rollout_results is None:
            with torch.no_grad():
                gen_samples, gen_states = self.generator.generate_samples(sample_batch_size)
            rollout_results = self.rollout_generator(gen_samples, gen_states, self.n_rollouts)

        with torch.no_grad():
            rewards = self.get_reward(rollout_results, ref_tensor)
        g_loss = self.generator.train_model(gen_samples, rewards)
        return g_loss, rewards.mean(), gen_samples, rollout_results

    def train_ranker(self, input_tensor: torch.Tensor, label_tensor: torch.Tensor, ref_tensor: torch.Tensor):
        self.ranker.train()

        batch_size = input_tensor.shape[0]

        # Generate samples
        with torch.no_grad():
            gen_outputs, _ = self.generator.generate_samples(batch_size)
        gen_outputs = gen_outputs.permute(1, 0)
        # (B, seq_len)
        gen_labels = torch.Tensor([0] * batch_size).type(torch.LongTensor).to(self.device)
        # Merge
        merged_input_tensor = torch.cat([input_tensor, gen_outputs], dim=0)
        merged_label_tensor = torch.concat([label_tensor, gen_labels])

        r_loss = self.ranker.train_model(merged_input_tensor, merged_label_tensor, ref_tensor)
        return r_loss

    def save_generator(self, file_name: str):
        self.last_generator_save_path = file_name
        torch.save(self.generator.state_dict(), os.path.join(self.save_path, file_name))
        print(f'Generator saved at {file_name}')

    def save_ranker(self, file_name: str):
        torch.save(self.ranker.state_dict(), os.path.join(self.save_path, file_name))
        print(f'Ranker saved at {file_name}')

    def sync_rollout_to_generator(self):
        assert self.last_generator_save_path != ''
        save_file_path = os.path.join(self.save_path, self.last_generator_save_path)
        self.rollout.load_state_dict(torch.load(save_file_path))
