from leakgan.generator import Generator
from leakgan.discriminator import Discriminator

import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import datetime
import pickle
from scipy.special import expit


class LeakGAN:
    def __init__(self, train_dataset: Dataset, n_vocabs: int, n_labels: int, batch_size: int,
                 gen_hidden_size: int, dis_hidden_size: int, n_rollouts: int, max_seq_len: int,
                 gen_lr: float, dis_lr: float, save_path: str,
                 msg_set: np.ndarray, msg_id_prob_dist: np.ndarray, device: torch.device):
        assert max_seq_len > 0

        self.n_vocabs = n_vocabs
        self.n_labels = n_labels
        self.n_rollouts = n_rollouts
        self.max_seq_len = max_seq_len

        self.save_path = save_path
        self.device = device

        generator_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        discriminator_criterion = torch.nn.BCEWithLogitsLoss()

        discriminator_optimizer = torch.optim.Adam
        generator_manager_optimizer = torch.optim.Adam
        generator_worker_optimizer = torch.optim.Adam

        self.discriminator \
            = Discriminator(n_vocabs, self.max_seq_len, dis_hidden_size, self.n_labels,
                            dis_lr, discriminator_criterion, discriminator_optimizer, self.device).to(self.device)

        goal_size = 16
        goal_out_size = self.discriminator.total_n_filters
        self.step_size = 4

        self.generator \
            = Generator(n_vocabs, self.n_labels, batch_size, self.max_seq_len,
                        gen_hidden_size, self.discriminator.total_n_filters, goal_size, goal_out_size, self.step_size,
                        gen_lr, generator_criterion, generator_manager_optimizer, generator_worker_optimizer,
                        msg_set, msg_id_prob_dist, self.device).to(self.device)

        self.train_dataloader = LeakGAN.__build_dataloader(train_dataset, batch_size)

    @staticmethod
    def __build_dataloader(data: Dataset, batch_size: int) -> DataLoader:
        return DataLoader(data, batch_size=batch_size, shuffle=True)

    def init_vars(self, batch_size: int):
        h_w_t, c_w_t = self.generator.init_hidden(batch_size)  # worker unit of gen
        # (B, hidden_size), (B, hidden_size)
        h_m_t, c_m_t = self.generator.init_hidden(batch_size)  # manager unit of gen
        # (B, hidden_size), (B, hidden_size)
        last_goal = torch.zeros((batch_size, self.generator.worker.goal_out_size), device=self.device)
        # (B, goal_out_size)
        real_goal = self.generator.manager.goal_init[:batch_size]
        # (B, goal_out_size)
        x_t = torch.nn.init.constant_(torch.Tensor(batch_size), self.discriminator.start_token)
        x_t = x_t.type(torch.LongTensor).to(self.device)
        # (B,)

        return [h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t]

    def train(self, gen_pre_epochs: int, dis_pre_epochs: int, adv_epochs: int) -> None:
        print_period = 100
        loss_acc_dict = {'pre_m_loss': [], 'pre_w_loss': [], 'pre_d_loss': [],
                         'adv_m_loss': [], 'adv_w_loss': [], 'adv_d_loss': [], 'adv_reward': []}

        # Pretraining Discriminator
        for epoch in range(dis_pre_epochs):
            batch_counter = 0
            avg_d_loss = 0
            for input_tensor, label_tensor in self.train_dataloader:
                # label_tensor_ = label_tensor.squeeze()
                d_loss, d_acc = self.train_discriminator(input_tensor, label_tensor)

                batch_counter += 1
                avg_d_loss += d_loss

                if batch_counter % print_period == 0:
                    print(f'[Pre]Discriminator loss: {d_loss} {batch_counter}')

            avg_d_loss /= batch_counter
            self.save_discriminator(f'pre_trained_dis_all_{epoch}')
            loss_acc_dict['pre_d_loss'].append(avg_d_loss)
            print(f'[Pre][{epoch}]Discriminator loss: {avg_d_loss} at {datetime.datetime.now()}\n')

        # Pretraining Generator
        for epoch in range(gen_pre_epochs):
            batch_counter = 0
            avg_mloss = 0
            avg_wloss = 0
            for input_tensor, label_tensor in self.train_dataloader:
                # label_tensor_ = label_tensor.squeeze()
                mloss, wloss = self.pre_train_generator(input_tensor, label_tensor)

                batch_counter += 1
                avg_mloss += mloss
                avg_wloss += wloss

                if batch_counter % print_period == 0:
                    print(f'[Pre]Manager loss: {mloss}, Worker loss: {wloss}')

            avg_mloss /= batch_counter
            avg_wloss /= batch_counter
            self.save_generator(f'pre_trained_gen_all_{epoch}')
            loss_acc_dict['pre_m_loss'].append(avg_mloss)
            loss_acc_dict['pre_w_loss'].append(avg_wloss)
            print(f'[Pre][{epoch}]Manager loss: {avg_mloss}, Worker loss: {avg_wloss} '
                  f'at {datetime.datetime.now()}\n')

        # Adversarial training
        for epoch in range(adv_epochs):
            batch_counter = 0
            avg_d_loss = 0
            avg_mloss = 0
            avg_wloss = 0
            avg_reward = 0

            for input_tensor, label_tensor in self.train_dataloader:
                # label_tensor_ = label_tensor.squeeze()
                mloss, wloss, rewards = self.train_generator(label_tensor, self.n_rollouts)
                d_loss, d_acc = self.train_discriminator(input_tensor, label_tensor)

                batch_counter += 1
                avg_d_loss += d_loss
                avg_mloss += mloss
                avg_wloss += wloss
                avg_reward += rewards

                if batch_counter % print_period == 0:
                    print(f'[Adv]Manager loss: {mloss}, Worker loss: {wloss}')
                    print(f'[Adv]Dis loss: {d_loss}')

            avg_d_loss /= batch_counter
            avg_mloss /= batch_counter
            avg_wloss /= batch_counter
            avg_reward /= batch_counter

            loss_acc_dict['adv_d_loss'].append(avg_d_loss)
            loss_acc_dict['adv_m_loss'].append(avg_mloss)
            loss_acc_dict['adv_w_loss'].append(avg_wloss)
            loss_acc_dict['adv_reward'].append(avg_reward)

            self.save_generator(f'adv_trained_gen_all_{epoch}')
            self.save_discriminator(f'adv_trained_dis_all_{epoch}')

            print(f'[Adv][{epoch}]Dis loss: {avg_d_loss}, Manager loss: {avg_mloss}, Worker loss: {avg_wloss} '
                  f'Reward: {avg_reward} at {datetime.datetime.now()}\n')

        loss_acc_path = os.path.join(self.save_path, 'loss_acc_dict(leakgan).pickle')
        with open(loss_acc_path, 'wb') as pickle_file:
            pickle.dump(loss_acc_dict, pickle_file)

    def pre_train_generator(self, input_tensor: torch.Tensor, label_tensor: torch.Tensor):
        self.generator.train()

        results = self.__forward_pre_train_generator(input_tensor, label_tensor)
        real_goal, prediction, delta_feature = results['real_goal'], results['prediction'], results['delta_feature']
        return self.generator.pre_train_model(input_tensor, real_goal, prediction, delta_feature)

    def __forward_pre_train_generator(self, input_tensor: torch.Tensor, label_tensor: torch.Tensor):
        batch_size = input_tensor.shape[0]

        self.discriminator.eval()
        temperature = 1.0

        h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = self.init_vars(batch_size)
        t = 0
        feature_list = []
        delta_feature_list = []  # F(St+c) - F(St) = used to calculate the gradient of manager module
        prediction_list = []
        real_goal_list = []
        seq_len = self.max_seq_len
        goal_out_size = self.generator.worker.goal_out_size

        while t < seq_len + 1:
            # Extract Feature from D
            if t == 0:
                cur_sen = torch.nn.init.constant_(torch.zeros(batch_size, seq_len), self.n_vocabs)
                cur_sen = cur_sen.type(torch.LongTensor)
                # (B, seq_len)
            else:
                cur_sen = input_tensor[:, :t]
                cur_sen = cur_sen.contiguous()
                cur_sen = torch.nn.functional.pad(cur_sen.view(-1, t), (0, seq_len - t), value=self.n_vocabs)

            cur_sen = cur_sen.to(self.device)
            with torch.no_grad():
                _, f_t = self.discriminator(cur_sen, label_tensor)
            x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, \
                sub_goal, probs, t_ = self.generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t,
                                                     last_goal, real_goal, label_tensor, t, temperature)
            if t % self.step_size == 0:
                if t > 0:
                    real_goal = last_goal
                last_goal = torch.zeros((batch_size, goal_out_size), device=self.device)
                real_goal_list.append(real_goal)
            """
            Store needed information for calculating loss function
            """
            feature_list.append(f_t)
            prediction_list.append(probs)
            if t > 0:
                if t % self.step_size == 0:
                    delta_feature_list.append(f_t - feature_list[t - self.step_size])
            t = t_

        """
        Post process and return variables needed for calculating loss
        """
        if len(real_goal_list) == len(delta_feature_list) + 1:
            real_goal_list = real_goal_list[:-1]  # exclude the last element
        prediction_list = prediction_list[:-1]
        real_goal_var = torch.stack(real_goal_list).permute(1, 0, 2)
        # (B, seq_len, goal_out_size)
        prediction_var = torch.stack(prediction_list).permute(1, 0, 2)
        # (B, seq_len, vocab_size)
        delta_feature_var = torch.stack(delta_feature_list).permute(1, 0, 2)
        # real_goal = g_t, prediction = generator sentence, delta_feature = F(s_(t+c))-F(s_t)

        results = {"real_goal": real_goal_var, "prediction": prediction_var, "delta_feature": delta_feature_var}
        for result in results.values():
            if result.is_contiguous():
                result = result.contiguous()
        return results

    def generate_samples(self, label_tensor: torch.Tensor):
        self.generator.eval()
        self.discriminator.eval()

        batch_size = label_tensor.shape[0]

        h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = self.init_vars(batch_size)
        t = 0
        gen_token_list = []
        seq_len = self.max_seq_len
        goal_out_size = self.generator.worker.goal_out_size
        while t < seq_len:
            if t == 0:
                cur_sen = torch.nn.init.constant_(torch.zeros(batch_size, seq_len), self.n_vocabs).to(self.device)
                # (B, seq_len)
                cur_sen = cur_sen.type(torch.LongTensor)
            else:
                cur_sen = torch.stack(gen_token_list).permute(1, 0)
                # (B, t)
                cur_sen = torch.nn.functional.pad(cur_sen, (0, seq_len - t), value=self.n_vocabs)
                # (B, seq_len)
            cur_sen = cur_sen.to(self.device)
            with torch.no_grad():
                _, f_t = self.discriminator(cur_sen, label_tensor)
                # G forward step
                x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, sub_goal, probs, t_ \
                    = self.generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, label_tensor, t)
            if t % self.step_size == 0:
                if t > 0:
                    real_goal = last_goal
                    last_goal = torch.zeros((batch_size, goal_out_size)).to(self.device)
            gen_token_list.append(x_t)
            t = t_
        samples = torch.stack(gen_token_list).permute(1, 0)
        # (B, seq_len)

        return samples

    def generate_labels(self, batch_size: int) -> torch.Tensor:
        label_tensor = torch.zeros(batch_size, self.n_labels)

        for idx in range(self.n_labels):
            start_idx = int(batch_size / self.n_labels) * idx
            end_idx = int(batch_size / self.n_labels) * (idx + 1)
            label_tensor[start_idx: end_idx, idx] = 1.0

        return label_tensor.to(self.device)

    def train_discriminator(self, pos_data: torch.Tensor, pos_label_tensor: torch.Tensor):
        batch_size = pos_data.shape[0]

        neg_label_tensor = self.generate_labels(batch_size)
        neg_data = self.generate_samples(neg_label_tensor)
        return self.discriminator.train_model(pos_data, pos_label_tensor, neg_data, neg_label_tensor)

    def train_generator(self, label_tensor: torch.Tensor, rollout_num=4):
        self.generator.train()
        self.generator.manager_optimizer.zero_grad()
        self.generator.worker_optimizer.zero_grad()

        # get all the return values
        adv_rets = self.__forward_train_generator(label_tensor)
        gen_token = adv_rets["gen_token"]

        # Manager loss
        with torch.no_grad():
            rewards = self.get_rewards(gen_token, label_tensor, rollout_num)
        m_loss, w_loss = self.generator.train_model(adv_rets, rewards)
        del adv_rets
        print("Adv-Manager loss: {:.5f} Adv-Worker loss: {:.5f}".format(m_loss, w_loss))

        return m_loss, w_loss, torch.mean(rewards).item()

    def __forward_train_generator(self, label_tensor: torch.Tensor):
        self.discriminator.eval()

        batch_size = label_tensor.shape[0]

        h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = self.init_vars(batch_size)
        t = 0
        feature_list = []
        delta_feature_list = []  # f_(t+c) - f_t
        delta_feature_for_worker_list = []  # f_t - f_(t-i)
        prediction_list = []
        real_goal_list = []
        all_goal_list = []
        gen_token_list = []
        seq_len = self.discriminator.max_seq_len
        goal_out_size = self.generator.worker.goal_out_size
        while t < seq_len + 1:
            if t == 0:
                cur_sen = torch.nn.init.constant_(torch.zeros((batch_size, seq_len), device=self.device),
                                                  self.n_vocabs)
                cur_sen = cur_sen.type(torch.LongTensor)
            else:
                cur_sen = torch.stack(gen_token_list).permute(1, 0)
                # (B, t)
                cur_sen = torch.nn.functional.pad(cur_sen, (0, seq_len - t), value=self.n_vocabs)
                # (B, seq_len)
            cur_sen = cur_sen.to(self.device)
            with torch.no_grad():
                _, f_t = self.discriminator(cur_sen, label_tensor)

            x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, sub_goal, probs, t_ \
                = self.generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, label_tensor, t)
            if t % self.step_size == 0:
                if t > 0:
                    real_goal = last_goal
                last_goal = torch.zeros((batch_size, goal_out_size), device=self.device)
                # (B, goal_out_size)
                real_goal_list.append(real_goal)
            # Store info for calculating loss function
            feature_list.append(f_t)
            prediction_list.append(probs)
            if t > 0:
                if t % self.step_size == 0:
                    delta_feature_list.append(f_t - feature_list[t - self.step_size])
                    delta_feature_for_worker_list.append(f_t - feature_list[t - self.step_size])
                else:
                    delta_feature_for_worker_list.append(f_t - feature_list[t - t % self.step_size])
                all_goal_list.append(real_goal)
            gen_token_list.append(x_t)  # next token generated by G
            t = t_

        # Post Process and return variables
        if len(real_goal_list) == len(delta_feature_list) + 1:
            real_goal_list = real_goal_list[:-1]
        prediction_list = prediction_list[:-1]
        gen_token_list = gen_token_list[:-1]
        real_goal_var = torch.stack(real_goal_list).permute(1, 0, 2)
        # (B, seq_len / 4, goal_out_size)
        all_goal_var = torch.stack(all_goal_list).permute(1, 0, 2)
        # (B, seq_len, goal_out_size)
        prediction_var = torch.stack(prediction_list).permute(1, 0, 2)
        # (B, seq_len, vocab_size)
        delta_feature_var = torch.stack(delta_feature_list).permute(1, 0, 2)
        # (B, seq_len / 4, total_n_filters)
        gen_token_var = torch.stack(gen_token_list).permute(1, 0)
        # (B, seq_len)
        delta_feature_for_worker_var = torch.stack(delta_feature_for_worker_list).permute(1, 0, 2)
        # (B, seq_len, total_n_filters)

        results = {"real_goal": real_goal_var,
                   "all_goal": all_goal_var,
                   "prediction": prediction_var,
                   "delta_feature": delta_feature_var,
                   "delta_feature_for_worker": delta_feature_for_worker_var,
                   "gen_token": gen_token_var}
        """
        for result in results.values():
            if result.is_contiguous():
                result = result.contiguous()
        """
        return results

    def get_rewards(self, input_x, label_tensor: torch.Tensor, rollout_num, temperature=1.0, delta=16.0):
        self.discriminator.eval()
        seq_len = self.discriminator.max_seq_len
        rewards = []
        for i in range(rollout_num):
            given_num = 0
            while given_num < seq_len:
                sample_for_reward = self.__rollout(input_x, label_tensor, given_num, temperature)
                with torch.no_grad():
                    score, _ = self.discriminator(sample_for_reward, label_tensor)
                # pred = torch.nn.functional.log_softmax(score, dim=1)  # (B, 1)
                pred = torch.nn.functional.logsigmoid(score)  # (B, 1)
                pred = pred[:, 0].data
                pred = pred.to('cpu')
                pred = pred.numpy()
                pred = pred.reshape(-1)
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[int(given_num / self.step_size - 1)] += pred
                given_num += self.step_size
        rewards = self.rescale(rewards, delta) / rollout_num
        rewards = rewards.to(self.device)
        return rewards

    def __rollout(self, input_x, label_tensor: torch.Tensor, given_num, temperature):
        self.discriminator.eval()

        batch_size = input_x.shape[0]

        h_w_t, c_w_t, h_m_t, c_m_t, last_goal, real_goal, x_t = self.init_vars(batch_size)
        t = 0
        gen_token_list = []
        seq_len = self.max_seq_len
        goal_out_size = self.generator.worker.goal_out_size
        while t < given_num + 1:
            if t == 0:
                cur_sen = torch.nn.init.constant_(torch.zeros((batch_size, seq_len), device=self.device),
                                                  self.n_vocabs)
                cur_sen = cur_sen.type(torch.LongTensor)
            else:
                cur_sen = torch.stack(gen_token_list).permute(1, 0)
                cur_sen = torch.nn.functional.pad(cur_sen, (0, seq_len - t), value=self.n_vocabs)
            cur_sen = cur_sen.to(self.device)
            with torch.no_grad():
                _, f_t = self.discriminator(cur_sen, label_tensor)

            _, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, sub_goal, probs, t_ \
                = self.generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, label_tensor,
                                 t, temperature)
            if t % self.step_size == 0:
                if t > 0:
                    real_goal = last_goal
                last_goal = torch.zeros((batch_size, goal_out_size), device=self.device)
            if t < given_num:
                x_t = input_x[:, t].contiguous()
                gen_token_list.append(x_t)
            t = t_

        # Rollout
        while t < seq_len + 1:
            if len(gen_token_list) == 0:
                cur_sen = torch.nn.init.constant_(torch.zeros((batch_size, seq_len), device=self.device),
                                                  self.n_vocabs)
                cur_sen = cur_sen.type(torch.LongTensor)
            else:
                cur_sen = torch.stack(gen_token_list).permute(1, 0)
                cur_sen = torch.nn.functional.pad(cur_sen, (0, seq_len - t + 1), value=self.n_vocabs)
            cur_sen = cur_sen.to(self.device)
            with torch.no_grad():
                _, f_t = self.discriminator(cur_sen, label_tensor)

            x_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, sub_goal, probs, t_ \
                = self.generator(x_t, f_t, h_m_t, c_m_t, h_w_t, c_w_t, last_goal, real_goal, label_tensor,
                                 t, temperature)
            if t % self.step_size == 0:
                real_goal = last_goal
            last_goal = torch.zeros((batch_size, goal_out_size), device=self.device)
            gen_token_list.append(x_t)
            t = t_
        gen_token = torch.stack(gen_token_list).permute(1, 0)
        return gen_token

    @staticmethod
    def rescale(rewards, delta=16.0):
        """
        Why Rescaled activation: during adversarial training of SeqGAN severe gradient vanishing occurs
            when D is much stronger than G, i.e. the reward is too small value to update the parameters
        and thus need to be rescaled before being fed into G.
            parameters for rewards:
                type: list
                length: seq_len / c, where c is c recent goals(steps into future)
                elements: np.array(size=batch_size)
                R(reward matrix) = expit(delta * (0.5 - rank(i)/B)), where expit, is an activation function
                    that re-projects the equidifferent scoring based on ranking to a more effective distribution.
                In this model authors of the paper decided expit to be sigmoid function: expit = 1/(1+exp(-x))
        """
        r = np.array(rewards)
        _, batch_size = r.shape
        order = np.argsort(r)
        rank = np.argsort(order)
        rank = batch_size - rank
        rescaled_rewards = expit(delta * (0.5 - rank / batch_size))
        rescaled_rewards = np.transpose(rescaled_rewards)
        return torch.from_numpy(rescaled_rewards)

    def save_generator(self, file_name: str):
        torch.save(self.generator.state_dict(), os.path.join(self.save_path, file_name))
        print(f'Generator saved at {file_name}')

    def save_discriminator(self, file_name: str):
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_path, file_name))
        print(f'Discriminator saved at {file_name}')
