from stepgan.criterion import TBCELoss, TCELoss, WeightedMSELoss

from stepgan.generator import Generator
from stepgan.discriminator import Discriminator

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import os
import datetime
import pickle


class ClippedAdam(torch.optim.Adam):
    def __init__(self, parameters, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)
        self.clip_value = None
        self._parameters = parameters

    def set_clip(self, clip_value):
        self.clip_value = clip_value

    def step(self, *args, **kwargs):
        # assert (self.clip_value is not None)
        if self.clip_value is None:
            self.clip_value = 5.0
        clip_grad_norm_(self._parameters, self.clip_value)
        super().step(*args, **kwargs)


class StepGAN:
    def __init__(self, train_dataset: Dataset, label_size: int, n_vocabs: int, batch_size: int,
                 gen_hidden_size: int, dis_hidden_size: int, gen_lr: float, dis_lr: float,
                 encoder_seq_len: int, decoder_seq_len: int, save_path: str, device: torch.device):
        criterion_g_pre = TCELoss()
        criterion_d = TBCELoss()
        criterion_c = WeightedMSELoss()

        optimizer_g_e = ClippedAdam
        optimizer_g_d = ClippedAdam
        optimizer_d_e = ClippedAdam
        optimizer_d_d = ClippedAdam
        optimizer_c_e = ClippedAdam
        optimizer_c_d = ClippedAdam

        self.device = device
        self.generator = Generator(n_vocabs, n_vocabs, gen_hidden_size, label_size, encoder_seq_len, decoder_seq_len,
                                   gen_lr, criterion_g_pre, optimizer_g_e, optimizer_g_d,
                                   device).to(self.device)
        self.discriminator = Discriminator(n_vocabs, 1, dis_hidden_size, label_size, encoder_seq_len, decoder_seq_len,
                                           dis_lr, criterion_d, optimizer_d_e, optimizer_d_d,
                                           device, True).to(self.device)
        # Critic은 discriminator와 구조가 같음
        self.critic = Discriminator(n_vocabs, 1, dis_hidden_size, label_size, encoder_seq_len, decoder_seq_len,
                                    dis_lr, criterion_c, optimizer_c_e, optimizer_c_d,
                                    device, False).to(self.device)

        self.save_path = save_path
        self.train_dataloader = StepGAN.__build_dataloader(train_dataset, batch_size)

    @staticmethod
    def __build_dataloader(data: Dataset, batch_size: int):
        return DataLoader(data, batch_size=batch_size, shuffle=True)

    def train(self, gen_pre_epochs: int, dis_pre_epochs: int, adv_epochs: int):
        print_period = 100
        loss_acc_dict = {'pre_gen_loss': [], 'pre_dis_loss': [], 'pre_dis_acc': [], 'pre_cri_loss': [],
                         'adv_gen_loss': [], 'adv_dis_loss': [], 'adv_dis_acc': [], 'adv_cri_loss': [],
                         'adv_reward': []}

        for epoch in range(gen_pre_epochs):
            batch_counter = 0
            avg_gen_nll_loss = 0
            for x_tensor, y_tensor, label_tensor in self.train_dataloader:
                gen_nll_loss = self.pre_train_generator(x_tensor, y_tensor, label_tensor)

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
            avg_disc_loss = 0
            avg_disc_acc = 0
            avg_critic_loss = 0
            for x_tensor, y_tensor, label_tensor in self.train_dataloader:
                disc_loss, disc_acc = self.train_discriminator(x_tensor, y_tensor, label_tensor)
                critic_loss = self.train_critic(x_tensor, y_tensor, label_tensor)

                batch_counter += 1
                avg_disc_loss += disc_loss
                avg_disc_acc += disc_acc
                avg_critic_loss += critic_loss

                if batch_counter % print_period == 0:
                    print(f'[Pre]Dis loss: {disc_loss}, Dis acc:{disc_acc}, Critic loss: {critic_loss}')

            avg_disc_loss /= batch_counter
            avg_disc_acc /= batch_counter
            avg_critic_loss /= batch_counter
            self.save_discriminator(f'pre_trained_dis_all_{epoch}')
            self.save_critic(f'pre_trained_cri_all_{epoch}')
            loss_acc_dict['pre_dis_loss'].append(avg_disc_loss)
            loss_acc_dict['pre_dis_acc'].append(avg_disc_acc)
            loss_acc_dict['pre_cri_loss'].append(avg_critic_loss)
            print(f'[Pre][{epoch}]Dis loss: {avg_disc_loss}, Dis acc:{avg_disc_acc}, Critic loss: {avg_critic_loss} '
                  f'at {datetime.datetime.now()}\n')

        for epoch in range(adv_epochs):
            print(f'Epoch: {epoch}')
            batch_counter = 0
            avg_disc_loss = 0
            avg_disc_acc = 0
            avg_gen_loss = 0
            avg_reward = 0
            avg_critic_loss = 0

            for x_tensor, y_tensor, label_tensor in self.train_dataloader:
                disc_loss, disc_acc = self.train_discriminator(x_tensor, y_tensor, label_tensor)
                critic_loss = self.train_critic(x_tensor, y_tensor, label_tensor)
                gen_loss = self.train_generator(x_tensor, y_tensor, label_tensor)

                batch_counter += 1
                avg_disc_loss += disc_loss
                avg_disc_acc += disc_acc
                avg_gen_loss += gen_loss
                avg_reward += (-gen_loss)
                avg_critic_loss += critic_loss

                if batch_counter % print_period == 0:
                    print(f'[Adv]Dis loss: {disc_loss}, Dis acc:{disc_acc}, Gen loss: {gen_loss}, '
                          f'Reward: {-gen_loss}, Cri loss: {critic_loss}')

            avg_disc_loss /= batch_counter
            avg_disc_acc /= batch_counter
            avg_gen_loss /= batch_counter
            avg_reward /= batch_counter
            avg_critic_loss /= batch_counter
            self.save_generator(f'adv_trained_gen_all_{epoch}')
            self.save_discriminator(f'adv_trained_dis_all_{epoch}')
            self.save_critic(f'adv_trained_cri_all_{epoch}')
            loss_acc_dict['adv_dis_loss'].append(avg_disc_loss)
            loss_acc_dict['adv_dis_acc'].append(avg_disc_acc)
            loss_acc_dict['adv_gen_loss'].append(avg_gen_loss)
            loss_acc_dict['adv_reward'].append(avg_reward)
            loss_acc_dict['adv_cri_loss'].append(avg_critic_loss)
            print(f'[Adv][{epoch}]Dis loss: {avg_disc_loss}, Dis acc: {avg_disc_acc}, Gen loss: {avg_gen_loss}, '
                  f'Reward: {avg_reward}, Cri loss: {avg_critic_loss} at {datetime.datetime.now()}\n')

        loss_acc_path = os.path.join(self.save_path, 'loss_acc_dict(maskgan).pickle')
        with open(loss_acc_path, 'wb') as pickle_file:
            pickle.dump(loss_acc_dict, pickle_file)

    def pre_train_generator(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, label_tensor: torch.Tensor):
        self.generator.train()
        return self.generator.pre_train_model(x_tensor, y_tensor, label_tensor)

    def train_generator(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, label_tensor: torch.Tensor):
        self.generator.train()
        generated_tensor, log_probs = self.generator(input_tensor, target_tensor, label_tensor)

        with torch.no_grad():
            # Discriminator 결괏값
            dis_logits, _ = self.discriminator(input_tensor, label_tensor, generated_tensor)
            # (seq_len, B, 1)
            dis_logits = dis_logits.squeeze(2).permute(1, 0)
            # (B, seq_len)
            q_values = nn.Sigmoid()(dis_logits)  # Score 계산
            # (B, seq_len)

            # Critic 베이스라인
            baselines, _ = self.critic(input_tensor, label_tensor, generated_tensor)
            # (seq_len, B, 1)
            baselines = baselines.squeeze(2).permute(1, 0)
            # (B, seq_len)

        gen_loss = self.generator.train_model(log_probs, q_values.detach().clone(), baselines.detach().clone())

        return gen_loss

    def train_discriminator(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, label_tensor: torch.Tensor):
        generated_tensor, _ = self.generator.sample_data(x_tensor, y_tensor, label_tensor)

        real_loss, r_answers = self.discriminator.train_discriminator_model(x_tensor, y_tensor,
                                                                            label_tensor, True)
        fake_loss, f_answers = self.discriminator.train_discriminator_model(x_tensor, generated_tensor,
                                                                            label_tensor, False)

        return (real_loss + fake_loss) / 2, (r_answers + f_answers) / (2 * x_tensor.shape[0])

    def train_critic(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, label_tensor: torch.Tensor):
        with torch.no_grad():
            generated_tensor, log_probs \
                = self.generator.sample_data(x_tensor, y_tensor, label_tensor)
            dis_logits, _ = self.discriminator(x_tensor, label_tensor, generated_tensor)
            dis_logits = dis_logits.squeeze(2).permute(1, 0)
            # (B, seq_len)
            q_values = nn.Sigmoid()(dis_logits)  # Score 계산

        self.critic.train()
        values, _ = self.critic(x_tensor, label_tensor, generated_tensor)
        values = values.squeeze(2).permute(1, 0)
        # (B, seq_len)

        # Cumulative rewards
        c_loss = self.critic.train_critic_model(values, q_values.detach().clone())
        return c_loss

    def save_models(self, target_info: dict) -> None:
        def _save(_model_name: str, _model_abbr: str, _model: nn.Module) -> None:
            # _model.cpu()
            torch.save(_model.state_dict(), target_info[_model_abbr], pickle_protocol=4)
            print(f'{_model_name} saved at {target_info[_model_abbr]}')

        if 'g' in target_info:
            _save('Generator', 'g', self.generator)
        if 'd' in target_info:
            _save('Discriminator', 'd', self.discriminator)
        if 'c' in target_info:
            _save('Critic', 'c', self.critic)
        return

    def save_generator(self, file_name: str):
        torch.save(self.generator.state_dict(), os.path.join(self.save_path, file_name))
        print(f'Generator saved at {file_name}')

    def save_discriminator(self, file_name: str):
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_path, file_name))
        print(f'Discriminator saved at {file_name}')

    def save_critic(self, file_name: str):
        torch.save(self.critic.state_dict(), os.path.join(self.save_path, file_name))
        print(f'Critic saved at {file_name}')
