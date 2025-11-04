import seqgan_gen_dis
import util_config

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np

from math import ceil
import sys
import copy
import datetime
import pickle
import os


def prepare_generator_batch(samples, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    batch_size, seq_len = samples.size()

    inp = samples[:, :seq_len-1]
    target = samples[:, 1:]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_discriminator_data(pos_samples, pos_labels, neg_samples, neg_samples_softmax, neg_labels,
                               gpu, is_wasserstein):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """

    if not is_wasserstein:
        inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
        target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
        target[pos_samples.size()[0]:] = 0
        labels = torch.cat((pos_labels, neg_labels), 0)

        # shuffle
        perm = torch.randperm(target.size()[0])
        target = target[perm]
        inp = inp[perm]
        labels = labels[perm]

        inp = Variable(inp)
        target = Variable(target)
        labels = Variable(labels)

        if gpu:
            inp = inp.cuda()
            target = target.cuda()
            labels = labels.cuda()
        return inp, target, labels
    else:
        inp_pos = pos_samples.type(torch.LongTensor)
        target_pos = torch.ones(pos_samples.size()[0])
        inp_neg = neg_samples.type(torch.LongTensor)
        target_neg = torch.zeros(neg_samples.size()[0])
        inp_neg_softmax = neg_samples_softmax.type(torch.LongTensor)
        target_neg_softmax = torch.zeros(neg_samples_softmax.size()[0])

        # shuffle
        perm = torch.randperm(target_pos.size()[0])
        target_pos = target_pos[perm]
        inp_pos = inp_pos[perm]
        label_pos = pos_labels[perm]

        perm = torch.randperm(target_neg.size()[0])
        target_neg = target_neg[perm]
        inp_neg = inp_neg[perm]
        label_neg = neg_labels[perm]
        inp_neg_softmax = inp_neg_softmax[perm]
        target_neg_softmax = target_neg_softmax[perm]

        inp_pos = Variable(inp_pos)
        target_pos = Variable(target_pos)
        label_pos = Variable(label_pos)
        inp_neg = Variable(inp_neg)
        target_neg = Variable(target_neg)
        label_neg = Variable(label_neg)
        inp_neg_softmax = Variable(inp_neg_softmax)
        target_neg_softmax = Variable(target_neg_softmax)

        if gpu:
            inp_pos = inp_pos.cuda()
            target_pos = target_pos.cuda()
            label_pos = label_pos.cuda()
            inp_neg = inp_neg.cuda()
            target_neg = target_neg.cuda()
            label_neg = label_neg.cuda()
            inp_neg_softmax = inp_neg_softmax.cuda()
            target_neg_softmax = target_neg_softmax.cuda()

        return inp_pos, target_pos, label_pos, inp_neg, target_neg, label_neg, inp_neg_softmax, target_neg_softmax


def batchwise_sample(gan, num_samples, batch_size, occurred_msg_id_list, msg_id_prob_dist):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """
    start_letters = np.random.choice(occurred_msg_id_list, size=num_samples, p=msg_id_prob_dist)
    labels = gan.generate_labels(num_samples)
    samples, samples_softmax = gan.gen.sample(num_samples, start_letters, labels)

    return samples, samples_softmax, labels


class SeqGAN:
    def __init__(self, encoded_data, labels, msg_set, msg_id_prob_dist, seq_len, save_path, cuda):
        np.random.seed(1)
        param_config = util_config.CONFIG.get_param_config()['seqgan']
        if cuda == 'cuda':
            self.CUDA = True
        else:
            self.CUDA = False
        self.MAX_SEQ_LEN = seq_len
        self.BATCH_SIZE = param_config['batch_size']

        self.MLE_TRAIN_EPOCHS = param_config['mle_train_epochs']
        self.pre_dis_steps = param_config['pre_dis_steps']
        self.pre_dis_epochs = param_config['pre_dis_epochs']
        self.ADV_TRAIN_EPOCHS = param_config['adv_train_epochs']
        self.adv_gen_epochs = param_config['adv_gen_epochs']
        self.adv_dis_steps = param_config['adv_dis_steps']
        self.adv_dis_epochs = param_config['adv_dis_epochs']

        self.pre_gen_lr = param_config['pre_gen_lr']
        self.pre_dis_lr = param_config['pre_dis_lr']
        self.adv_gen_lr = param_config['adv_gen_lr']
        self.adv_dis_lr = param_config['adv_dis_lr']

        gen_embedding_dim = 16
        gen_hidden_dim = 64
        dis_embedding_dim = 32
        dis_hidden_dim = 128

        self.msg_id = torch.tensor(encoded_data).view(-1, self.MAX_SEQ_LEN)
        self.labels = torch.tensor(labels).type(torch.FloatTensor)
        self.n_labels = 4
        self.failure_type = 'all'
        self.VOCAB_SIZE = len(msg_set)
        self.POS_NEG_SAMPLES = self.msg_id.shape[0]

        self.occurred_msg_id_list = np.array(list(msg_set.keys()))
        self.msg_id_prob_dist = msg_id_prob_dist
        print(self.msg_id_prob_dist)

        self.UNROLLED_STEPS = param_config['unrolled_step']  # Unrolled GAN
        self.IS_WASSERSTEIN = param_config['is_wasserstein']
        self.CLIPPING = param_config['clipping']    # WGAN
        self.gp_weight = param_config['gp_weight']     # WGAN-GP

        self.gen = seqgan_gen_dis.Generator(gen_embedding_dim, gen_hidden_dim, self.VOCAB_SIZE, self.MAX_SEQ_LEN,
                                            self.n_labels, gpu=self.CUDA)
        self.dis = seqgan_gen_dis.Discriminator(dis_embedding_dim, dis_hidden_dim, self.VOCAB_SIZE, self.MAX_SEQ_LEN,
                                                self.n_labels, gpu=self.CUDA)
        if self.CUDA:
            self.msg_id = self.msg_id.cuda()
            self.labels = self.labels.cuda()
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

        self.n_dis_train = 0

        self.save_path = save_path
        util_config.make_dirs(self.save_path)

        self.print_info()

    def choose_start_letter(self, n_letter: int):
        return np.random.choice(self.occurred_msg_id_list, size=n_letter, p=self.msg_id_prob_dist)

    def generate_labels(self, batch_size: int) -> torch.Tensor:
        label_tensor = torch.zeros(batch_size, self.n_labels)

        for idx in range(self.n_labels):
            start_idx = int(batch_size / self.n_labels) * idx
            end_idx = int(batch_size / self.n_labels) * (idx + 1)
            label_tensor[start_idx: end_idx, idx] = 1.0

        if self.CUDA:
            label_tensor = label_tensor.cuda()

        return label_tensor

    def generate_specific_labels(self, batch_size: int, label_idx: int) -> torch.Tensor:
        assert 0 <= label_idx < self.n_labels

        label_tensor = torch.zeros(batch_size, self.n_labels)
        label_tensor[:, label_idx] = 1.0

        if self.CUDA:
            label_tensor = label_tensor.cuda()

        return label_tensor

    def train_generator_mle(self, gen_opt):
        """
        Max Likelihood Pretraining for the generator
        """
        total_loss_list = []
        for epoch in range(self.MLE_TRAIN_EPOCHS):
            print('epoch %d : ' % (epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0

            for i in range(0, self.POS_NEG_SAMPLES, self.BATCH_SIZE):
                inp, target = prepare_generator_batch(self.msg_id[i:i + self.BATCH_SIZE], gpu=self.CUDA)
                labels = self.labels[i:i + self.BATCH_SIZE]

                gen_opt.zero_grad()
                loss = self.gen.batchNLLLoss(inp, target, labels)
                loss.backward()
                gen_opt.step()

                total_loss += loss.data.item()

                # roughly every 10% of an epoch
                if (i / self.BATCH_SIZE) % ceil(
                        ceil(self.POS_NEG_SAMPLES / float(self.BATCH_SIZE)) / 10.) == 0:
                    print('.', end='')
                    sys.stdout.flush()

            # each loss in a batch is loss per sample
            total_loss = total_loss / ceil(self.POS_NEG_SAMPLES / float(self.BATCH_SIZE)) / self.MAX_SEQ_LEN

            total_loss_list.append(total_loss)
            print(' average_train_NLL = %.4f' % total_loss)

        return total_loss_list

    def train_generator_pg(self, gen_optimizer, gen_epochs, dis_optimizer):
        """
        The generator is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        reward_sum = 0
        pg_loss_sum = 0
        for _ in range(gen_epochs):
            s_batch_size = 128
            for idx in range(0, self.BATCH_SIZE, s_batch_size):
                sample_unit = s_batch_size
                start_letters = self.choose_start_letter(sample_unit)
                labels = self.generate_labels(sample_unit)
                samples, _ = self.gen.sample(sample_unit, start_letters, labels, self.MAX_SEQ_LEN)

                # s, _ = self.gen.sample(self.BATCH_SIZE * 2, start_letter)  # 64 works best
                inp, target = prepare_generator_batch(samples, gpu=self.CUDA)

                dis_backup = None
                if self.UNROLLED_STEPS > 0:
                    dis_backup = copy.deepcopy(self.dis)  # 원래의 discriminator 백업
                    for _ in range(self.UNROLLED_STEPS):
                        self.train_discriminator(dis_optimizer, self.adv_dis_steps,
                                                 self.adv_dis_epochs, is_unrolled_step=True)

                with torch.no_grad():
                    rewards = self.dis.batchClassify(target, labels, sig=True)
                    # rewards = (rewards - 0.5) * 2

                gen_optimizer.zero_grad()
                pg_loss = self.gen.batchPGLoss(inp, target, rewards, labels)
                print(f'PG Loss: {pg_loss}')
                pg_loss_sum += pg_loss.data.item()
                pg_loss.backward()
                gen_optimizer.step()

                batch_size, _ = inp.size()
                reward_sum += torch.sum(rewards).data.item() / batch_size

                if self.UNROLLED_STEPS > 0:  # Discriminator 복원
                    assert dis_backup is not None
                    self.dis.restore(dis_backup)
                    del dis_backup

        return pg_loss_sum / gen_epochs, reward_sum / gen_epochs

    def train_discriminator(self, dis_opt, d_steps, epochs, is_unrolled_step=False, is_gradient_penalty=False):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
        Samples are drawn d_steps times, and the discriminator is trained for epochs.
        """

        def _train_by_vanilla():
            total_loss_list = []
            total_acc_list = []
            val_acc_list = []

            for d_step in range(d_steps):
                for epoch in range(epochs):
                    if not is_unrolled_step:
                        print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
                    sys.stdout.flush()

                    s, _, sampled_labels = batchwise_sample(self, self.BATCH_SIZE, self.BATCH_SIZE,
                                                            self.occurred_msg_id_list, self.msg_id_prob_dist)
                    offset = self.n_dis_train % (int(self.POS_NEG_SAMPLES / self.BATCH_SIZE) + 1)

                    real_data = self.msg_id[offset: offset + self.BATCH_SIZE]
                    real_label = self.labels[offset: offset + self.BATCH_SIZE]

                    inp, target, labels = prepare_discriminator_data(real_data, real_label,
                                                                     s, None, sampled_labels,
                                                                     gpu=self.CUDA,
                                                                     is_wasserstein=False)

                    # (B*2, seq_len)
                    dis_opt.zero_grad()
                    out = self.dis.batchClassify(inp, labels, sig=True)
                    loss_fn = nn.BCELoss()
                    loss = loss_fn(out, target)
                    loss.backward()
                    dis_opt.step()

                    total_loss = loss.data.item()
                    total_acc = torch.sum((out > 0.5) == (target > 0.5)).data.item() / (2 * self.BATCH_SIZE)

                    val_pred = self.dis.batchClassify(val_inp, val_labels, sig=True)
                    val_acc = torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / val_pred.shape[0]

                    total_loss_list.append(total_loss)
                    total_acc_list.append(total_acc)
                    val_acc_list.append(val_acc)

                    if not is_unrolled_step:
                        print(
                            ' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (total_loss, total_acc, val_acc))

            return total_loss_list, total_acc_list, val_acc_list

        def _train_by_wasserstein():
            total_loss_list = []
            total_acc_list = []
            val_acc_list = []

            for d_step in range(d_steps):
                for epoch in range(epochs):
                    if not is_unrolled_step:
                        print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
                    sys.stdout.flush()

                    s, s_softmax, sampled_labels = batchwise_sample(self, self.BATCH_SIZE, self.BATCH_SIZE,
                                                                    self.occurred_msg_id_list, self.msg_id_prob_dist)
                    offset = self.n_dis_train % (int(self.POS_NEG_SAMPLES / self.BATCH_SIZE) + 1)

                    real_data = self.msg_id[offset: offset + self.BATCH_SIZE]
                    real_label = self.labels[offset: offset + self.BATCH_SIZE]

                    inp_pos, target_pos, label_pos, inp_neg, target_neg, label_neg, \
                    inp_neg_softmax, target_neg_softmax \
                        = prepare_discriminator_data(real_data, real_label, s, s_softmax, sampled_labels,
                                                     gpu=self.CUDA, is_wasserstein=True)

                    # Gradient descent using W-Loss
                    dis_opt.zero_grad()
                    out_pos = self.dis.batchClassify(inp_pos, label_pos, sig=False)
                    out_neg = self.dis.batchClassify(inp_neg, label_neg, sig=False)

                    if not is_gradient_penalty:     # Vanilla WGAN
                        loss = - torch.mean(out_pos) + torch.mean(out_neg)
                        loss.backward()

                        # Clipping
                        nn.utils.clip_grad_norm_(self.dis.parameters(), self.CLIPPING)
                    else:       # WGAN-GP
                        print('Temporary')
                        raise NotImplementedError

                    dis_opt.step()
                    avg_loss = loss.data.item()

                    # Accuracy 계산
                    out_pos = self.dis.batchClassify(inp_pos, label_pos, sig=True)
                    out_neg = self.dis.batchClassify(inp_neg, label_neg, sig=True)
                    total_acc = torch.sum((out_pos < 0.5) == (target_pos < 0.5)).data.item()
                    total_acc += torch.sum((out_neg > 0.5) == (target_neg > 0.5)).data.item()
                    total_acc /= (2 * self.BATCH_SIZE)

                    if not is_unrolled_step:
                        print(f'Discriminator total loss: {avg_loss}')

                    val_pred = self.dis.batchClassify(val_inp, val_labels, sig=True)
                    val_acc = torch.sum((val_pred > 0.5) == (val_target > 0.5)).data.item() / val_pred.shape[0]

                    total_loss_list.append(avg_loss)
                    total_acc_list.append(total_acc)
                    val_acc_list.append(val_acc)

                    if not is_unrolled_step:
                        print(
                            ' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (avg_loss, total_acc, val_acc))

            return total_loss_list, total_acc_list, val_acc_list

        neg_val, _, neg_label = batchwise_sample(self, 200, 1, self.occurred_msg_id_list, self.msg_id_prob_dist)
        neg_val_1, _, neg_label_1 = batchwise_sample(self, 200, 1, self.occurred_msg_id_list, self.msg_id_prob_dist)
        val_inp, val_target, val_labels = prepare_discriminator_data(neg_val_1, neg_label, neg_val, None, neg_label_1,
                                                                     gpu=self.CUDA, is_wasserstein=False)
        val_target[:] = 0

        self.n_dis_train += 1

        if not self.IS_WASSERSTEIN:
            return _train_by_vanilla()
        else:
            return _train_by_wasserstein()

    def pre_train_generator(self):
        print('Starting Generator MLE Training...')
        gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.pre_gen_lr)

        pre_gen_total_loss_list = self.train_generator_mle(gen_optimizer)
        assert len(pre_gen_total_loss_list) == self.MLE_TRAIN_EPOCHS

        return pre_gen_total_loss_list

    def pre_train_discriminator(self):
        print('\nStarting Discriminator Pre-Training...')
        dis_optimizer = optim.Adagrad(self.dis.parameters(), lr=self.pre_dis_lr)

        return self.train_discriminator(dis_optimizer, self.pre_dis_steps, self.pre_dis_epochs)

    def train_adversarially(self):
        gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.adv_gen_lr)
        dis_optimizer = optim.RMSprop(self.dis.parameters(), lr=self.adv_dis_lr)

        # ADVERSARIAL TRAINING
        print('\nStarting Adversarial Training...')

        whole_adv_dis_total_loss_list = []
        whole_adv_dis_total_acc_list = []
        whole_adv_dis_val_acc_list = []
        whole_pg_loss_list = []
        whole_reward_list = []
        for epoch in range(self.ADV_TRAIN_EPOCHS):
            print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
            # TRAIN GENERATOR
            print('\nAdversarial Training Generator : ', end='')
            sys.stdout.flush()
            avg_pg_loss, avg_reward = self.train_generator_pg(gen_optimizer, self.adv_gen_epochs, dis_optimizer)
            whole_pg_loss_list.append(avg_pg_loss)
            whole_reward_list.append(avg_reward)
            print(f'Avg. of PG loss: {avg_pg_loss}')
            print(f'Avg. of reward: {avg_reward}')

            # TRAIN DISCRIMINATOR
            print('\nAdversarial Training Discriminator : ')
            adv_dis_total_loss_list, adv_dis_total_acc_list, adv_dis_val_acc_list \
                = self.train_discriminator(dis_optimizer, self.adv_dis_steps, self.adv_dis_epochs,
                                           is_gradient_penalty=False)
            assert len(adv_dis_total_loss_list) == len(adv_dis_total_acc_list) == len(adv_dis_val_acc_list) \
                   == (self.adv_dis_steps * self.adv_dis_epochs)

            for adv_dis_total_loss, adv_dis_total_acc, adv_dis_val_acc \
                    in zip(adv_dis_total_loss_list, adv_dis_total_acc_list, adv_dis_val_acc_list):
                whole_adv_dis_total_loss_list.append(adv_dis_total_loss)
                whole_adv_dis_total_acc_list.append(adv_dis_total_acc)
                whole_adv_dis_val_acc_list.append(adv_dis_val_acc)

            torch.save(self.gen.state_dict(),
                       os.path.join(self.save_path, f'adv_trained_gen_{self.failure_type}_{epoch}'))
            torch.save(self.dis.state_dict(),
                       os.path.join(self.save_path, f'adv_trained_dis_{self.failure_type}_{epoch}'))

        return whole_adv_dis_total_loss_list, whole_adv_dis_total_acc_list, \
               whole_adv_dis_val_acc_list, whole_pg_loss_list, whole_reward_list

    def print_info(self) -> None:
        print("\n*****Model Info.*****")
        print(f"CUDA: {self.CUDA}")

        print(f"Train data shape: {self.msg_id.shape}")
        print(f"Sequence length: {self.MAX_SEQ_LEN}")
        print(f"Failure type: {self.failure_type}")
        print(f"Vocab size: {self.VOCAB_SIZE}")

        print(f"Wasserstein: {self.IS_WASSERSTEIN}")
        print(f"Weight clip: {self.CLIPPING}")
        print(f"Unrolled step(s): {self.UNROLLED_STEPS}")

        print(f"Batch size: {self.BATCH_SIZE}")
        print(f"Epochs of pre-training generator: {self.MLE_TRAIN_EPOCHS}")
        print(f"Epochs of pre-training discriminator: {(50, 3)}")
        print(f"Epochs of adversarial training: {self.ADV_TRAIN_EPOCHS}")

        print(f"Save path: {self.save_path}")
        print("********************\n")

    def save(self, loss_acc_data) -> None:
        if not self.IS_WASSERSTEIN:
            loss_acc_file_path = f'loss_acc_{self.failure_type}.pickle'
        else:
            loss_acc_file_path = f'loss_acc_{self.failure_type}.pickle'
        loss_acc_file_path = os.path.join(self.save_path, loss_acc_file_path)

        with open(loss_acc_file_path, 'wb') as loss_acc_file:
            pickle.dump(loss_acc_data, loss_acc_file)
            print(f'\nWhole Loss and Accuracy information has been written at {datetime.datetime.now()}')

    def train(self, do_pretrain: bool = True) -> None:
        # self.print_info()

        if do_pretrain:
            # 1. Pre-train generator
            pre_gen_total_loss_list = self.pre_train_generator()
            torch.save(self.gen.state_dict(), os.path.join(self.save_path, f'pre_trained_gen_{self.failure_type}'))

            # 2. Pre-train discriminator
            pre_dis_total_loss_list, pre_dis_total_acc_list, pre_dis_val_acc_list = self.pre_train_discriminator()
            torch.save(self.dis.state_dict(), os.path.join(self.save_path, f'pre_trained_dis_{self.failure_type}'))
        else:
            pre_gen_total_loss_list = []
            pre_dis_total_loss_list = []
            pre_dis_total_acc_list = []
            pre_dis_val_acc_list = []

        # 3. Adversarial-train
        whole_adv_dis_total_loss_list, whole_adv_dis_total_acc_list, whole_adv_dis_val_acc_list, \
        whole_pg_loss_list, whole_reward_list \
            = self.train_adversarially()
        torch.save(self.gen.state_dict(), os.path.join(self.save_path, f'adv_trained_gen_{self.failure_type}'))
        torch.save(self.dis.state_dict(), os.path.join(self.save_path, f'adv_trained_dis_{self.failure_type}'))

        loss_acc = {'pre_gen_total_loss_list': pre_gen_total_loss_list,
                    'pre_dis_total_loss_list': pre_dis_total_loss_list,
                    'pre_dis_total_acc_list': pre_dis_total_acc_list,
                    'pre_dis_val_acc_list': pre_dis_val_acc_list,
                    'whole_adv_dis_total_loss_list': whole_adv_dis_total_loss_list,
                    'whole_adv_dis_total_acc_list': whole_adv_dis_total_acc_list,
                    'whole_adv_dis_val_acc_list': whole_adv_dis_val_acc_list,
                    'whole_pg_loss_list': whole_pg_loss_list,
                    'whole_reward_list': whole_reward_list
                    }
        self.save(loss_acc)

    def generate_data(self, num_of_samples: int, label_idx: int) -> np.array:
        sample_unit = 1000
        generated_data = []
        for _ in range(int(num_of_samples / sample_unit)):
            start_letters = self.choose_start_letter(sample_unit)
            labels = self.generate_specific_labels(sample_unit, label_idx)
            generated_data_batch, _ = self.gen.sample(sample_unit, start_letters, labels)
            generated_data.append(generated_data_batch.to(torch.device('cpu')).numpy())
        generated_data = np.vstack(generated_data)

        assert generated_data.shape[0] == num_of_samples and generated_data.shape[1] == self.MAX_SEQ_LEN
        return generated_data
