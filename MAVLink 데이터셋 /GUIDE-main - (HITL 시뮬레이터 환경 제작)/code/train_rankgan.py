from rankgan.model import RankGAN
from rankgan.dataset import MSGIDSequence

import util_config
import util_data
import data_preparation

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_rankgan():
    param_config = util_config.get_param_config()
    label_info_list = param_config['label_info_list']

    train_data, train_label, _, _, _, _, _ \
        = data_preparation.load_data(128, label_info_list, 'before_aug', '', -1)

    print(f'Train Data Info.')
    print(f'Attack-free: {(train_label == 0).sum()}')
    print(f'Heartbeat: {(train_label == 1).sum()}')
    print(f'Ping: {(train_label == 2).sum()}')
    print(f'Request: {(train_label == 3).sum()}')

    train_dataset = MSGIDSequence(train_data, DEVICE)

    n_vocabs = len(util_data.INT_TO_MSG_ID_CONVERTER)
    batch_size = 512
    ref_batch_size = 16
    gen_hidden_size = 64
    dis_hidden_size = 64
    seq_len = 128
    gen_lr = 0.0001
    ran_lr = 0.001
    save_path = ''
    n_rollouts = 4

    msg_set, msg_id_counts = np.unique(train_data, return_counts=True)
    msg_id_prob_dist = msg_id_counts / msg_id_counts.sum()

    print(msg_set)
    print(msg_id_prob_dist)

    gen_pre_epochs = 1
    dis_pre_epochs = 1
    adv_epochs = 1

    rank_gan = RankGAN(train_dataset, n_vocabs, batch_size, ref_batch_size,
                       gen_hidden_size, dis_hidden_size, n_rollouts, seq_len,
                       gen_lr, ran_lr, save_path,
                       msg_set, msg_id_prob_dist, DEVICE)

    rank_gan.train(gen_pre_epochs, dis_pre_epochs, adv_epochs)

    return


if __name__ == '__main__':
    run_rankgan()
