import data_preparation
import util_data
import util_config
import seqgan_model

from itertools import product
import torch
import os
import numpy as np

util_config.CONFIG.reload_param_config()
util_config.CONFIG.reload_path_config()
param_config = util_config.CONFIG.get_param_config()
path_config = util_config.CONFIG.get_path_config()

failure_type_list = param_config['(our)label_list']
num_of_msg = len(util_data.INT_TO_MSG_ID_CONVERTER)
num_of_labels = len(failure_type_list)
msg_id_frequency = np.load(path_config['msg_id_frequency'])

CUDA = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

seq_len = 128
for failure_type in failure_type_list:
    train_data, train_label, _, _, _, _, _ \
        = data_preparation.load_data(seq_len, [failure_type], 'before_aug_none', '', -1)

    msg = ''
    msg += '======= Info. =======\n'
    msg += f'Sequence Length: {seq_len}\n'
    msg += f'CUDA: {CUDA}\n'
    msg += f'# of messages: {num_of_msg}\n'
    msg += f'# of labels: {num_of_labels}\n'
    msg += '\n'
    prefix = 'train'
    msg += f'{prefix} X shape: {train_data.shape}\n'
    msg += f'{prefix} y shape: {train_label.shape}\n'
    msg += str((train_label == 0.0).sum()) + ' '
    msg += str((train_label == 1.0).sum()) + ' '
    msg += str((train_label == 2.0).sum()) + ' '
    msg += str((train_label == 3.0).sum()) + ' '
    msg += str((train_label == 4.0).sum()) + ' '
    msg += '\n'
    print(msg)

    if CUDA:
        train_label = torch.Tensor(train_label).float().cuda()

    perm = np.random.permutation(train_data.shape[0])
    train_data = train_data[perm]
    train_label = train_label[perm]
    print(train_label)

    gan_type_template = param_config['seqgan']['gan_type_template']
    gan_type = gan_type_template.format(param_config['seqgan']['unrolled_step'],
                                        param_config['seqgan']['is_wasserstein'])
    seqgan_save_path = path_config['seqgan_model_save_dir_path_template'].format(f'{seq_len}', gan_type)
    util_config.make_dirs(seqgan_save_path)
    model = seqgan_model.SeqGAN(train_data, train_label, util_data.INT_TO_MSG_ID_CONVERTER, msg_id_frequency, seq_len,
                                seqgan_save_path, CUDA)
    model.train()


def generate_data(epoch_idx: int, label_idx: int):
    assert epoch_idx > 0
    util_config.CONFIG.reload_param_config()
    util_config.CONFIG.reload_path_config()
    param_config = util_config.CONFIG.get_param_config()
    path_config = util_config.CONFIG.get_path_config()

    failure_type_list = param_config['(our)label_list']

    CUDA = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    seq_len = 128
    for failure_type in failure_type_list:
        train_data, train_label, _, _, _, _, _ \
            = data_preparation.load_data(seq_len, [failure_type], 'before_aug_none', '', -1)

        gan_type = param_config['seqgan']['gan_type_template'].format(param_config['seqgan']['unrolled_step'],
                                                                      param_config['seqgan']['is_wasserstein'])
        seqgan_save_path = path_config['seqgan_model_save_dir_path_template'].format(f'{seq_len}', gan_type)
        model = seqgan_model.SeqGAN(train_data, train_label, util_data.INT_TO_MSG_ID_CONVERTER, msg_id_frequency,
                                    seq_len, seqgan_save_path, CUDA)

        # Generator load
        seqgan_generator_save_file_name = path_config['seqgan_generator_save_file_name'].format(failure_type, epoch_idx)
        generator_save_path = os.path.join(seqgan_save_path, seqgan_generator_save_file_name)
        model.gen.load_state_dict(torch.load(generator_save_path))
        print(f'{seqgan_generator_save_file_name} has been loaded.')

        # Data generation
        generated_data = model.generate_data(param_config['seqgan']['num_of_generated_data'], label_idx)
        print(f'# of generated data: {len(generated_data)}')

        # Generated data save
        additional_info = f'({epoch_idx})'
        data_type = f'{failure_type}_{len(generated_data)}{additional_info}.npy'
        generated_data_path = path_config['seqgan_generated_data_file_path_template'].format(seq_len, gan_type, data_type)
        util_config.make_dirs(generated_data_path)
        np.save(generated_data_path, generated_data)


for _idx in range(50, 2050, 50):
    # generate_data(_idx - 1, 0)
    pass
