import util_config

import numpy as np
import os


def __load_real_data(seq_len: int, target_failure_type: str) -> np.ndarray:
    path_config = util_config.CONFIG.get_path_config()

    # Real data 로드
    real_data_path = os.path.join(path_config['preprocessed_file_dir_template'].format(seq_len),
                                  path_config['msg_id_file_name_template'].format('train', target_failure_type, ''))
    real_data = np.load(real_data_path)

    return real_data


def build_augmented_data(seq_len: int, target_failure_type: str, epoch_idx: int, gan_model: str) -> None:
    param_config = util_config.CONFIG.get_param_config()
    path_config = util_config.CONFIG.get_path_config()

    # Load real data
    real_data = __load_real_data(seq_len, target_failure_type)

    # Load generated data
    gan_type = util_config.get_gan_type()
    gan_type += param_config['exp_id']
    additional_info = f'{param_config[gan_model]["num_of_generated_data"]}({epoch_idx})'  # TODO

    data_type = f'{target_failure_type}_{additional_info}.npy'
    generated_data_path = path_config['leakgan_generated_data_file_path_template'].format(seq_len, gan_type, data_type)

    print(f'Generated data path: {generated_data_path}')
    generated_data = np.load(generated_data_path)
    augmented_data = np.vstack([real_data, generated_data])
    augmented_data_path \
        = os.path.join(path_config['augmented_file_dir_template'].format(seq_len, gan_type),
                       path_config['msg_id_file_name_template'].format('train', target_failure_type, additional_info))
    util_config.make_dirs(augmented_data_path)
    np.save(augmented_data_path, augmented_data)
    print(augmented_data_path, augmented_data.shape)

    return


def build_oversampling_data(seq_len: int, target_failure_type: str, added_ratio: float):
    path_config = util_config.CONFIG.get_path_config()

    # Real data 로드
    real_data = __load_real_data(seq_len, target_failure_type)

    n_added = int(real_data.shape[0] * added_ratio)
    random_indices = np.random.choice(real_data.shape[0], n_added, replace=False)

    print(f'Before augmentation: {real_data.shape}')
    augmented_data = np.vstack([real_data, real_data[random_indices, :]])
    print(f'After augmentation: {augmented_data.shape}')
    augmented_data_path \
        = os.path.join(path_config['augmented_file_dir_template'].format(seq_len, 'oversampling'),
                       path_config['msg_id_file_name_template'].format('train', target_failure_type, ''))
    util_config.make_dirs(augmented_data_path)
    np.save(augmented_data_path, augmented_data)
    print(augmented_data_path, augmented_data.shape)

    return


def build_undersampling_data(seq_len: int, target_failure_type: str, use_ratio: float):
    path_config = util_config.CONFIG.get_path_config()

    # Real data 로드
    real_data = __load_real_data(seq_len, target_failure_type)

    n_use = int(real_data.shape[0] * use_ratio)
    random_indices = np.random.choice(real_data.shape[0], n_use, replace=False)

    print(f'Before undersampling: {real_data.shape}')
    augmented_data = real_data[random_indices, :]
    print(f'After undersampling: {augmented_data.shape}')
    augmented_data_path \
        = os.path.join(path_config['augmented_file_dir_template'].format(seq_len, 'undersampling'),
                       path_config['msg_id_file_name_template'].format('train', target_failure_type, ''))
    util_config.make_dirs(augmented_data_path)
    np.save(augmented_data_path, augmented_data)
    print(augmented_data_path, augmented_data.shape)

    return


def build_random_data(seq_len: int, target_failure_type: str, added_ratio: float):
    path_config = util_config.CONFIG.get_path_config()

    # Real data 로드
    real_data = __load_real_data(seq_len, target_failure_type)

    msg_id, msg_id_freq = np.unique(real_data, return_counts=True)
    msg_id_freq = msg_id_freq / msg_id_freq.sum()

    n_added = int(real_data.shape[0] * added_ratio)
    generated_data = np.random.choice(msg_id, size=(n_added, seq_len), replace=True, p=msg_id_freq)

    file_name = path_config['msg_id_file_name_template'].format('train', target_failure_type, '')
    generated_data_path = path_config['seqgan_generated_data_file_path_template'].format(seq_len, 'random', file_name)
    util_config.make_dirs(generated_data_path)
    np.save(generated_data_path, generated_data)

    print(f'Before augmentation: {real_data.shape}')
    augmented_data = np.vstack([real_data, generated_data])
    print(f'After augmentation: {augmented_data.shape}')
    augmented_data_path \
        = os.path.join(path_config['augmented_file_dir_template'].format(seq_len, 'random'),
                       path_config['msg_id_file_name_template'].format('train', target_failure_type, ''))
    util_config.make_dirs(augmented_data_path)
    np.save(augmented_data_path, augmented_data)
    print(augmented_data_path, augmented_data.shape)

    return


def build_noise_data(seq_len: int, target_failure_type: str, added_ratio: float):
    path_config = util_config.CONFIG.get_path_config()

    # Real data 로드
    real_data = __load_real_data(seq_len, target_failure_type)

    msg_id, msg_id_freq = np.unique(real_data, return_counts=True)
    msg_id_freq = msg_id_freq / msg_id_freq.sum()

    n_added = int(real_data.shape[0] * added_ratio)
    random_indices = np.random.choice(real_data.shape[0], n_added, replace=False)
    generated_data = real_data[random_indices]

    random_indices = np.random.choice(seq_len, size=int(seq_len * 0.3))
    for _data in generated_data:
        for idx in random_indices:
            random_msg_id = np.random.choice(msg_id, p=msg_id_freq)
            _data[idx] = random_msg_id

    file_name = path_config['msg_id_file_name_template'].format('train', target_failure_type, '')
    generated_data_path = path_config['seqgan_generated_data_file_path_template'].format(seq_len, 'noise', file_name)
    util_config.make_dirs(generated_data_path)
    np.save(generated_data_path, generated_data)

    print(f'Before augmentation: {real_data.shape}')
    augmented_data = np.vstack([real_data, generated_data])
    print(f'After augmentation: {augmented_data.shape}')
    augmented_data_path \
        = os.path.join(path_config['augmented_file_dir_template'].format(seq_len, 'noise'),
                       path_config['msg_id_file_name_template'].format('train', target_failure_type, ''))
    util_config.make_dirs(augmented_data_path)
    np.save(augmented_data_path, augmented_data)
    print(augmented_data_path, augmented_data.shape)

    return


if __name__ == '__main__':
    _param_config = util_config.CONFIG.get_param_config()

    for _seq_len in _param_config['sequence_length_list']:
        # build_augmented_data(_seq_len, 'heartbeat', 2999)
        build_augmented_data(_seq_len, 'normal', 0, 'seqgan_model')
        build_augmented_data(_seq_len, 'heartbeat', 0, 'seqgan_model')
        build_augmented_data(_seq_len, 'ping', 0, 'seqgan_model')
        build_augmented_data(_seq_len, 'request', 0, 'seqgan_model')
        """
        build_noise_data(_seq_len, 'normal', 0.2)
        build_noise_data(_seq_len, 'heartbeat', 0.2)
        build_noise_data(_seq_len, 'ping', 0.2)
        build_noise_data(_seq_len, 'request', 0.2)
        """
        pass
    pass
