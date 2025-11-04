import util_config

import pickle
import numpy as np
import os
from typing import Final

_param_config = util_config.CONFIG.get_param_config()
_path_config = util_config.CONFIG.get_path_config()

with open(_path_config['msg_id_to_int_converter'], 'rb') as _pickle_file:
    MSG_ID_TO_INT_CONVERTER: Final[dict] = pickle.load(_pickle_file)
with open(_path_config['int_to_msg_id_converter'], 'rb') as _pickle_file:
    INT_TO_MSG_ID_CONVERTER: Final[dict] = pickle.load(_pickle_file)

del _path_config
del _param_config


def __convert_msg_id_to_int(original_msg_id: int) -> int:
    return MSG_ID_TO_INT_CONVERTER[original_msg_id]


def convert_msg_id_to_int(original_msg_id: int):
    return np.vectorize(__convert_msg_id_to_int)(original_msg_id)


def __convert_int_to_msg_id(integered_msg_id: int) -> int:
    return INT_TO_MSG_ID_CONVERTER[integered_msg_id]


def convert_int_to_msg_id(integered_msg_id: int):
    return np.vectorize(__convert_int_to_msg_id)(integered_msg_id)


def transform_npy_seq_to_str(sequence: np.ndarray) -> str:
    return ' '.join(sequence.astype('str'))


def transform_str_seq_to_npy(sequence: str) -> np.ndarray:
    return np.array(sequence.split(' ')).astype('int')


def load_duplicated_data(shortage_ratio: float, seq_len: int, data_type: str, label: str = 'heartbeat'):
    param_config = util_config.get_param_config()
    path_config = util_config.get_path_config()

    aug_data_type = 'duplicated_data'
    additional_info = f'({param_config["data_augmentation"]["duplication_iteration"]})'
    aug_dir_path = path_config['augmented_file_dir_template'].format(shortage_ratio, seq_len,
                                                                          aug_data_type)

    a = np.load(os.path.join(aug_dir_path,
                             path_config['msg_id_file_name_template'].format(data_type, label,
                                                                                  additional_info)))

    return a


def load_noise_data(shortage_ratio: float, seq_len: int, data_type: str, label: str = 'heartbeat',
                    _noise_ratio: float = -1.0):
    param_config = util_config.get_param_config()
    path_config = util_config.get_path_config()

    aug_data_type = 'noise_data'
    if _noise_ratio == -1.0:
        noise_ratio = param_config["data_augmentation"]["noise_ratio"]
    else:
        noise_ratio = _noise_ratio
    additional_info = f'({noise_ratio}_{param_config["data_augmentation"]["noise_iteration"]})'
    aug_dir_path = path_config['augmented_file_dir_template'].format(shortage_ratio, seq_len,
                                                                          aug_data_type)

    a = np.load(os.path.join(aug_dir_path,
                             path_config['msg_id_file_name_template'].format(data_type, label,
                                                                                  additional_info)))

    return a


def load_random_data(shortage_ratio: float, seq_len: int, data_type: str, label: str = 'heartbeat'):
    param_config = util_config.get_param_config()
    path_config = util_config.get_path_config()

    aug_data_type = 'random_data'
    additional_info = f'({param_config["data_augmentation"]["num_of_random_data"]})'
    aug_dir_path = path_config['augmented_file_dir_template'].format(shortage_ratio, seq_len,
                                                                          aug_data_type)

    a = np.load(os.path.join(aug_dir_path,
                             path_config['msg_id_file_name_template'].format(data_type, label,
                                                                                  additional_info)))

    return a
