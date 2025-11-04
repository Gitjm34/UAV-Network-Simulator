import util_config

from pathlib import Path
import numpy as np
import os
import pickle
import dpkt
import struct
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class mavlink2(object):
    def __init__(self, pkt_buf):
        msg_len = pkt_buf[1]
        unpacker_format = f'BBBBBBB3s{msg_len}s2s'

        unpacked = struct.unpack(unpacker_format, pkt_buf)
        self.magic = unpacked[0]
        self.len = unpacked[1]
        self.incompat_flags = unpacked[2]
        self.compat_flags = unpacked[3]
        self.seq = unpacked[4]
        self.sysid = unpacked[5]
        self.compid = unpacked[6]

        self.msgid = bytearray(unpacked[7])
        self.msgid.reverse()
        self.msgid = f'0x{self.msgid.hex()}'

        self.msg = unpacked[8]

        self.checksum = bytearray(unpacked[9])
        self.checksum.reverse()
        self.checksum = f'0x{self.checksum.hex()}'

        assert len(self.msg) == self.len


def _load_and_read_pcap(file_name):
    src_file = open(file_name, 'rb')

    if file_name.find('.pcapng') >= 0:
        read_instance = dpkt.pcapng.Reader(src_file)
    elif file_name.find('.pcap') >= 0:
        read_instance = dpkt.pcap.Reader(src_file)
    else:
        raise NotImplementedError

    return read_instance


# 0. It reads PCAP files, extracts message IDs, and builds npy files.
def get_sequences(filename):
    filename = Path(filename)
    filename_out = Path(str(filename)[:-7] + '_sequences.npy')
    if filename_out.exists():
        return

    reader = _load_and_read_pcap(str(filename))
    sequences = list()
    for packet_idx, (ts, buf) in enumerate(reader):
        ether_level = dpkt.ethernet.Ethernet(buf)
        if ether_level.type != dpkt.ethernet.ETH_TYPE_IP:  # IPv4 check
            continue
        ip_level = ether_level.data
        if ip_level.p != dpkt.ip.IP_PROTO_UDP:  # UDP check
            continue
        udp_level = ip_level.data
        """
        if hex(udp_level.data_hitl[0]) != '0xfd':  # MAVLink 2 check
            continue
        """
        ml2 = mavlink2(udp_level.data)
        sequences.append(int(ml2.msgid, 16))

    sequences = np.array(sequences)
    print(filename_out)
    print(sequences.shape)
    np.save(str(filename_out), sequences)


def build_msg_id_converter() -> None:
    path_config = util_config.get_path_config()
    npy_dir_path = path_config['raw_data']

    # Data load
    unique_msg_id_list = []
    labels = ['normal', 'heartbeat', 'ping', 'request']
    for label in labels:
        file_names = [f'{label}(filtered)_sequences.npy', f'hitl_100000_{label}_sequences.npy']
        for file_name in file_names:
            npy_file_path = os.path.join(npy_dir_path, file_name.format(label))
            npy_arr = np.load(npy_file_path)

            unique_msg_id_list.append(npy_arr)

    # Building message ID converter
    unique_msg_id_list, counts = np.unique(np.hstack(unique_msg_id_list), return_counts=True)
    print(unique_msg_id_list)
    print(counts)
    msg_id_to_int_converter = {}
    int_to_msg_id_converter = {}
    for idx, msg_id in enumerate(unique_msg_id_list):
        msg_id_to_int_converter[msg_id] = idx
        int_to_msg_id_converter[idx] = msg_id
    util_config.make_dirs(path_config['msg_id_to_int_converter'])
    util_config.make_dirs(path_config['int_to_msg_id_converter'])
    with open(path_config['msg_id_to_int_converter'], 'wb') as file:
        pickle.dump(msg_id_to_int_converter, file)
    with open(path_config['int_to_msg_id_converter'], 'wb') as file:
        pickle.dump(int_to_msg_id_converter, file)


def build_msg_id_frequency() -> None:
    path_config = util_config.get_path_config()
    param_config = util_config.get_param_config()

    our_labels = param_config['full_label_list']
    dir_path = path_config['raw_data']
    data_dict = []

    # Data load
    for label in our_labels:
        for _file_path in Path(dir_path).glob(f'*{label}_sequences.npy'):
            file_path = str(_file_path)
            print(file_path)
            np_arr = np.load(file_path)
            data_dict.append(np_arr)

        for _file_path in Path(dir_path).glob(f'*{label}2_sequences.npy'):
            file_path = str(_file_path)
            print(file_path)
            np_arr = np.load(file_path)
            data_dict.append(np_arr)

        for _file_path in Path(dir_path).glob(f'*{label}(filtered)_sequences.npy'):
            file_path = str(_file_path)
            print(file_path)
            np_arr = np.load(file_path)
            data_dict.append(np_arr)

    data_dict = np.hstack(data_dict)
    print(f'Whole sequence length: {data_dict.shape}')

    unique_id_list, counts = np.unique(data_dict, return_counts=True)
    print(unique_id_list)
    print(counts)
    frequency = counts / counts.sum()
    np.save(path_config['msg_id_frequency'], frequency)

    # Load message ID converter
    with open(path_config['msg_id_to_int_converter'], 'rb') as file:
        msg_id_to_int_converter = pickle.load(file)
    print(msg_id_to_int_converter)


def _filter_unique(_data: np.ndarray) -> np.ndarray:
    _unique_seq_list = set()
    for _seq in _data:
        seq = str(list(_seq))
        _unique_seq_list.add(seq)
    _unique_seq_arr = [np.array(eval(x)) for x in _unique_seq_list]
    return np.array(_unique_seq_arr)


def split_msg_id_seq(sequence: np.ndarray, length: int, label: int) -> (np.ndarray, np.ndarray):
    assert length > 0 and label >= 0
    tg = TimeseriesGenerator(sequence, [label, ]*sequence.shape[0], length=length, batch_size=sequence.shape[0])
    x, y = next(iter(tg))
    return x, y


# 1. This builds fixed-sized sequences of message IDs from npy files built by get_sequences() above. (for old dataset)
def prepare_data(seq_len: int):
    def __convert_seq(target_seq: np.ndarray, converter: dict) -> np.ndarray:
        converted_seq = []
        for _msg_id in target_seq:
            converted_seq.append(converter[_msg_id])
        return np.array(converted_seq)

    def __export_npy(target_npy_arr: np.ndarray, data_type: str, target_label: int) -> None:
        if target_label == 0:
            label_str = 'normal'
        elif target_label == 1:
            label_str = 'heartbeat'
        elif target_label == 2:
            label_str = 'ping'
        elif target_label == 3:
            label_str = 'request'
        else:
            raise NotImplementedError

        export_path = os.path.join(
            path_config['preprocessed_file_dir_template'].format(seq_len),
            path_config['msg_id_file_name_template'].format(data_type, label_str, ''))

        util_config.make_dirs(export_path)
        np.save(export_path, target_npy_arr)
        desc_log.append(f'{export_path} saved with {target_npy_arr.shape}; label:({target_label})')

    def __preprocess_seq(target_seq: np.ndarray, lab: int):
        selected_interval = __convert_seq(target_seq, msg_id_to_int_converter)
        _x, _ = split_msg_id_seq(selected_interval, seq_len, lab)
        return _x

    # np.random.seed(1)
    path_config = util_config.get_path_config()
    param_config = util_config.get_param_config()

    our_labels = param_config['full_label_list']
    dir_path = path_config['raw_data']
    data_dict = {}

    # Data load
    for label in our_labels:
        data_dict[label] = []
        for _file_path in Path(dir_path).glob(f'hitl*{label}_sequences.npy'):
            file_path = str(_file_path)
            print(file_path)
            np_arr = np.load(file_path)
            data_dict[label].append(np_arr)

        for _file_path in Path(dir_path).glob(f'hitl*{label}2_sequences.npy'):
            file_path = str(_file_path)
            print(file_path)
            np_arr = np.load(file_path)
            data_dict[label].append(np_arr)

    # Load message ID converter
    with open(path_config['msg_id_to_int_converter'], 'rb') as file:
        msg_id_to_int_converter = pickle.load(file)
    print(msg_id_to_int_converter)

    # Building dataset
    desc_log = []
    msg_id_list = [[], [], [], []]

    for label in data_dict.keys():
        if label == 'normal':
            label_int = 0
        elif label == 'heartbeat':
            label_int = 1
        elif label == 'ping':
            label_int = 2
        elif label == 'request':
            label_int = 3
        else:
            raise NotImplementedError

        assert len(data_dict[label]) == 2
        # Inference data
        x = __preprocess_seq(data_dict[label][0], label_int)
        msg_id_list[label_int].append(x)
        x = __preprocess_seq(data_dict[label][1], label_int)
        msg_id_list[label_int].append(x)

    # From list to numpy array; train test split; save
    for label in range(len(msg_id_list)):
        msg_id_list[label] = np.vstack(msg_id_list[label])
        train_x, inf_x = train_test_split(msg_id_list[label], test_size=0.2,
                                          random_state=777, shuffle=True)
        train_x = _filter_unique(train_x)

        __export_npy(train_x, 'train', label)
        __export_npy(inf_x, 'inf', label)

    # Logging
    log_path = os.path.join(path_config['preprocessed_file_dir_template'].format(seq_len),
                            'data_info.txt')
    with open(log_path, 'w') as file:
        for log_str in desc_log:
            file.write(log_str)
            file.write('\n')

    return


# 1-1. This is used to build new datasets.
def prepare_2nd_inference(seq_len: int):
    def __convert_seq(target_seq: np.ndarray, converter: dict) -> np.ndarray:
        converted_seq = []
        for _msg_id in target_seq:
            converted_seq.append(converter[_msg_id])

        return np.array(converted_seq)

    def __export_npy(target_npy_arr: np.ndarray, data_type: str, target_label: int) -> None:
        if target_label == 0:
            str_label = 'normal'
        elif target_label == 1:
            str_label = 'heartbeat'
        elif target_label == 2:
            str_label = 'ping'
        elif target_label == 3:
            str_label = 'request'

        # New attack types
        elif target_label == 4:
            str_label = 'suspension'
        elif target_label == 5:
            str_label = 'corruption'
        else:
            str_label = str(target_label)

        export_path = os.path.join(
            path_config['preprocessed_file_dir_template'].format(seq_len),
            path_config['msg_id_file_name_template'].format(data_type, str_label, '_2'))
        util_config.make_dirs(export_path)
        np.save(export_path, target_npy_arr)
        desc_log.append(f'{export_path} saved with {target_npy_arr.shape}; label:({target_label})')

    def __convert_and_split(interval: np.ndarray, lab: int):
        selected_interval = __convert_seq(interval, msg_id_to_int_converter)
        _x, _ = split_msg_id_seq(selected_interval, seq_len, lab)
        return _x

    path_config = util_config.get_path_config()

    dir_path = path_config['raw_data']
    data_dict = {}
    for label in ['normal', 'heartbeat', 'ping', 'request', 'suspension', 'corruption']:
        for _file_path in Path(dir_path).glob(f'{label}(filtered)_sequences.npy'):
            file_path = str(_file_path)
            print(file_path)
            data_dict[label] = np.load(file_path)

    # Load message ID converter
    with open(path_config['msg_id_to_int_converter'], 'rb') as file:
        msg_id_to_int_converter = pickle.load(file)
    print(msg_id_to_int_converter)

    desc_log = []

    # Building dataset
    inf_msg_id_list = [[], [], [], [], [], [], [], [], []]

    for label in data_dict.keys():
        if label == 'normal':
            label_int = 0
        elif label == 'heartbeat':
            label_int = 1
        elif label == 'ping':
            label_int = 2
        elif label == 'request':
            label_int = 3

        # New attack types
        elif label == 'suspension':
            label_int = 4
        elif label == 'corruption':
            label_int = 5

        else:
            raise NotImplementedError

        # Inference data
        x = __convert_and_split(data_dict[label], label_int)
        inf_msg_id_list[label_int].append(x)

    # From list to numpy array
    for label in range(len(inf_msg_id_list)):
        if len(inf_msg_id_list[label]) == 0:
            continue

        inf_msg_id_list[label] = np.vstack(inf_msg_id_list[label])
        inf_msg_id_list[label] = _filter_unique(inf_msg_id_list[label])

    # Save
    for label in range(len(inf_msg_id_list)):
        if len(inf_msg_id_list[label]) == 0:
            continue
        __export_npy(inf_msg_id_list[label], 'inf', label)

    # Logging
    log_path = os.path.join(path_config['preprocessed_file_dir_template'].format(seq_len),
                            'data_info2.txt')
    with open(log_path, 'a') as file:
        for log_str in desc_log:
            file.write(log_str)
            file.write('\n')

    return


# 2. This loads training or test data.
def load_data(seq_len: int, label_info_list: list,
              exp_type: str, exp_id: str, gan_epoch_idx: int = -1) -> tuple:
    path_config = util_config.get_path_config()
    param_config = util_config.get_param_config()

    train_data, train_label = [], []
    inf_data, inf_label = [], []
    inf_data2, inf_label2 = [], []

    for data_type in ['train', 'inf', 'inf2']:
        if data_type == 'train':
            data_list, labels = train_data, train_label
        elif data_type == 'inf':
            data_list, labels = inf_data, inf_label
        elif data_type == 'inf2':
            data_list, labels = inf_data2, inf_label2
        else:
            raise NotImplementedError

        for label_info in label_info_list:
            label_str, label_int, is_used, is_augmented = label_info
            if not is_used:
                continue

            # Augmented Data
            if data_type == 'train' and is_augmented and exp_type != 'before_aug_none' and exp_type != 'before_aug_wr':
                if exp_type in ['gan_aug', 'gan_aug_gen_only']:
                    raise NotImplementedError
                elif exp_type in ['seqgan', 'maskgan', 'rankgan', 'stepgan', 'leakgan']:
                    aug_data_type = param_config['seqgan']['gan_type_template'].format(
                        param_config['seqgan']['unrolled_step'],
                        f"{param_config['seqgan']['is_wasserstein']}{exp_id}")
                    additional_info = f'{param_config[exp_type]["num_of_generated_data"]}({gan_epoch_idx})'
                elif exp_type in ['oversampling']:
                    aug_data_type = 'oversampling'
                    additional_info = ''
                elif exp_type in ['undersampling']:
                    aug_data_type = 'undersampling'
                    additional_info = ''
                elif exp_type in ['random']:
                    aug_data_type = 'random'
                    additional_info = ''
                elif exp_type in ['noise']:
                    aug_data_type = 'noise'
                    additional_info = ''
                else:
                    raise NotImplementedError

                aug_dir_path = path_config['augmented_file_dir_template'].format(seq_len, aug_data_type)
                print(f'Augmented data directory: {aug_dir_path}')
                a = np.load(os.path.join(aug_dir_path,
                                         path_config['msg_id_file_name_template'].format(data_type, label_str,
                                                                                         additional_info)))
            elif data_type == 'train':
                dir_path = path_config['preprocessed_file_dir_template'].format(seq_len)
                a = np.load(os.path.join(dir_path,
                                         path_config['msg_id_file_name_template'].format(data_type, label_str, '')))
            elif data_type == 'inf':
                dir_path = path_config['preprocessed_file_dir_template'].format(seq_len)
                a = np.load(os.path.join(dir_path,
                                         path_config['msg_id_file_name_template'].format(data_type, label_str, '')))
            elif data_type == 'inf2':
                dir_path = path_config['preprocessed_file_dir_template'].format(seq_len)
                a = np.load(os.path.join(dir_path,
                                         path_config['msg_id_file_name_template'].format('inf', label_str, '_2')))
            else:
                raise NotImplementedError

            b = np.array([float(label_int)] * len(a))
            data_list.append(a)
            labels.append(b)

    new_inf_dict = {}
    for label_str in ['suspension', 'corruption']:
        new_inf_dict[label_str] = {}
        new_inf_dict[label_str]['data'] = []
        new_inf_dict[label_str]['label'] = []

        dir_path = path_config['preprocessed_file_dir_template'].format(seq_len)
        a = np.load(os.path.join(dir_path,
                                 path_config['msg_id_file_name_template'].format('inf', label_str, '_2')))
        label_int = 1

        b = np.array([float(label_int)] * len(a))
        new_inf_dict[label_str]['data'].append(a)
        new_inf_dict[label_str]['label'].append(b)

    train_data = np.vstack(train_data)
    train_label = np.hstack(train_label)
    inf_data = np.vstack(inf_data)
    inf_label = np.hstack(inf_label)
    inf_data2 = np.vstack(inf_data2)
    inf_label2 = np.hstack(inf_label2)
    for label_str in new_inf_dict.keys():
        new_inf_dict[label_str]['data'] = np.vstack(new_inf_dict[label_str]['data'])
        new_inf_dict[label_str]['label'] = np.hstack(new_inf_dict[label_str]['label'])

    return train_data, train_label, inf_data, inf_label, inf_data2, inf_label2, new_inf_dict


if __name__ == '__main__':
    """
    for path in Path('../data/raw').glob('.pcapng'):
        get_sequences(path)
    """

    build_msg_id_converter()
    build_msg_id_frequency()

    prepare_data(128)
    prepare_2nd_inference(128)

    pass
