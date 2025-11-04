import util_data

MASK_SYMBOL = '<m>'


class MSGIDDictionary:
    def __init__(self):
        self.eos = '<eos>'
        self.mask = MASK_SYMBOL
        self.msg_id_idx = {}
        self.idx_msg_id = {}

        msg_id_converter: dict = util_data.MSG_ID_TO_INT_CONVERTER
        for msg_id in msg_id_converter.keys():
            self.add_symbol(msg_id)

        self.add_symbol(self.eos)
        self.add_symbol(self.mask)

    def get_eos_idx(self) -> int:
        return self.msg_id_idx[self.eos]

    def get_mask_idx(self) -> int:
        return self.msg_id_idx[self.mask]

    def add_symbol(self, symbol_str) -> None:
        if symbol_str in self.msg_id_idx:
            return

        next_idx = len(self.msg_id_idx)
        self.msg_id_idx[symbol_str] = next_idx
        self.idx_msg_id[next_idx] = symbol_str

        assert len(self.msg_id_idx) == len(self.idx_msg_id)

    def get_n_vocabs(self) -> int:
        return len(self.msg_id_idx)


if __name__ == '__main__':
    msg_dict = MSGIDDictionary()
    print(msg_dict.msg_id_idx)
    print(msg_dict.idx_msg_id)
    print(util_data.MSG_ID_TO_INT_CONVERTER)
    print(util_data.INT_TO_MSG_ID_CONVERTER)
