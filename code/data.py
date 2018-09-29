import sys
import torch
import torch.utils.data as data
import tables
import random
import numpy as np
import pickle

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
use_cuda = torch.cuda.is_available()

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class CodeSearchDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """

    def __init__(self, data_dir, conf, dataset):

        if dataset == "train":
            f_qt, qt_len, f_code, code_len, f_qb, qb_len = conf['train_qt'], conf['qt_len'], conf['train_code'], \
                                                           conf['code_len'], conf['train_qb'], conf['qb_len']
        elif dataset == "val":
            f_qt, qt_len, f_code, code_len, f_qb, qb_len = conf['val_qt'], conf['qt_len'], conf['val_code'], \
                                                           conf['code_len'], conf['val_qb'], conf['qb_len']
        else:
            f_qt, qt_len, f_code, code_len, f_qb, qb_len = conf['test_qt'], conf['qt_len'], conf['test_code'], \
                                                           conf['code_len'], conf['test_qb'], conf['qb_len']

        self.qt_len = qt_len
        self.code_len = code_len
        self.qb_len = qb_len

        # Load Vocab Files
        self.path = conf['workdir']
        self.vocab_qt = limit_vocab(load_dict(self.path + conf['vocab_qt']), conf['qt_n_words'])
        self.vocab_code = limit_vocab(load_dict(self.path + conf['vocab_code']), conf['code_n_words'])
        self.vocab_qb = limit_vocab(load_dict(self.path + conf['vocab_qb']), conf['qb_n_words'])

        # Processing Text
        # 1. Load Preprocessed Dataset
        print("loading data...")
        self.list_of_qt_strings = pickle.load(open(data_dir + f_qt))
        self.list_of_qb_strings = pickle.load(open(data_dir + f_qb))
        self.list_of_code_strings = pickle.load(open(data_dir + f_code))

        assert len(self.list_of_qt_strings) == len(self.list_of_qb_strings)

        # Convert string of indices to list of indices
        self.processed_qt = self.get_preprocessed_text(self.list_of_qt_strings, self.vocab_qt, dataset)
        self.processed_qb = self.get_preprocessed_text(self.list_of_qb_strings, self.vocab_qb, dataset)
        self.processed_code = self.get_preprocessed_text(self.list_of_code_strings, self.vocab_code, dataset)

        # Data Length
        self.data_len = len(self.processed_qt)
        print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_token] * maxlen)
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
        return seq

    def get_preprocessed_text(self, list_of_strings, vocab, dataset):
        preprocessed_text = []
        strings = all(isinstance(n, str) for n in list_of_strings)
        for data_point in list_of_strings:
            if strings:
                string = data_point
            else:
                string = data_point[1]
            words = map(int, string.split())
            preprocessed_text.append(np.array(words))
        return preprocessed_text

    def __getitem__(self, offset):
        # Question Title
        qt = self.processed_qt[offset].astype('int64')
        qt = self.pad_seq(qt, self.qt_len)

        # < QT,Code,QB>
        good_code = self.processed_code[offset].astype('int64')
        good_code = self.pad_seq(good_code, self.code_len)

        good_qb = self.processed_qb[offset].astype('int64')
        good_qb = self.pad_seq(good_qb, self.qb_len)

        # < QT,~Code,~QB>
        rand_offset = random.randint(0, self.data_len - 1)
        while rand_offset == offset:
            rand_offset = random.randint(0, self.data_len - 1)

        bad_code = self.processed_code[rand_offset].astype('int64')
        bad_code = self.pad_seq(bad_code, self.code_len)

        bad_qb = self.processed_qb[rand_offset].astype('int64')
        bad_qb = self.pad_seq(bad_qb, self.qb_len)

        return qt, good_code, bad_code, good_qb, bad_qb

    def __len__(self):
        return self.data_len


class CodennDataset(data.Dataset):
    """
    Dataset that has only positive samples.
    """

    def __init__(self, data_dir, conf, dataset):
        if dataset == "val":
            f_qt, qt_len, f_code, code_len, f_qb, qb_len = conf['val_qt'], conf['qt_len'], conf['val_code'], \
                                                           conf['code_len'], conf['val_qb'], conf['qb_len']
        else:
            f_qt, qt_len, f_code, code_len, f_qb, qb_len = conf['test_qt'], conf['qt_len'], conf['test_code'], \
                                                           conf['code_len'], conf['test_qb'], conf['qb_len']

        self.qt_len = qt_len
        self.code_len = code_len
        self.qb_len = qb_len

        # Load Vocab Files
        self.path = conf['workdir']
        self.vocab_qt = limit_vocab(load_dict(self.path + conf['vocab_qt']), conf['qt_n_words'])
        self.vocab_code = limit_vocab(load_dict(self.path + conf['vocab_code']), conf['code_n_words'])
        self.vocab_qb = limit_vocab(load_dict(self.path + conf['vocab_qb']), conf['qb_n_words'])

        # Pre-processing Step
        # 1. Load Preprocessed Dataset
        print("loading data...")
        self.list_of_qt_strings = pickle.load(open(data_dir + f_qt))
        self.list_of_qb_strings = pickle.load(open(data_dir + f_qb))
        self.list_of_code_strings = pickle.load(open(data_dir + f_code))

        assert len(self.list_of_qt_strings) == len(self.list_of_qb_strings)

        # Convert string of indices to list of indices
        self.processed_qt = self.get_preprocessed_text(self.list_of_qt_strings, self.vocab_qt, dataset)
        self.processed_qb = self.get_preprocessed_text(self.list_of_qb_strings, self.vocab_qb, dataset)
        self.processed_code = self.get_preprocessed_text(self.list_of_code_strings, self.vocab_code, dataset)

        # Data Length
        self.data_len = len(self.processed_qt)
        print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_token] * maxlen)
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
        return seq

    def get_preprocessed_text(self, list_of_strings, vocab, dataset):
        preprocessed_text = []
        strings = all(isinstance(n, str) for n in list_of_strings)
        for data_point in list_of_strings:
            if strings:
                string = data_point
            else:
                string = data_point[1]
            words = map(int, string.split())
            preprocessed_text.append(np.array(words))
        return preprocessed_text

    def __getitem__(self, offset):
        # Question Title
        qt = self.processed_qt[offset].astype('int64')
        qt = self.pad_seq(qt, self.qt_len)

        # < QT,Code,QB>
        code = self.processed_code[offset].astype('int64')
        code = self.pad_seq(code, self.code_len)

        qb = self.processed_qb[offset].astype('int64')
        qb = self.pad_seq(qb, self.qb_len)

        return qt, code, qb

    def __len__(self):
        return self.data_len


def load_dict(filename):
    return pickle.load(open(filename, 'rb'))


def limit_vocab(old_vocab_dict, vocab_length):
    new_vocab_dict = {}
    for word, key in old_vocab_dict.iteritems():
        if key < vocab_length:
            new_vocab_dict[word] = key
    return new_vocab_dict