from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class BOWEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, config):
        super(BOWEncoder, self).__init__()
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.config = config

    def forward(self, input, input_lengths=None):
        batch_size, seq_len = input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        embedded = F.dropout(embedded, self.config['bow_dropout'], self.training)  # [batch_size x seq_len x emb_size]
        output_pool = F.max_pool1d(embedded.transpose(1, 2), seq_len).squeeze(2)  # [batch_size x emb_size]
        encoding = F.tanh(output_pool)
        return encoding


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, config, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.config = config

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        for w in self.lstm.parameters():  # initialize the gate weights with orthogonal
            if w.dim() > 1:
                weight_init.xavier_normal_(w)

    def forward(self, input, input_lengths=None):
        batch_size, seq_len = input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        embedded = F.dropout(embedded, self.config['seqenc_dropout'], self.training)
        rnn_output, hidden = self.lstm(embedded)  # out:[b x seq x hid_sz*2](biRNN)
        rnn_output = F.dropout(rnn_output, self.config['seqenc_dropout'], self.training)
        rnn_output[rnn_output == 0.0] = -100.0  # Replace padding by -100 to suppress in max pool
        output_pool = F.max_pool1d(rnn_output.transpose(1, 2), seq_len).squeeze(2)  # [batch_size x hid_size*2]
        encoding = F.tanh(output_pool)
        return encoding


class JointEmbeder(nn.Module):
    def __init__(self, config):
        super(JointEmbeder, self).__init__()
        self.conf = config
        self.margin = config['margin']
        self.qt_encoder = SeqEncoder(config['qt_n_words'], config['emb_size'], config['lstm_dims'], self.conf)

        if self.conf['use_qb']:
            self.qb_encoder = SeqEncoder(config['qb_n_words'], config['emb_size'], config['lstm_dims'], self.conf)

            if self.conf['code_encoder'] == "bilstm":
                self.code_encoder = SeqEncoder(config['code_n_words'], config['emb_size'], config['lstm_dims'],
                                               self.conf)  # Bi-LSTM
                # Fusing Code and QB together
                self.fuse = nn.Linear(2 * config['lstm_dims'] + 2 * config['lstm_dims'], 2 * config['lstm_dims'])
            else:
                self.code_encoder = BOWEncoder(config['code_n_words'], config['emb_size'], self.conf)  # MLP
                # Fusing Code and QB together
                self.fuse = nn.Linear(config['emb_size'] + 2 * config['lstm_dims'], 2 * config['lstm_dims'])
        else:
            if self.conf['code_encoder'] == "bilstm":
                self.code_encoder = SeqEncoder(config['code_n_words'], config['emb_size'], config['lstm_dims'],
                                               self.conf)  # Bi-LSTM
            else:
                self.code_encoder = BOWEncoder(config['code_n_words'], 2 * config['lstm_dims'], self.conf)  # MLP

    def code_encoding(self, code, qb):
        '''
        :param code: processed code
        :param qb: processed qb
        :return: vector representation for code
        Encoded Code representation using both code annotation + code
        '''
        code_repr = self.code_encoder(code)
        if self.conf['use_qb']:
            # print(" Model using QB")
            qb_repr = self.qb_encoder(qb)
            code_repr = self.fuse(torch.cat((code_repr, qb_repr), 1))
        code_repr = F.tanh(code_repr)
        return code_repr

    def qt_encoding(self, qt):
        '''
        Encodes Question title
        :param qt: processed question title
        :return: vector representation for qt
        Encoded Question Title
        '''
        qt_repr = self.qt_encoder(qt)
        return qt_repr

    def forward(self, qt, good_code, bad_code, good_qb, bad_qb):
        good_code_repr = self.code_encoding(good_code, good_qb)
        bad_code_repr = self.code_encoding(bad_code, bad_qb)

        qt_repr = self.qt_encoding(qt)

        good_sim = F.cosine_similarity(qt_repr, good_code_repr)
        bad_sim = F.cosine_similarity(qt_repr, bad_code_repr)

        loss = (self.margin - good_sim + bad_sim).clamp(min=1e-6).mean()
        return loss


class JointEmbederQB(nn.Module):
    def __init__(self, config):
        super(JointEmbederQB, self).__init__()
        self.conf = config
        self.margin = config['margin']

        self.qt_encoder = SeqEncoder(config['qt_n_words'], config['emb_size'], config['lstm_dims'], self.conf)

        if self.conf['use_qb']:
            self.qb_encoder = SeqEncoder(config['qb_n_words'], config['emb_size'], config['lstm_dims'], self.conf)

            self.code_encoder = SeqEncoder(config['code_n_words'], config['emb_size'], config['lstm_dims'],
                                           self.conf)  # Bi-LSTM
            # Fusing QT, Code and QB together
            self.fuse = nn.Linear(2 * config['lstm_dims'] + 4 * config['lstm_dims'], 2 * config['lstm_dims'])
        else:
            self.code_encoder = SeqEncoder(config['code_n_words'], config['emb_size'], config['lstm_dims'],
                                           self.conf)  # Bi-LSTM
            # Fusing QT, Code together
            self.fuse = nn.Linear(2 * config['lstm_dims'] + 2 * config['lstm_dims'], 2 * config['lstm_dims'])

        self.final_layer = nn.Linear(2 * config['lstm_dims'], 1)  # FF Layer
        self.output_activation = nn.Sigmoid()  # Using Sigmoid activation as similarity measure

    def code_encoding(self, code, qb):
        """
        :param code: processed code
        :param qb: processed qb
        :return: vector representation for code
        Encoded Code representation using both code annotation + code
        """
        code_repr = self.code_encoder(code)
        qb_repr = self.qb_encoder(qb)
        code_qb_repr = torch.cat((code_repr, qb_repr), 1)
        code_qb_repr = F.tanh(code_qb_repr)
        return code_qb_repr

    def qt_encoding(self, qt):
        """
        Encodes Question title
        :param qt: processed question title
        :return: vector representation for qt
        Encoded Question Title
        """
        qt_repr = self.qt_encoder(qt)
        return qt_repr

    def combine_qt_and_code(self, qt_repr, code_repr):
        combined_repr = self.fuse(torch.cat((qt_repr, code_repr), 1))
        similarity_score = self.output_activation(self.final_layer(combined_repr))
        print(similarity_score.data.cpu().numpy())
        return similarity_score

    def forward(self, qt, good_code, bad_code, good_qb, bad_qb):
        qt_repr = self.qt_encoding(qt)

        good_code_repr = self.code_encoding(good_code, good_qb)
        bad_code_repr = self.code_encoding(bad_code, bad_qb)

        good_sim = self.combine_qt_and_code(qt_repr, good_code_repr)
        bad_sim = self.combine_qt_and_code(qt_repr, bad_code_repr)

        loss = (self.margin - good_sim + bad_sim).clamp(min=1e-6).mean()
        return loss
