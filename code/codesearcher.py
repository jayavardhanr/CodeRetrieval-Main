import os
import random
import numpy as np
import math
import argparse
import logging

import torch
from torch import optim
import torch.nn.functional as F

from utils import gVar
from configs import get_config
from data import CodeSearchDataset, CodennDataset
from models import JointEmbeder, JointEmbederQB, JointEmbederQB2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


class CodeSearcher:
    def __init__(self, conf):
        self.conf = conf
        self.path = conf['workdir']
        self.valid_set = None

    #######################
    # Model Loading / saving #####
    #######################
    def save_model(self, model, epoch):
        if not os.path.exists(self.conf['model_directory']):
            os.makedirs(self.conf['model_directory'])
        torch.save(model.state_dict(), self.conf['model_directory'] + 'best_model.h5')

    def load_model(self, model, epoch):
        assert os.path.exists(self.conf['model_directory'] + 'best_model.h5'), 'Weights for saved model not found'
        model.load_state_dict(torch.load(self.conf['model_directory'] + 'best_model.h5'))

    #######################
    # Training #####
    #######################
    def train(self, model):
        """
        Trains an initialized model
        :param model: Initialized model
        :return: None
        """
        log_every = self.conf['log_every']
        valid_every = self.conf['valid_every']
        batch_size = self.conf['batch_size']
        nb_epoch = self.conf['nb_epoch']
        max_patience = self.conf['patience']

        train_set = CodeSearchDataset(self.path, self.conf, "train")
        data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                                  shuffle=True, drop_last=True, num_workers=1)

        # MRR for the Best Saved model, if reload > 0, else -1
        if self.conf['reload'] > 0:
            _, max_mrr, _, _ = self.eval(model, 50, 10, "val")
        else:
            max_mrr = -1

        patience = 0
        for epoch in range(self.conf['reload'] + 1, nb_epoch):
            itr = 1
            losses = []

            model = model.train()
            for qts, good_codes, bad_codes, good_qbs, bad_qbs in data_loader:
                qts, good_codes, bad_codes, good_qbs, bad_qbs = gVar(qts), gVar(good_codes), gVar(bad_codes), gVar(
                    good_qbs), gVar(bad_qbs)
                loss = model(qts, good_codes, bad_codes, good_qbs, bad_qbs)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if itr % log_every == 0:
                    logger.info('epo:[%d/%d] itr:%d Loss=%.5f' % (epoch, nb_epoch, itr, np.mean(losses)))
                    losses = []
                itr = itr + 1

            if epoch % valid_every == 0:
                logger.info("validating..")
                acc1, mrr, map, ndcg = self.eval(model, 50, 10, "val")

                if mrr > max_mrr:
                    self.save_model(model, epoch)
                    patience = 0
                    print("Model improved. Saved model at %d epoch" % epoch)
                    max_mrr = mrr
                else:
                    print("Model didn't improve for ", patience + 1, " epochs")
                    patience += 1

            if patience >= max_patience:
                logger.info("Patience Limit Reached. Stopping Training")
                break

    ########################
    # Evaluation on CodeNN #
    ########################
    def eval_codenn(self, model, poolsize, K, dataset):
        """
        simple validation in a code pool.
        :param model: Trained Model
        :param poolsize: poolsize - size of the code pool, if -1, load the whole test set
        :param K: Top K results
        :param dataset: which dataset to evaluate on
        :return: Accuracy, MRR, MAP, nDCG
        """
        if self.valid_set is None:
            if dataset == "val":
                self.valid_set = CodennDataset(self.path, self.conf, "val")  # load dev set
            else:
                self.valid_set = CodennDataset(self.path, self.conf, "test")  # load eval set

        data_loader = torch.utils.data.DataLoader(dataset=self.valid_set, batch_size=poolsize, shuffle=False,
                                                  num_workers=1)
        model = model.eval()

        accs, mrrs, maps, ndcgs = [], [], [], []
        for qts, codes, qbs in data_loader:
            qts, codes, qbs = gVar(qts), gVar(codes), gVar(qbs)
            code_repr = model.code_encoding(codes)

            if self.conf['use_qb']:
                qb_repr = model.qb_encoding(qbs)
            else:
                qb_repr = None

            qt = gVar(qts[0].expand(poolsize, -1))
            qt_repr = model.qt_encoding(qt)

            sims = model.score_qt_code_qb(qt_repr, code_repr, qb_repr).data.cpu().numpy()
            # sims = model.combine_qt_and_code(qt_repr, code_repr).data.cpu().numpy()
            # sims = F.cosine_similarity(code_repr, qt_repr).data.cpu().numpy()
            # n_results = K

            negsims = np.negative(sims)
            predict = np.argsort(negsims)
            # predict = np.argpartition(negsims, kth=n_results-1)
            # predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [0]  # index of the positive sample

            mrrs.append(MRR(real, predict))
            accs.append(ACC(real, predict))
            maps.append(MAP(real, predict))
            ndcgs.append(NDCG(real, predict))

        logger.info(
            'ACC={}, MRR={}, MAP={}, nDCG={}'.format(np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)

    #######################
    # Evaluation on StaQC #####
    #######################
    def eval(self, model, poolsize, K, dataset):
        """
        simple validation in a code pool.
        :param model: Trained Model
        :param poolsize: poolsize - size of the code pool, if -1, load the whole test set
        :param K: Top K results
        :param dataset: which dataset to evaluate on
        :return: Accuracy, MRR, MAP, nDCG
        """
        if self.valid_set is None:  # load evaluation dataset
            if dataset == "train":
                self.valid_set = CodeSearchDataset(self.path, self.conf, "train")  # load train set
            elif dataset == "val":
                self.valid_set = CodeSearchDataset(self.path, self.conf, "val")  # load val set
            else:
                self.valid_set = CodeSearchDataset(self.path, self.conf, "test")  # load test set

        data_loader = torch.utils.data.DataLoader(dataset=self.valid_set, batch_size=poolsize,
                                                  shuffle=True, drop_last=True, num_workers=1)

        model = model.eval()
        accs, mrrs, maps, ndcgs = [], [], [], []
        for qts, codes, _, qbs, _ in data_loader:
            qts, codes, qbs = gVar(qts), gVar(codes), gVar(qbs)
            code_repr = model.code_encoding(codes)
            if self.conf['use_qb']:
                qb_repr = model.qb_encoding(qbs)
            else:
                qb_repr = None

            for i in range(poolsize):
                qt = gVar(qts[i].expand(poolsize, -1))
                qt_repr = model.qt_encoding(qt)

                sims = model.score_qt_code_qb(qt_repr, code_repr, qb_repr).data.cpu().numpy()
                # sims = model.combine_qt_and_code(qt_repr, code_repr).data.cpu().numpy()
                # sims = F.cosine_similarity(torch.concat(qt_repr,code_repr)).data.cpu().numpy()
                # n_results = K

                negsims = np.negative(sims)
                predict = np.argsort(negsims)
                # predict = np.argpartition(negsims, kth=n_results-1)
                # predict = predict[:n_results]
                predict = [int(k) for k in predict]
                real = [i]  # index of positive sample
                accs.append(ACC(real, predict))
                mrrs.append(MRR(real, predict))
                maps.append(MAP(real, predict))
                ndcgs.append(NDCG(real, predict))
        logger.info(
            'ACC={}, MRR={}, MAP={}, nDCG={}'.format(np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)))
        return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)


########################
# Metric Calculations ##
########################


def ACC(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1
    return sum / float(len(real))


def MAP(real, predict):
    sum = 0.0
    for id, val in enumerate(real):
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + (id + 1) / float(index + 1)
    return sum / float(len(real))


def MRR(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))


def NDCG(real, predict):
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i + 1
            dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    return dcg / float(idcg)


def IDCG(n):
    idcg = 0
    itemRelevance = 1
    for i in range(n):
        idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
    return idcg


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument("-m", "--mode", choices=["train", "eval"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                             " the `eval` mode evaluates models on a test set ",
                        required=True)
    parser.add_argument("--verbose", action="store_true", default=True, help="Be verbose")
    parser.add_argument("--use_qb", type=int, default=1, help="Should I use Question body for CR?", required=True)
    parser.add_argument("--code_enc", type=str, default="config", help="which encoder to use for code?", required=True)
    parser.add_argument("--reload", type=int, default=-1, help=" Should I reload saved model, yes if reload>0?",
                        required=True)
    parser.add_argument("--dropout", type=float, default=0.0, help="What is the dropout?", required=True)
    parser.add_argument("--emb_size", type=int, default=100, help="What is the embedding size?", required=True)
    parser.add_argument("--lstm_dims", type=int, default=200, help="What is the lstm dimension?", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="What is the batch size?", required=True)
    return parser.parse_args()


def create_model_name_string(c):
    string1 = 'qtlen_{}_qblen_{}_codelen_{}_qtnwords_{}_qbnwords_{}_codenwords_{}_batch_{}_optimizer_{}_lr_{}'. \
        format(c['qt_len'], c['qb_len'], c['code_len'], c['qt_n_words'], c['qb_n_words'], c['code_n_words'],
               c['batch_size'], c['optimizer'], str(c['lr'])[2:])
    string2 = '_embsize_{}_lstmdims_{}_bowdropout_{}_seqencdropout_{}_simmeasure_{}'. \
        format(c['emb_size'], c['lstm_dims'], str(c['bow_dropout'])[2:], str(c['seqenc_dropout'])[2:], c['sim_measure'])
    string3 = '_maxpool'
    string = string1 + string2 + string3
    return string


if __name__ == '__main__':
    args = parse_args()
    conf = get_config()

    conf['bow_dropout'] = args.dropout
    conf['seqenc_dropout'] = args.dropout
    conf['emb_size'] = args.emb_size
    conf['lstm_dims'] = args.lstm_dims
    conf['batch_size'] = args.batch_size
    conf['reload'] = args.reload
    conf['code_encoder'] = args.code_enc

    if conf['reload'] <= 0 and args.mode == 'eval':
        print("For eval mode, please give reload=1. If you looking to train the model, change the mode to train. "
              "\n Note: Train overrides previously saved model, if it had exactly the same parameters")
    else:
        if args.mode == 'train':
            print("Warning: Train overrides previously saved model, if it had exactly the same parameters")
            print("If retraining the model from previous check point, set reload >0 to start training from previous "
                  "checkpoint")

        if args.use_qb == 0:
            conf['use_qb'] = False  # Using QB in Code Retrieval Model
        else:
            conf['use_qb'] = True  # Not Using QB in Code Retrieval Model
        print("Model using Question Body : ", conf['use_qb'])

        print(" Code encoder : ", conf['code_encoder'])
        print(" Dropout : ", conf['seqenc_dropout'])
        print(" Embedding size : ", conf['emb_size'])
        print(" LSTM hidden dimension : ", conf['lstm_dims'])

        # Creating unique model string based on parameters defined. Helps differentiate between different runs of model
        model_string = create_model_name_string(conf)

        if conf['use_qb']:
            if conf['code_encoder'] == 'bilstm':
                conf['model_directory'] = conf['workdir'] + 'with_qb/' + 'bilstm/' + model_string + '/'
            else:
                conf['model_directory'] = conf['workdir'] + 'with_qb/' + 'bow/' + model_string + '/'
        else:
            if conf['code_encoder'] == 'bilstm':
                conf['model_directory'] = conf['workdir'] + 'no_qb/' + 'bilstm/' + model_string + '/'
            else:
                conf['model_directory'] = conf['workdir'] + 'no_qb/' + 'bow/' + model_string + '/'

        if not os.path.exists(conf['model_directory']):
            os.makedirs(conf['model_directory'])
        print(" Model Directory : ")
        print(conf['model_directory'])

        searcher = CodeSearcher(conf)

        #####################
        # Define model ######
        #####################
        logger.info('Building Model')
        model = JointEmbederQB2(conf)  # Initialize the model

        if conf['reload'] > 0:
            if args.mode == 'eval':
                print("Reloading saved model for evaluating")
            else:
                print("Reloading saved model for Re-training")
            searcher.load_model(model, conf['reload'])

        if torch.cuda.is_available():
            print('using GPU')
            model = model.cuda()
        else:
            print('using CPU')

        optimizer = optim.Adam(model.parameters(), lr=conf['lr'])

        if args.mode == 'train':
            logger.info('Training Model')
            searcher.train(model)

        elif args.mode == 'eval':
            logger.info('Evaluating Model')

            # The last parameter specifies which dataset to evaluate on
            # "test" for test set for staQC | eval set for CodeNN
            # "val for val set for staQC | dev set for CodeNN
            # "train" for train for staQC | we don't have train set evaluation for CodeNN

            # Eval - evaluate for a StaQC data
            searcher.eval(model, 50, 10, "test")

            # Eval_codenn - Testing on Codenn Dataset - Make sure to change the config file
            # to load appropriate codenn data
            # searcher.eval_codenn(model, 50, 10, "test")
        else:
            print("Please provide a Valid argument for mode - train/eval")
