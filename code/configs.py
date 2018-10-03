def get_config():
    conf = {
        'workdir': '../data/',
        # data_params

        ########################
        # Training data ########
        ########################

        # StaQC Dataset
        'train_qt': 'sql.train.qt.pkl',
        'train_code': 'sql.train.code.pkl',
        'train_qb': 'sql.train.qb.pkl',

        # Dataset to check learning from Code and QB separately
        # 'train_code': 'sql.train.sanity.code.pkl'  # Code Replaced empty String
        # 'train_qb': 'sql.train.sanity.qt.pkl'  # QB Replaced by empty String

        # Check generalization to Code Annotation
        # 'train_qb': 'sql.train.anno.qt.pkl',  # QT created using QB vocab

        ########################
        # Validation data ######
        ########################

        # StaQC Dataset
        'val_qt': 'sql.val.qt.pkl',
        'val_code': 'sql.val.code.pkl',
        'val_qb': 'sql.val.qb.pkl',

        # Dataset to check learning from Code and QB separately
        # 'val_code': 'sql.val.sanity.code.pkl'  # Code Replaced empty String
        # 'val_qb': 'sql.val.sanity.qt.pkl'  # QB Replaced by empty String

        # Check generalization to Code Annotation
        # 'val_qb': 'sql.val.anno.qt.pkl', # QT created using QB vocab

        # CodeNN data
        # 'val_qt':'codenn_combine.sql.dev.qt.pkl', # CodeNN Validation set
        # 'val_code':'codenn_combine.sql.dev.code.pkl', # CodeNN Validation set
        # 'val_qb':'codenn_combine.sql.dev.qt.pkl', # CodeNN Validation set - Using QT inplace of QB for now


        ########################
        # Test data ##########
        ########################

        # StaQC Dataset
        'test_qt': 'sql.test.qt.pkl',
        'test_code': 'sql.test.code.pkl',
        'test_qb': 'sql.test.qb.pkl',

        # Dataset to check learning from Code and QB separately
        # 'test_code': 'sql.test.sanity.code.pkl'  # Code Replaced empty String
        # 'test_qb': 'sql.test.sanity.qt.pkl'  # QB Replaced empty String

        # Check generalization to Code Annotation
        # 'test_qb': 'sql.test.anno.qt.pkl', # QT created using QB vocab

        # CodeNN data
        # 'test_qt': 'codenn_combine.sql.eval.qt.pkl', # CodeNN Eval set
        # 'test_qb': 'codenn_combine.sql.eval.qt.pkl', # CodeNN Eval set - Using QT inplace of QB for now
        # 'test_code': 'codenn_combine.sql.eval.code.pkl', # CodeNN Eval set

        # Sanity Check Datasets
        # 'test_code': 'codenn_combine_sanity_pad.sql.eval.code.pkl', # CodeNN Eval Sanity Check- only pad neg samples
        # 'test_code': 'codenn_combine_sanity_unk.sql.eval.code.pkl', # CodeNN Eval Sanity Check- only UNK neg samples

        # parameters
        'qt_len': 14,  # 9,
        'qb_len': 83,  # 82,  # 44,
        'code_len': 118,  # 119,  # 60,

        'qt_n_words': 4947,  # 4 is added for UNK, EOS, SOS, PAD
        'qb_n_words': 38008,
        'code_n_words': 7726,

        # vocabulary info
        'vocab_qt': 'sql.qt.vocab.pkl',
        'vocab_code': 'sql.code.vocab.pkl',
        'vocab_qb': 'sql.qb.vocab.pkl',

        # training_params
        'batch_size': 1024,
        'nb_epoch': 500,
        'optimizer': 'adam',
        'lr': 0.001,
        'valid_every': 1,
        'n_eval': 100,
        'log_every': 1000,
        'save_every': 10,
        'patience': 20,
        'reload': 1,  # reload>0, model is reloaded. If reload<=0, then train from scratch

        # model_params
        'emb_size': 200,
        # 'n_hidden': 400,#number of hidden dimension of code/desc representation
        # recurrent
        'lstm_dims': 400,  # * 2
        'bow_dropout': 0.25,  # dropout for BOW encoder
        'seqenc_dropout': 0.25,  # dropout for sequence encoder encoder
        'init_embed_weights_qt': None,  # word2vec_100_qt.h5,
        'init_embed_weights_code': None,  # 'word2vec_100_code.h5',
        'init_embed_weights_qb': None,  # 'word2vec_100_qb.h5',
        'margin': 0.05,
        'sim_measure': 'sigmoid',  # similarity measure: gesd, cosine, aesd
        'use_qb': 1,
        'code_encoder': 'bilstm'  # bow, bilstm

    }
    return conf
