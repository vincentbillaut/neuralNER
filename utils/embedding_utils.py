import time
import os
import logging
from collections import Counter

import numpy as np

class Config(object):
    # language = 'english'
    # with_punct = True
    # unlabeled = True
    # lowercase = True
    # use_pos = True
    # use_dep = True
    # use_dep = use_dep and (not unlabeled)
    # data_path = './data'
    # train_file = 'train.conll'
    # dev_file = 'dev.conll'
    # test_file = 'test.conll'
    embedding_file = './data/en-cw.txt'


def load_embedding():
    config = Config()
    print("Loading pretrained embeddings...",)
    start = time.time()
    word_vectors = {}
    for line in open(config.embedding_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')

    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
    print "took {:.2f} seconds".format(time.time() - start)
