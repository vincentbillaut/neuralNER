import time
import os

import numpy as np

class Config(object):
    # language = 'english'
    # with_punct = True
    # unlabeled = True
    lowercase = True
    # use_pos = True
    # use_dep = True
    # use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'train.conll'
    # dev_file = 'dev.conll'
    # test_file = 'test.conll'
    embedding_file = './data/en-cw.txt'


def load_and_preprocess_data(reduced=True, embed_size=50):
    config = Config()

    print("Loading {}data...".format("(reduced) " if reduced else ''), end='')
    start = time.time()
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    # dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
    #                      lowercase=config.lowercase)
    # test_set = read_conll(os.path.join(config.data_path, config.test_file),
    #                       lowercase=config.lowercase)
    if reduced:
        train_set = train_set[:1000]
        # dev_set = dev_set[:500]
        # test_set = test_set[:500]
    print("took {:.2f} seconds".format(time.time() - start))

    print("Generating tokens...", end='')
    start = time.time()
    tok2id = {l: i for (i, l) in enumerate(set(train_set[:,0]))}
    print("took {:.2f} seconds".format(time.time() - start))

    print("Loading pretrained embeddings...",end='')
    start = time.time()
    word_vectors = {}
    for line in open(config.embedding_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (len(tok2id), embed_size)), dtype='float32')

    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
    print("took {:.2f} seconds".format(time.time() - start))

    return embeddings_matrix
