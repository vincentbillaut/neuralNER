import time
import os

import numpy as np
import pandas as pd


class Embedder(object):
    # language = 'english'
    # with_punct = True
    # unlabeled = True
    lowercase = True
    # use_pos = True
    # use_dep = True
    # use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'ner_dataset.csv'
    # dev_file = 'dev.conll'
    # test_file = 'test.conll'
    embedding_file = './data/en-cw.txt'

    def read_conll(self, path, lowercase=False):
        input_data_frame = pd.read_csv(path,
                                     encoding="ISO-8859-1")
        input_array = np.array(input_data_frame.loc[:, ["Word", "Tag"]])
        return input_array

    def embed(self, wordsArray):
        return np.array([self.tok2id[word] for word in wordsArray])

    def load_and_preprocess_data(self, reduced=True, embed_size=50):
        print("Loading {}data...".format("(reduced) " if reduced else ''), end='')
        start = time.time()
        train_set = self.read_conll(os.path.join(self.data_path, self.train_file),
                                    lowercase=self.lowercase)
        # dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
        #                      lowercase=config.lowercase)
        # test_set = read_conll(os.path.join(config.data_path, config.test_file),
        #                       lowercase=config.lowercase)
        if reduced:
            train_set = train_set[:100000]
            # dev_set = dev_set[:500]
            # test_set = test_set[:500]
        print("took {:.2f} seconds".format(time.time() - start))

        print("Generating tokens...", end='')
        start = time.time()
        self.tok2id = {l: i for (i, l) in enumerate(set(train_set[:, 0]))}
        print("took {:.2f} seconds".format(time.time() - start))

        print("Loading pretrained embeddings...", end='')
        start = time.time()
        word_vectors = {}
        for line in open(self.embedding_file).readlines():
            sp = line.strip().split()
            word_vectors[sp[0]] = [float(x) for x in sp[1:]]
        embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (len(self.tok2id), embed_size)), dtype='float32')

        for token in self.tok2id:
            i = self.tok2id[token]
            if token in word_vectors:
                embeddings_matrix[i] = word_vectors[token]
            elif token.lower() in word_vectors:
                embeddings_matrix[i] = word_vectors[token.lower()]
        print("took {:.2f} seconds".format(time.time() - start))

        train_set[:, 0] = self.embed(train_set[:, 0])
        return embeddings_matrix, self.tok2id, train_set
