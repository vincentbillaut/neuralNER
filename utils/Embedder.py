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

    def read_conll(seld, path, lowercase=False):
        """
        Reads a input stream @fstream (e.g. output of `open(fname, 'r')`) in CoNLL file format.
        @returns a list of examples [(tokens), (labels)]. @tokens and @labels are lists of string.
        """
        ret = []

        current_toks, current_lbls = [], []
        for line in open(path, 'r', encoding="ISO-8859-1"):
            line = line.strip()
            if len(line) == 0 or "Sentence" in line:
                if len(current_toks) > 0:
                    assert len(current_toks) == len(current_lbls)
                    ret.append((current_toks, current_lbls))
                current_toks, current_lbls = [], []
            else:
                assert "," in line, r"Invalid CONLL format; expected a ',' in {}".format(line)
                splitted_line = line.split(",")
                tok = "".join(splitted_line[1:-2])
                lbl = splitted_line[-1]
                current_toks.append(tok)
                current_lbls.append(lbl)
        if len(current_toks) > 0:
            assert len(current_toks) == len(current_lbls)
            ret.append((current_toks, current_lbls))
        return ret

    def embed_sentence(self, wordsArray):
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
            train_set = train_set[:1000]
            # dev_set = dev_set[:500]
            # test_set = test_set[:500]
        print("took {:.2f} seconds".format(time.time() - start))

        print("Generating tokens...", end='')
        start = time.time()
        unique_words = frozenset().union(*[set(sentence[0]) for sentence in train_set])
        self.tok2id = {l: i for (i, l) in enumerate(unique_words)}
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

        train_set_embedded = [(self.embed_sentence(sentence), label) for sentence, label in train_set]
        return embeddings_matrix, self.tok2id, train_set_embedded
