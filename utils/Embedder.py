import os
import time

import numpy as np

from utils.LabelsHandler import LabelsHandler


class Embedder(object):
    # language = 'english'
    # with_punct = True
    # unlabeled = True
    lowercase = False
    # use_pos = True
    # use_dep = True
    # use_dep = use_dep and (not unlabeled)

    start_token = "<s>"
    end_token = "</s>"

    def __init__(self, args):
        self.train_file = args.data_train
        self.embedding_file = args.vectors

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

    def load_and_preprocess_data(self, reduced=False, embed_size=50):
        print("Loading {}data...".format("(reduced) " if reduced else ''), end='')
        start = time.time()
        learning_set = self.read_conll(os.path.join('', self.train_file),
                                       lowercase=self.lowercase)
        if reduced:
            learning_set = learning_set[:5000]

        print("took {:.2f} seconds".format(time.time() - start))

        print("Generating tokens...", end='')
        start = time.time()
        unique_words = frozenset().union(*[set(sentence[0]) for sentence in learning_set],
                                         {self.start_token, self.end_token})
        self.tok2id = {l: i for (i, l) in enumerate(unique_words)}
        learning_set_embedded = [(self.embed_sentence(sentence), label) for sentence, label in learning_set]

        labels_handler = LabelsHandler()
        learning_set_embedded_labelled = [(train_example, labels_handler.to_label_ids(labels))
                                          for train_example, labels in learning_set_embedded]
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

        return embeddings_matrix, self.tok2id, learning_set_embedded_labelled

    def start_token_id(self):
        return self.tok2id[self.start_token]

    def end_token_id(self):
        return self.tok2id[self.end_token]
