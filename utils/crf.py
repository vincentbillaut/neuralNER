import numpy as np
import time

from utils.LabelsHandler import LabelsHandler


class Crf(object):
    def predict_proba_label(self, label):
        return self.predict_proba_index(self.labelsHandler.labels_map[label])

    def predict_proba_index(self, index):
        return self.transitionMatrix[index, :]

    def buildCRF(self, transitionData):
        """Returns the CRF transition matrix from labelled sentences.

        Args:
            transitionData: List of sentences, each sentence being a list of NER labels.
        Returns:
            transitionMatrix: The transition matrix of the CRF based on input data: transitionMatrix[i, j]
                    is the frequency of observed transitions from label i to label j.
        """
        print("Building CRF...", end=' ')
        start = time.time()

        transitionMatrix = np.zeros(shape=(self.labelsHandler.num_labels(),
                                           self.labelsHandler.num_labels()))
        for sentence in transitionData:
            for i, j in zip(sentence, sentence[1:]):
                transitionMatrix[i, j] += 1

        transitionMatrix /= np.sum(transitionMatrix, axis=1).reshape((-1, 1))
        print("took {:.2f} seconds\n".format(time.time() - start))
        return transitionMatrix

    def __init__(self, transitionData):
        self.labelsHandler = LabelsHandler()
        self.transitionMatrix = self.buildCRF(transitionData)
