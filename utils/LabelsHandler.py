import numpy as np


class LabelsHandler(object):
    def __init__(self):
        self.labels_map = {'B-art': 0,
                           'B-eve': 1,
                           'B-geo': 2,
                           'B-gpe': 3,
                           'B-nat': 4,
                           'B-org': 5,
                           'B-per': 6,
                           'B-tim': 7,
                           'I-art': 8,
                           'I-eve': 9,
                           'I-geo': 10,
                           'I-gpe': 11,
                           'I-nat': 12,
                           'I-org': 13,
                           'I-per': 14,
                           'I-tim': 15,
                           'O': 16}
        pass

    def to_label_ids(self, labels):
        return np.array([self.labels_map[label] for label in labels])

    def keys(self):
        return ['B-art',
                'B-eve',
                'B-geo',
                'B-gpe',
                'B-nat',
                'B-org',
                'B-per',
                'B-tim',
                'I-art',
                'I-eve',
                'I-geo',
                'I-gpe',
                'I-nat',
                'I-org',
                'I-per',
                'I-tim',
                'O']
