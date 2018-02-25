import os
import time

import tensorflow as tf

from utils.Embedder import Embedder
from utils.LabelsHandler import LabelsHandler

from naive_model import NaiveConfig, NaiveModel


def main(debug=True):
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = NaiveConfig()
    # parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
    embedder = Embedder()
    embeddings, tok2idMap, train_set = embedder.load_and_preprocess_data(debug)

    labels_handler = LabelsHandler()
    train_set = [(train_example, labels_handler.to_label_ids(labels)) for train_example, labels in train_set]
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default() as graph:
        print("Building model...", end=' ')
        start = time.time()
        model = NaiveModel(config, embeddings)
        # parser.model = model
        init_op = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()
        print("took {:.2f} seconds\n".format(time.time() - start))
    graph.finalize()

    with tf.Session(graph=graph) as session:
        # parser.session = session
        session.run(init_op)

        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        model.fit(session, saver, train_set, train_set)  # TODO dev set

        if not debug:
            print(80 * "=")
            print("TESTING")
            print(80 * "=")
            print("Restoring the best model weights found on the dev set")
            saver.restore(session, './data/weights/parser.weights')
            print("Final evaluation on test set", end=' ')
            UAS, dependencies = parser.parse(test_set)
            print("- test UAS: {:.2f}".format(UAS * 100.0))
            print("Writing predictions")
            with open('q2_test.predicted.pkl', 'w') as f:
                cPickle.dump(dependencies, f, -1)
            print("Done!")


if __name__ == '__main__':
    main(debug=False)
