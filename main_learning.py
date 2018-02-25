import os
import time

import tensorflow as tf

from naive_model import NaiveConfig, NaiveModel
from utils.Embedder import Embedder


def main(debug=True):
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = NaiveConfig()
    embedder = Embedder()
    embeddings, tok2idMap, learning_set = embedder.load_and_preprocess_data(debug)

    train_set = learning_set[:int(len(learning_set) * .9)]
    dev_set = learning_set[int(len(learning_set) * .9):]

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
        session.run(init_op)

        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        model.fit(session, saver, train_set, dev_set)


if __name__ == '__main__':
    main(debug=False)
