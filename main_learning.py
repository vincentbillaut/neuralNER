import os
import time

import argparse

import tensorflow as tf

from naive_model import NaiveConfig, NaiveModel
from lstm_model import LSTMConfig, LSTMModel
from utils.Embedder import Embedder


def main(args):
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    if args.model == "naive":
        config = NaiveConfig(args)
    elif args.model == "lstm":
        config = LSTMConfig(args)
    else:
        raise NotImplementedError("Invalid model specified")

    embedder = Embedder(args)
    embeddings, tok2idMap, learning_set = embedder.load_and_preprocess_data(args.tiny)

    train_set = learning_set[:int(len(learning_set) * .9)]
    dev_set = learning_set[int(len(learning_set) * .9):]

    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default() as graph:
        print("Building model...", end=' ')
        start = time.time()

        if args.model == "naive":
            model = NaiveModel(config, embeddings)
        elif args.model == "lstm":
            model = LSTMModel(config, embeddings)
        else:
            raise NotImplementedError("Invalid mode")

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        print("took {:.2f} seconds\n".format(time.time() - start))
    graph.finalize()

    with tf.Session(graph=graph) as session:
        session.run(init_op)

        print(80 * "=")
        print("TRAINING")
        print(80 * "=")
        model.fit(session, saver, train_set, dev_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-m', '--model', choices=["lstm", "naive"], default="naive", help="Type of model to use.")
    command_parser.add_argument('-vv', '--vectors', type=str, default="data/en-cw.txt", help="Path to word vectors file.")
    command_parser.add_argument('-b', '--batch_size', type=int, default=128, help="Size of batches.")
    command_parser.add_argument('-n', '--n_epochs', type=int, default=10, help="Number of epochs.")
    command_parser.add_argument('-t', '--tiny', type=bool, default=False, help="Whether to run on reduced dataset.")
    command_parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help="Learning rate.")
    command_parser.add_argument('-dt', '--data_train', type=str, default="data/ner_dataset.csv", help="Training data")
    # command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/tiny.conll", help="Dev data")
    # command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.set_defaults(func=main)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
