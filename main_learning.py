import argparse
import os
import sys
import time

import tensorflow as tf

from models.lstm_model import LSTMConfig, LSTMModel
from models.naive_model import NaiveConfig, NaiveModel
from models.bilstm_model import BiLSTMConfig, BiLSTMModel
from models.stacked_lstm_model import StackedLSTMConfig, StackedLSTMModel
from models.crf_model import CRFModel

from utils.Embedder import Embedder
from utils.crf import Crf


def main(args):
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")

    embedder = Embedder(args)
    embeddings, tok2idMap, learning_set = embedder.load_and_preprocess_data(args.tiny)

    # choosing train vs. test sets within our data
    # TODO make a separate test set ; in order to report out-of-sample
    # performance. Instead of 90% train and 10% dev, we'll do 80% train,
    # 10% dev and 10% test.
    train_set = learning_set[:int(len(learning_set) * args.train_fraction)]
    dev_set = learning_set[int(len(learning_set) * args.train_fraction):]

    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    with tf.Graph().as_default() as graph:
        print("Building model...", end=' ')
        start = time.time()

        if args.model == "naive":
            config = NaiveConfig(args)
            model = NaiveModel(config, embeddings, embedder)
        elif args.model == "lstm":
            config = LSTMConfig(args)
            model = LSTMModel(config, embeddings, embedder)
        elif args.model == "bilstm":
            config = BiLSTMConfig(args)
            model = BiLSTMModel(config, embeddings, embedder)
        elif args.model == "stacked_lstm":
            config = StackedLSTMConfig(args)
            model = StackedLSTMModel(config, embeddings, embedder)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        print("took {:.2f} seconds\n".format(time.time() - start))

        if args.crf:
            CRF = Crf([labels for sentence, labels in train_set])
            model = CRFModel(model, CRF, args)

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
    command_parser.add_argument('-m', '--model', choices=["lstm", "naive", "bilstm", "stacked_lstm"],
                                default="naive",
                                help="Type of model to use.")

    command_parser.add_argument('-f', '--five_classes', action='store_true',
                                help="Whether to train on the set of five classes (ORG, PER, LOC, MISC, O) instead of the original 17.")

    command_parser.add_argument('-crf', action='store_true', help="Use a fitted CRF to help predicting labels.")
    command_parser.add_argument('-s1', '--hidden_size', type=int, default=20, help="Size of hidden layers.")
    command_parser.add_argument('-s2', '--other_layer_size', type=int, default=20, help="Size of extra hidden layer.")
    command_parser.add_argument('-e', '--extra_layer', action='store_true',
                                help="Whether to add an extra layer on top (for LSTM) or a second one (for BiLSTM).")
    command_parser.add_argument('-ee', action='store_true',
                                help="Whether to add a second extra layer on top (for LSTM). Will trigger --extra_layer, no need to cumulate.")

    command_parser.add_argument('-b', '--batch_size', type=int, default=128, help="Size of batches.")
    command_parser.add_argument('-n', '--n_epochs', type=int, default=10, help="Number of epochs.")
    command_parser.add_argument('-l2', type=float, default=None, help="Beta for L2 regularization.")
    command_parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help="Learning rate.")
    command_parser.add_argument('-d', '--dropout', type=float, default=0, help="Dropout rate.")
    command_parser.add_argument('-alr', '--adaptative_lr', action='store_true',
                                help="Use an adaptative learning rate with the AdaGrad method.")

    command_parser.add_argument('-dt', '--data_train', type=str, default="data/ner_dataset.csv", help="Training data")
    command_parser.add_argument('-vv', '--vectors', type=str, default="data/en-cw.txt",
                                help="Path to word vectors file.")
    command_parser.add_argument('-tf', '--train_fraction', type=float, default=.9,
                                help="The fraction of the dataset to use for training.")

    command_parser.add_argument('-t', '--tiny', action='store_true', help="Whether to run on reduced dataset.")
    command_parser.add_argument('-no_result', action='store_true',
                                help="Activate in order NOT to store the run's result files.")

    command_parser.set_defaults(func=main)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
