import os
import time
import logging

import tensorflow as tf

from model import Model
from utils.Embedder import Embedder
from utils.LabelsHandler import LabelsHandler
from utils.minibatches import minibatches

logger = logging.getLogger("hw3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """


class NERModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict whether an input word is a Named Entity
    """

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch.reshape(-1, 1), labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def test_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch.reshape(-1, 1), labels_batch=labels_batch)
        loss = sess.run([self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess, train_examples, dev_set):
        n_minibatches = 1 + len(train_examples) / self.config.batch_size
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)], force=i + 1 == n_minibatches)

        for batch in minibatches(dev_set, dev_set.shape[0]):
            break
        loss = self.test_on_batch(sess, *batch)
        print("Evaluating on dev set", end=' ')
        dev_UAS, _ = parser.parse(dev_set)
        print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
        return dev_UAS
        return - loss[0]

    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            # You may use the progress bar to monitor the training progress
            # Addition of progress bar will not be graded, but may help when debugging
            prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))

            # The general idea is to loop over minibatches from train_examples, and run train_on_batch inside the loop
            # Hint: train_examples could be a list containing the feature data and label data
            # Read the doc for utils.get_minibatches to find out how to use it.
            # Note that get_minibatches could either return a list, or a list of list
            # [features, labels]. This makes expanding tuples into arguments (* operator) handy

            for i, minibatch in enumerate(minibatches(train_examples, self.config.batch_size)):
                loss = self.train_on_batch(sess, *minibatch)
                prog.update(i + 1, [("loss = ", loss)])

            logger.info("Evaluating on development data")
            token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
            logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
            logger.debug("Token-level scores:\n" + token_cm.summary())
            logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

            score = entity_scores[-1]

            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
        return best_score

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()

