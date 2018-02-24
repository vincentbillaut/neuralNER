import logging

import tensorflow as tf

from utils.minibatches import minibatches
from utils.Progbar import Progbar
from utils.LabelsHandler import LabelsHandler
from utils.ConfusionMatrix import ConfusionMatrix
from utils.parser_utils import get_chunks

logger = logging.getLogger("NERproject")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_classes = 17
    n_epochs = 10
    lr = 0.0005


class NERModel(object):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict whether an input word is a Named Entity
    """

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labels_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        print("Each Model must re-implement this method?")
        return examples

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch.reshape(-1, 1), labels_batch=labels_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def test_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch.reshape(-1, 1), labels_batch=labels_batch)
        loss = sess.run([self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        return predictions

    def run_epoch(self, sess, train_examples, dev_set):
        n_minibatches = 1 + len(train_examples) / self.config.batch_size
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)], force=i + 1 == n_minibatches)

        for batch in minibatches(dev_set, dev_set.shape[0]):
            break
        loss = self.test_on_batch(sess, *batch)
        # print("Evaluating on dev set", end=' ')
        # dev_UAS, _ = parser.parse(dev_set)
        # print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
        # return dev_UAS
        return - loss[0]

    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        token_cm = ConfusionMatrix(labels=self.labelsHandler.keys())

        correct_preds, total_correct, total_preds = 0., 0., 0.
        for _, labels, labels_ in self.output(sess, examples_raw, examples):
            for l, l_ in zip(labels, labels_):
                token_cm.update(l, l_)
            gold = set(get_chunks(labels, self.labelsHandler.noneIndex()))
            pred = set(get_chunks(labels_, self.labelsHandler.noneIndex()))
            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return token_cm, (p, r, f1)

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)

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
        self.labelsHandler = LabelsHandler()
