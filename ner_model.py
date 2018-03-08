import json
import logging
import os
import random
from datetime import datetime

import tensorflow as tf

from utils.ConfusionMatrix import ConfusionMatrix
from utils.LabelsHandler import LabelsHandler
from utils.Progbar import Progbar
from utils.minibatches import minibatches2
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
    embed_size = 50

    def __init__(self, args):
        name = type(self).__name__
        self.no_result_storage = args.no_result
        self.output_path = "results/" + name + "/{:%Y%m%d_%H%M%S}/".format(datetime.now())

        logger.info("starting job at " + self.output_path)
        if not os.path.exists(self.output_path) and not self.no_result_storage:
            os.makedirs(self.output_path)

        if not self.no_result_storage:
            with open(os.path.join(self.output_path, "params.json"), "w") as f:
                dico = vars(args)
                del dico['func']
                json.dump(dico, f)
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "window_predictions.conll"
        # parameters passed by args
        self.n_epochs = args.n_epochs
        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.regularization = args.l2
        self.adaptative_lr = args.adaptative_lr


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

    def create_feed_dict(self, inputs, mask_batch, labels_batch=None, dropout=0):
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

    def add_regularization_op(self, loss, beta):
        """Adds Ops to regularize the loss function to the computational graph.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            regularized_loss: A 0-d tensor (scalar) output
        """
        reg_loss = beta * sum([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
                               if not ("noreg" in tf_var.name or "Bias" in tf_var.name)])
        return reg_loss + loss

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
        if self.config.adaptative_lr:
            opt = tf.train.AdagradOptimizer(learning_rate=self.config.lr)
        else:
            opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)

        train_op = opt.minimize(loss)
        return train_op

    def add_predict_onehot(self):
        return tf.argmax(self.pred, axis=2)

    def add_predict_proba(self):
        return tf.nn.softmax(self.pred, axis=2)

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.pred_onehot = self.add_predict_onehot()
        self.pred_proba = self.add_predict_proba()
        self.loss = self.add_loss_op(self.pred)

        if self.config.regularization is not None:
            self.regularized_loss = self.add_regularization_op(
                self.loss,
                self.config.regularization)
        else:
            self.regularized_loss = self.loss
        self.train_op = self.add_training_op(self.regularized_loss)

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        examples_with_mask = [(ex[0], ex[1], mask) for ex, mask in zip(examples, iter(lambda: True, False))]
        return examples_with_mask

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch,  # .reshape(-1, 1),
                                     labels_batch=labels_batch,
                                     mask_batch=mask_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def test_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch,  # .reshape(-1, 1),
                                     labels_batch=labels_batch,
                                     mask_batch=mask_batch)
        loss = sess.run([self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch,  # .reshape(-1, 1),
                                     mask_batch=mask_batch)
        predictions = sess.run(self.pred_onehot, feed_dict=feed)
        return predictions

    def predict_proba_on_batch(self, sess, inputs_batch, mask_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        feed = self.create_feed_dict(inputs_batch,  # .reshape(-1, 1),
                                     mask_batch=mask_batch)
        predictions = sess.run(self.pred_proba, feed_dict=feed)
        return predictions

    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples_raw: A list of the original input/output sequence pairs.
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

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches2(inputs, self.config.batch_size, shuffle=False)):
            preds_ = self.predict_on_batch(sess, inputs_batch=batch[0], mask_batch=batch[2])
            preds_proba_ = self.predict_proba_on_batch(sess, inputs_batch=batch[0], mask_batch=batch[2])
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def fit(self, sess, saver, train_examples_raw, dev_examples_raw):
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_examples = self.preprocess_sequence_data(dev_examples_raw)

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

            losses = []
            for i, minibatch in enumerate(
                    minibatches2(train_examples, self.config.batch_size)):
                loss = self.train_on_batch(sess, *minibatch)
                losses.append(loss)
                prog.update(i + 1, [("loss = ", loss)])

            random_train_examples_id = random.sample(range(len(train_examples)), k=len(dev_examples))
            _, train_entity_scores = self.evaluate(sess,
                                                   [train_examples[ind] for ind in random_train_examples_id],
                                                   [train_examples_raw[ind] for ind in random_train_examples_id])
            p, r, f1 = train_entity_scores
            if not self.config.no_result_storage:
                with open(self.config.output_path + "train_losses.los", "a") as f:
                    for item in losses:
                        f.write("%s\n" % item)
                with open(self.config.output_path + "train_f1.los", "a") as f:
                    f.write("%s\n" % f1)
                with open(self.config.output_path + "train_precision.los", "a") as f:
                    f.write("%s\n" % p)
                with open(self.config.output_path + "train_recall.los", "a") as f:
                    f.write("%s\n" % r)

            logger.info("Evaluating on development data")
            for dev_minibatch in minibatches2(dev_examples, len(dev_examples)):
                dev_loss = self.test_on_batch(sess, *dev_minibatch)[0]
                break
            token_cm, dev_entity_scores = self.evaluate(sess,
                                                        dev_examples,
                                                        dev_examples_raw)
            logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
            logger.debug("Token-level scores:\n" + token_cm.summary())
            logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *dev_entity_scores)
            logger.info("Dev set loss: %.4f", dev_loss)
            p, r, f1 = dev_entity_scores
            if not self.config.no_result_storage:
                with open(self.config.output_path + "dev_losses.los", "a") as f:
                    f.write("%s\n" % dev_loss)
                with open(self.config.output_path + "dev_f1.los", "a") as f:
                    f.write("%s\n" % f1)
                with open(self.config.output_path + "dev_precision.los", "a") as f:
                    f.write("%s\n" % p)
                with open(self.config.output_path + "dev_recall.los", "a") as f:
                    f.write("%s\n" % r)

            score = dev_entity_scores[-1]

            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
        return best_score

    def __init__(self, config, pretrained_embeddings, embedder):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()
        self.labelsHandler = LabelsHandler()
        self.embedder = embedder
