import os
import time
import tensorflow as tf
import numpy as np

from ner_model import NERModel
from lstm_model import LSTMModel
from utils.minibatches import minibatches


def compute_transition_matrix(config, examples):
    """Computes the transition matrix A based on the training set provided
    in argument. A[i,j] corresponds to the likelihood of seeing a transition
    of type "ith label --> jth label". 0 and n_classes are the start and end
    tags of a sequence.

    See
    https://arxiv.org/pdf/1603.01360.pdf
    section 2.2 for more details.

    Parameters
    ----------
    config : LSTMConfig
        Config object, to have access to n_classes.
    examples : list
        Training set, to have access to our training labels

    Returns
    -------
    ndarray
        Transition matrix computed from the training examples.

    """
    transit = np.zeros((config.n_classes+2, config.n_classes+2))
    for _, labels in examples:
        transit[0,labels[0]] += 1
        for i in range(len(labels)-1):
            transit[labels[i]+1, labels[i+1]+1] += 1
        transit[labels[-1],config.n_classes+1] += 1
    transit = (transit.T * 1.0 / transit.sum(axis=1)).T
    return transit


class LSTMCRFConfig(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    # n_features = 36
    n_classes = 17
    embed_size = 50
    hidden_size = 200


class LSTMCRFModel(LSTMModel):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict whether an input word is a Named Entity
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        input_placeholder: Input placeholder tensor of  shape (None, 1), type tf.int32
            containing index of our word in the embedding
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout rate placeholder, scalar, type float32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        """
        self.input_placeholder = tf.placeholder(tf.int32, (None, 1))
        self.labels_placeholder = tf.placeholder(
            tf.float32, (None, self.config.n_classes))
        self.mask_placeholder = tf.placeholder(tf.bool, [None, self.max_length])

    def create_feed_dict(self, inputs, mask_batch, labels_batch=None, dropout=0):
        """Creates the feed_dict for the dependency parser.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: Dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """

        feed_dict = {
            self.input_placeholder: inputs,
            self.mask_placeholder: mask_batch
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors.

        Returns:
            embeddings: tf.Tensor of shape (None, 1*embed_size)
        """

        init_embed = tf.Variable(initial_value=self.pretrained_embeddings)
        embeddings0 = tf.nn.embedding_lookup(
            params=init_embed, ids=self.input_placeholder)
        embeddings = tf.reshape(
            tensor=embeddings0, shape=[-1, 1 * self.config.embed_size])

        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled LSTM

        Returns:
            pred:   tf.Tensor of shape (batch_size, pad_length, n_classes)
        """

        x = self.add_embedding()
        cell = tf.rnn.LSTMCell()
        preds = []

        hidden_state = tf.constant(tf.zeros(self.config.hidden_size))

        for t in range(self.config.pad_length):
            output, hidden_state = cell(self.input_placeholder, hidden_state)
            preds.append(output)

        pred = tf.stack(preds)

        assert preds.get_shape().as_list() == [None, self.pad_length, self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.pad_length, self.config.n_classes], preds.get_shape().as_list())
        return pred

    def add_loss_op(self, pred):
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """

        # pred_with_transition?

        cross_entropy = tf.boolean_mask(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels_placeholder,
                    logits=pred
                ),
                self.mask_placeholder
            )
        loss = tf.reduce_mean(cross_entropy)

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """

        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        train_op = opt.minimize(loss)

        return train_op

    def fit(self, sess, saver, train_examples_raw, dev_examples_raw):
        self.transition_matrix = tf.constant(compute_transition_matrix(self.config, train_examples_raw))
        return super().fit(sess, saver, train_examples_raw, dev_examples_raw)
