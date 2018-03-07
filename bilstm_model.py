import os
import time
import tensorflow as tf
import numpy as np

from ner_model import NERModel
from lstm_model import LSTMModel, LSTMConfig
from utils.minibatches import minibatches


class BiLSTMConfig(LSTMConfig):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    # n_features = 36
    pass


class BiLSTMModel(LSTMModel):
    """
    Implements a bidirectional LSTM network with an embedding layer.
    This network will predict whether an input word is a Named Entity
    """

    def add_prediction_op(self):
        """Adds the unrolled Bi-LSTM

        Returns:
            pred:   tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        initializer = tf.contrib.layers.xavier_initializer()

        fwcell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_size, initializer=initializer)
        bwcell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_size, initializer=initializer)

        fwcell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(fwcell, input_keep_prob=1., output_keep_prob=1. - dropout_rate)
        bwcell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(bwcell, input_keep_prob=1., output_keep_prob=1. - dropout_rate)

        fwinitial_state = fwcell_with_dropout.zero_state(tf.shape(x)[0], dtype=tf.float32)
        bwinitial_state = bwcell_with_dropout.zero_state(tf.shape(x)[0], dtype=tf.float32)

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                                        fwcell_with_dropout,
                                        bwcell_with_dropout,
                                        x,
                                        initial_state_fw=fwinitial_state,
                                        initial_state_bw=bwinitial_state,
                                        time_major=False
                                        )

        concat_output = tf.concat(outputs, 2)
        concat_state = tf.concat(output_states, 2)

        #TODO maybe make the optional extra layer a *second* layer?
        # to go from concatenated outputs to prediction, there needs to be
        # an extra layer so the option as in LSTM doesn't make sense anymore
        # if True or self.config.extra_layer:

        U = tf.get_variable("U",
                            shape=(2 * self.config.hidden_size, self.config.n_classes),
                            initializer=initializer)
        b2 = tf.get_variable("b2",
                             shape=self.config.n_classes,
                             initializer=tf.constant_initializer())

        inline_outputs = tf.reshape(concat_output, shape=(-1, 2 * self.config.hidden_size))
        inline_preds = tf.nn.sigmoid(tf.matmul(inline_outputs, U) + b2)

        preds = tf.reshape(inline_preds, shape=(tf.shape(concat_output)[0], self.config.max_length, self.config.n_classes))

        assert preds.get_shape().as_list() == [None, self.config.max_length, self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.config.max_length, self.config.n_classes], preds.get_shape().as_list())
        return preds

    def add_regularization_op(self, loss, beta):
        """Adds Ops to regularize the loss function to the computational graph.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            regularized_loss: A 0-d tensor (scalar) output
        """
        # TODO: regularizers + regularized_loss
        # regularizers look like
        # regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + ...
        # with the weights being the weight parameters, model specific
        #
        # then regularized_loss = tf.reduce_mean(loss + beta * regularizers)
        return loss
