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

        y = self.labels_placeholder

        cross_entropy = tf.boolean_mask(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=y,
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
