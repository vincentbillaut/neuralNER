import tensorflow as tf

from models.lstm_model import LSTMModel, LSTMConfig


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

        fwcell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(fwcell, input_keep_prob=1.,
                                                            output_keep_prob=1. - dropout_rate)
        bwcell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(bwcell, input_keep_prob=1.,
                                                            output_keep_prob=1. - dropout_rate)

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
        # concat_state = tf.concat(output_states, 2)

        if not self.config.extra_layer:
            inline_outputs = tf.reshape(concat_output, shape=(-1, 2 * self.config.hidden_size))
            inline_preds = self.add_extra_layer(inline_outputs, input_size=2 * self.config.hidden_size, postfix="0")
        else:
            inline_outputs = tf.reshape(concat_output, shape=(-1, 2 * self.config.hidden_size))
            inline_hidden = self.add_extra_layer(inline_outputs, input_size=2 * self.config.hidden_size, postfix="1",
                                                 output_size=self.config.other_layer_size)
            inline_preds = self.add_extra_layer(inline_hidden, input_size=self.config.other_layer_size,
                                                postfix="2")  # default : out = self.config.n_classes

        preds = tf.reshape(inline_preds,
                           shape=(tf.shape(concat_output)[0], self.config.max_length, self.config.n_classes))

        assert preds.get_shape().as_list() == [None, self.config.max_length,
                                               self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format(
            [None, self.config.max_length, self.config.n_classes], preds.get_shape().as_list())
        return preds
