import logging

import tensorflow as tf

from models.lstm_model import LSTMConfig, LSTMModel

logger = logging.getLogger("NERproject")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class StackedLSTMConfig(LSTMConfig):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_features = 0
    max_length = 120

    def __init__(self, args):
        super().__init__(args)
        self.extra_layer = args.extra_layer or args.ee
        self.second_extra_layer = args.ee
        self.other_layer_size = args.other_layer_size
        if (not self.extra_layer) and (self.other_layer_size != 17):
            logger.info("Without extra layer (no -e), other_layer_size forced to 17.")
            self.other_layer_size = 17


class StackedLSTMModel(LSTMModel):
    """
    Implements a feedforward recurrent neural network with an two stacked LSTMs.
    This network will predict whether an input word is a Named Entity.
    """

    def add_prediction_op(self):
        """Adds the unrolled LSTM

        Returns:
            pred:   tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        initializer = tf.contrib.layers.xavier_initializer()

        # create 2 LSTMCells
        rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=size, initializer=initializer),
                                                    input_keep_prob=1., output_keep_prob=1. - dropout_rate)
                      for size in [self.config.hidden_size, self.config.other_layer_size]]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        # 'outputs' is a tensor of shape [batch_size, max_time, self.config.other_layer_size]
        # 'state' is a pair containing a tf.contrib.rnn.LSTMStateTuple for each cell
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=x,
                                           dtype=tf.float32)

        if self.config.extra_layer:
            inline_outputs = tf.reshape(outputs, shape=(-1, self.config.other_layer_size))
            inline_preds = self.add_extra_layer(inline_outputs, self.config.other_layer_size, '0')

            preds = tf.reshape(inline_preds,
                               shape=(tf.shape(outputs)[0], self.config.max_length, self.config.n_classes))
        else:
            preds = tf.nn.sigmoid(outputs)

        assert preds.get_shape().as_list() == [None, self.config.max_length,
                                               self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format(
            [None, self.config.max_length, self.config.n_classes], preds.get_shape().as_list())
        return preds

    def add_extra_layer(self, input_tensor, input_size, postfix, output_size=None, activation=tf.nn.sigmoid,
                        initializer=tf.contrib.layers.xavier_initializer()):
        """Creates variable and processes the input to produce the output of
        an additional hidden layer.

        output = act(x * U + b)

        Parameters
        ----------
        input_tensor : tf.Tensor
            Input tensor for the additional layer, of shape [None, input_size].
        input_size : int
            Dimension of the input.
        postfix : str
            Postfix to give to the variable names (useful if creating several
            additional layers, to be sure the parameter variables are distinct).
        output_size : int
            Desired size of input. Defaults to self.config.n_classes, to fit the
            prediction task at hand.
        activation : tf.nn.[function]
            Activation function, usually tf.nn.sigmoid, tf.nn.tanh or tf.nn.relu.
        initializer : tf.initializer?
            Initializer for weight variables.

        Returns
        -------
        tf.Tensor
            Output of the added hidden layer.

        """
        output_s = output_size or self.config.n_classes
        U = tf.get_variable("U" + postfix,
                            shape=(input_size, output_s),
                            initializer=initializer)
        b = tf.get_variable("b" + postfix + "-noreg",
                            shape=output_s,
                            initializer=tf.constant_initializer())
        return activation(tf.matmul(input_tensor, U) + b)
