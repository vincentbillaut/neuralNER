import tensorflow as tf
import numpy as np
import logging

from ner_model import NERModel, Config
from utils.parser_utils import window_iterator

logger = logging.getLogger("NERproject")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class LSTMConfig(Config):
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
        self.extra_layer = args.extra_layer
        if (not self.extra_layer) and (self.hidden_size != self.labelsHandler.num_labels()):
            logger.info("Without extra layer (no -e), hidden_size forced to 17.")
            self.hidden_size = self.labelsHandler.num_labels()



class LSTMModel(NERModel):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict whether an input word is a Named Entity
    """

    def pad_sequences(self, data, max_length):
        """Ensures each input-output seqeunce pair in @data is of length
        @max_length by padding it with zeros and truncating the rest of the
        sequence.

        Args:
            data: is a list of (sentence, labels) tuples. @sentence is a list
                containing the words in the sentence and @label is a list of
                output labels. Each word is itself a list of
                @n_features features.
            max_length: the desired length for all input/output sequences.
        Returns:
            a new list of data points of the structure (sentence', labels', mask).
            Each of sentence', labels' and mask are of length @max_length.
        """
        ret = []

        # Use this zero vector when padding sequences.
        zero_vector = [self.embedder.null_token_id()] * (2 * self.config.n_features + 1)
        zero_label = self.labelsHandler.noneIndex()  # corresponds to the 'O' tag

        for sentence, labels in data:
            paddedSentence = np.concatenate([sentence[:max_length], [zero_vector] * (max_length - len(sentence))])
            paddedLabels = np.concatenate([labels[:max_length], [zero_label] * (max_length - len(sentence))])
            mask = [True] * min(len(sentence), max_length) + [False] * (max_length - len(sentence))
            ret.append((paddedSentence, paddedLabels, mask))
        return ret

    def preprocess_sequence_data(self, examples):
        def featurize_windows(data, start, end, window_size=1):
            """Uses the input sequences in @data to construct new windowed data points.
            """
            ret = []
            for sentence, labels in data:
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(window)  # sum(window, []))
                ret.append((sentence_, labels))
            return ret

        examples = featurize_windows(examples,
                                     self.embedder.start_token_id(), self.embedder.end_token_id(),
                                     self.config.n_features)
        return self.pad_sequences(examples, self.config.max_length)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m]  # only select elements of mask.
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        input_placeholder: Input placeholder tensor of  shape (None, 1), type tf.int32
            containing index of our word in the embedding
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout rate placeholder, scalar, type float32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        """
        self.input_placeholder = tf.placeholder(tf.int32,
                                                (None, self.config.max_length, 2 * self.config.n_features + 1))
        self.labels_placeholder = tf.placeholder(tf.int32, (None, self.config.max_length))
        self.mask_placeholder = tf.placeholder(tf.bool, [None, self.config.max_length])
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def create_feed_dict(self, inputs, mask_batch, labels_batch=None, dropout=0.):
        """Creates the feed_dict for the dependency parser.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: Dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {}
        for feed, placeholder in zip([inputs, labels_batch, mask_batch, dropout],
                                     [self.input_placeholder,
                                      self.labels_placeholder,
                                      self.mask_placeholder,
                                      self.dropout_placeholder]):
            if feed is not None:
                feed_dict[placeholder] = feed

        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors.

        Returns:
            embeddings: tf.Tensor of shape (None, 1*embed_size)
        """

        embeddingTable = tf.Variable(initial_value=self.pretrained_embeddings)
        embeddingsTensor = tf.nn.embedding_lookup(embeddingTable, self.input_placeholder)
        embeddings = tf.reshape(embeddingsTensor,
                                shape=(
                                    -1, self.config.max_length,
                                    (2 * self.config.n_features + 1) * self.config.embed_size))
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled LSTM

        Returns:
            pred:   tf.Tensor of shape (batch_size, pad_length, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        initializer = tf.contrib.layers.xavier_initializer()

        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_size, initializer=initializer)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1., output_keep_prob=1. - dropout_rate)
        initial_state = dropout_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(dropout_cell, x,
                                           initial_state=initial_state,
                                           dtype=tf.float32)

        if self.config.extra_layer:
            U = tf.get_variable("U",
                                shape=(self.config.hidden_size, self.config.n_classes),
                                initializer=initializer)
            b2 = tf.get_variable("b2",
                                 shape=self.config.n_classes,
                                 initializer=tf.constant_initializer())

            inline_outputs = tf.reshape(outputs, shape=(-1, self.config.hidden_size))
            inline_preds = tf.nn.sigmoid(tf.matmul(inline_outputs, U) + b2)
            preds = tf.reshape(inline_preds, shape=(tf.shape(outputs)[0], self.config.max_length, self.config.n_classes))
        else:
            preds = tf.nn.sigmoid(outputs)

        assert preds.get_shape().as_list() == [None, self.config.max_length,
                                               self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format(
            [None, self.config.max_length, self.config.n_classes], preds.get_shape().as_list())
        return preds

    def add_predict_onehot(self):
        return tf.argmax(self.pred, axis=2)

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
        cross_entropy = tf.boolean_mask(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder, logits=pred),
            self.mask_placeholder)
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
