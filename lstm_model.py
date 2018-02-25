import tensorflow as tf

from ner_model import NERModel


class LSTMConfig(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_features = 1
    n_classes = 17
    embed_size = 50
    hidden_size = 200
    batch_size = 128
    n_epochs = 10
    lr = 0.0005
    max_length = 120


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
        zero_vector = [0] * self.config.n_features
        zero_label = 4  # corresponds to the 'O' tag

        for sentence, labels in data:
            paddedSentence = sentence[:max_length] + [zero_vector] * (max_length - len(sentence))
            paddedLabels = labels[:max_length] + [zero_label] * (max_length - len(sentence))
            mask = [True] * min(len(sentence), max_length) + [False] * (max_length - len(sentence))
            ret.append((paddedSentence, paddedLabels, mask))
        return ret

    def preprocess_sequence_data(self, examples):
        def featurize_windows(data, start, end, window_size=1):
            """Uses the input sequences in @data to construct new windowed data points.
            """
            ret = []
            for sentence, labels in data:
                from utils.parser_utils import window_iterator
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(sum(window, []))
                ret.append((sentence_, labels))
            return ret

        examples = featurize_windows(examples, self.embedder.start_token, self.embedder.end_token)
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
        self.input_placeholder = tf.placeholder(tf.int32, (None, 1))
        self.labels_placeholder = tf.placeholder(
            tf.float32, (None, self.config.n_classes))
        self.mask_placeholder = tf.placeholder(tf.bool, [None, self.config.max_length])
        self.dropout_placeholder = tf.placeholder(tf.float32)

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
                                shape=(-1, self.config.max_length, self.config.n_features * self.config.embed_size))
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled LSTM

        Returns:
            pred:   tf.Tensor of shape (batch_size, pad_length, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        cell = tf.contrib.rnn.LSTMCell(num_units=self.config.hidden_size)
        preds = []

        for t in range(self.config.max_length):
            U = tf.get_variable("U",
                                shape=(self.config.hidden_size, self.config.n_classes),
                                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable("b2",
                                 shape=self.config.n_classes,
                                 initializer=tf.constant_initializer())
            hidden_state = tf.zeros(shape=(tf.shape(x)[0], self.config.hidden_size))

            with tf.variable_scope("LSTM"):
                for time_step in range(self.config.max_length):
                    o, hidden_state = cell(
                        tf.reshape(x[:, time_step, :], shape=(-1, self.config.n_features * self.config.embed_size)),
                        hidden_state, "LSTM")
                    tf.get_variable_scope().reuse_variables()
                    oDrop = tf.nn.dropout(o, keep_prob=dropout_rate)
                    pred = tf.matmul(oDrop, U) + b2
                    preds.append(pred)

        preds = tf.concat([tf.reshape(pred, shape=(-1, 1, self.config.n_classes)) for pred in preds], axis=1)

        assert preds.get_shape().as_list() == [None, self.config.max_length,
                                               self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format(
            [None, self.config.max_length, self.config.n_classes], preds.get_shape().as_list())
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
        # loss = tf.reduce_mean(
        #     tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_placeholder,
        #                                                                    logits=preds),
        #                     self.mask_placeholder))
        # ### END YOUR CODE
        # return loss

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
