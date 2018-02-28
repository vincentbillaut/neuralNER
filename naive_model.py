import tensorflow as tf
import numpy as np
from ner_model import NERModel, Config


class NaiveConfig(Config):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    dropout = 0.5
    embed_size = 50


class NaiveModel(NERModel):
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
        words = np.concatenate([example[0] for example in examples])
        labels = np.concatenate([example[1] for example in examples])
        examples_with_mask = [(ex, label, mask) for ex, label, mask in zip(words, labels, iter(lambda: True, False))]
        return examples_with_mask

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        i = 0
        for sentence, labels in examples_raw:
            labels_ = preds[i:i + len(sentence)]
            i += len(sentence)
            ret.append([sentence, labels, labels_])
        return ret

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        input_placeholder: Input placeholder tensor of  shape (None, 1), type tf.int32
            containing index of our word in the embedding
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout rate placeholder, scalar, type float32
        """
        self.input_placeholder = tf.placeholder(tf.int32, (None, ))
        self.labels_placeholder = tf.placeholder(
            tf.int32, None)
        self.mask_placeholder = tf.placeholder(tf.bool, None)
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

        init_embed = tf.Variable(initial_value=self.pretrained_embeddings)
        embeddings0 = tf.nn.embedding_lookup(
            params=init_embed, ids=self.input_placeholder)
        embeddings = tf.reshape(
            tensor=embeddings0, shape=[-1, 1 * self.config.embed_size])

        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """

        x = self.add_embedding()

        init = tf.contrib.layers.xavier_initializer()

        W = tf.Variable(init(shape=[self.config.embed_size, self.config.hidden_size]))
        U = tf.Variable(init(shape=[self.config.hidden_size, self.config.n_classes]))

        b1 = tf.Variable(tf.zeros((1, self.config.hidden_size)), "b1")
        b2 = tf.Variable(tf.zeros((1, self.config.n_classes)), "b2")

        h = tf.nn.relu(tf.matmul(x, W) + b1)
        h_drop = tf.nn.dropout(h, keep_prob=(1 - self.dropout_placeholder))
        pred = tf.matmul(h_drop, U) + b2

        return pred

    def add_predict_onehot(self):
        return tf.argmax(self.pred, axis=1)

    def add_predict_proba(self):
        return tf.nn.softmax(self.pred, axis=1)

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

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels_placeholder,
            logits=pred
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
