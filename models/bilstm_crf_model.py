import numpy as np
import tensorflow as tf

from models.bilstm_model import BiLSTMModel, BiLSTMConfig


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
    transit = np.zeros((config.n_classes + 2, config.n_classes + 2))
    for _, labels in examples:
        transit[0, labels[0]] += 1
        for i in range(len(labels) - 1):
            transit[labels[i] + 1, labels[i + 1] + 1] += 1
        transit[labels[-1], config.n_classes + 1] += 1
    transit = (transit.T * 1.0 / transit.sum(axis=1)).T
    return transit


class BiLSTMCRFConfig(BiLSTMConfig):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    pass


class BiLSTMCRFModel(BiLSTMModel):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict whether an input word is a Named Entity
    """

    def __init__(self):
        raise NotImplementedError("Model to be finished before use.")

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

        # pred_with_transition?
        # y_i = softmax(...)
        # y_i+1 = softmax(...)
        # loss += y_i.T * A * y_i+1
        #
        # If we don't consider it one-hot, the pred is basically
        #   - two dimensional (batch_size, max_length)
        #   - for ex: pred[i,:] = [1,1,16,0,9,16] (elements are in [0, n_classes-1])

        cross_entropy = tf.boolean_mask(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y,
                logits=pred
            ),
            self.mask_placeholder
        )
        loss = tf.reduce_mean(cross_entropy)

        return loss

    def fit(self, sess, saver, train_examples_raw, dev_examples_raw):
        self.transition_matrix = tf.constant(compute_transition_matrix(self.config, train_examples_raw))
        return super().fit(sess, saver, train_examples_raw, dev_examples_raw)
