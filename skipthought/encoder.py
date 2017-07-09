from __future__ import unicode_literals
from __future__ import print_function

import tensorflow as tf

from skipthought.utils import create_embeddings_matrix


class Encoder:
    """Class which implements RNN encoder. It uses tf.contrib.rnn.LSTMBlockFusedCell for speedup.


    # TODO: different cell types.
    # TODO: freeze embedding matrix.
    # TODO: several layers.

    """
    def __init__(self, num_units, embedding_size, num_tokens, pad_idx, embedding_matrix=None):
        """
        Args:
            num_units (int): Hidden state size in rnn cell.
            embedding_size (int): Embedding size.
            num_tokens (int): Vocabulary size.
            pad_idx (int): Padding token index in vocabulary.
            embedding_matrix (tf.Variable, optional): Pretrained embedding_matrix created with utils.create_embeddings_matrix.
        """
        self._num_units = num_units
        self._embedding_size = embedding_size
        self._num_tokens = num_tokens
        self._pad_idx = pad_idx
        self._embedding_matrix = embedding_matrix

        if embedding_matrix is not None and embedding_matrix.shape.as_list() != [num_tokens, embedding_size]:
            shape = embedding_matrix.shape.as_list()
            raise ValueError("embedding_matrix must have shape=[{}, {}], you passed [{}, {}]".format(num_tokens,
                                                                                                     embedding_size,
                                                                                                     shape[0],
                                                                                                     shape[1]))
        self._build()

    def _build(self):
        self._inputs = tf.placeholder(tf.int32, shape=[None, None], name='encoder_inputs')
        self._sequence_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self._inputs, self._pad_idx)), axis=1)

        with tf.variable_scope('encoder'):
            if self._embedding_matrix is None:
                self._embedding_matrix = create_embeddings_matrix(self._num_tokens, self._embedding_size)

            cell_input = tf.nn.embedding_lookup(self._embedding_matrix, ids=self._inputs)

            # Swap axis to [time, batch, embedding_size] in order to pass cell_input to fused lstm cell.
            cell_input = tf.transpose(cell_input, [1, 0, 2])
            self._cell = tf.contrib.rnn.LSTMBlockFusedCell(self._num_units)

            self._final_state = self._cell(cell_input, dtype=tf.float32, sequence_length=self._sequence_length)[1]

    @property
    def inputs(self):
        """inputs placeholder"""
        return self._inputs

    @property
    def sequence_lengths(self):
        """inputs sequence lengths placeholder"""
        return self._sequence_length

    @property
    def embedding_matrix(self):
        """tf.Variable which holds current embedding matrix tensor.
        """
        return self._embedding_matrix

    @property
    def final_cell(self):
        """Final LSTM cell state (after processing input sequence).
        """
        return self._final_state.c

    @property
    def final_hidden(self):
        """Final LSTM hidden state (after processing input sequence).
        """
        return self._final_state.h

    @property
    def final_lstm_state(self):
        """Final LSTMStateTuple (after processing input sequence).
        """
        return self._final_state

    def encode_fn(self, sess):
        """Make a callable function which takes input batch [batch_size, time]
        of word ids and returns final hidden state as numpy-array.
        """
        return sess.make_callable(self.final_hidden, feed_list=[self._inputs])

    def encode(self, sess, inputs):
        """Run session on _inputs (numpy-array [batch, time] of word ids) and return rnn final hidden state.
        """
        return sess.run(self.final_hidden, feed_dict={self._inputs: inputs})
