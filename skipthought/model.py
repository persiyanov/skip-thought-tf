from __future__ import unicode_literals

import tensorflow as tf

from skipthought.encoder import Encoder
from skipthought.decoder import DecoderOutput, Decoder
from skipthought.utils import create_embeddings_matrix


class SkipThoughtModel:
    """Main class which implements skip-thought vectors model.

    # TODO: sampled softmax loss
    # TODO: grad clip

    """
    def __init__(self, num_units, embedding_size, num_tokens, pad_idx, eos_idx, embedding_matrix=None):
        self._num_units = num_units
        self._embedding_size = embedding_size
        self._num_tokens = num_tokens
        self._pad_idx = pad_idx
        self._eos_idx = eos_idx
        self._embedding_matrix = embedding_matrix

        if embedding_matrix is not None and embedding_matrix.shape.as_list() != [num_tokens, embedding_size]:
            shape = embedding_matrix.shape.as_list()
            raise ValueError("embedding_matrix must have shape=[{}, {}], you passed [{}, {}]".format(num_tokens,
                                                                                                     embedding_size,
                                                                                                     shape[0],
                                                                                                     shape[1]))

        self._build()

    def _build(self):
        with tf.variable_scope('skipthought_model'):
            if self._embedding_matrix is None:
                self._embedding_matrix = create_embeddings_matrix(self._num_tokens, self._embedding_size)

            self._encoder = Encoder(self._num_units, self._pad_idx, self._embedding_matrix)
            with tf.variable_scope('prev_decoder'):
                self._prev_decoder = Decoder(self._num_units, self._encoder.final_lstm_state,
                                             self._pad_idx, self._eos_idx, self._embedding_matrix)

            with tf.variable_scope('next_decoder'):
                self._next_decoder = Decoder(self._num_units, self._encoder.final_lstm_state,
                                             self._pad_idx, self._eos_idx, self._embedding_matrix)

    @property
    def embedding_matrix(self):
        return self._embedding_matrix

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoders(self):
        return [self._prev_decoder, self._next_decoder]

    @property
    def prev_decoder(self):
        return self._prev_decoder

    @property
    def next_decoder(self):
        return self._next_decoder

    def train(self, sess, batch):
        pass

