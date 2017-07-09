from __future__ import unicode_literals

import collections

import tensorflow as tf
from tensorflow.python.layers.core import Dense

from skipthought.utils import prepare_inputs_for_decoder, create_embeddings_matrix


class DecoderOutput(collections.namedtuple("DecoderOutput", ("logits", "sample_ids", "sequence_length"))):
    pass


class Decoder:
    """Class which implements RNN decoder in SkipThoughtModel.

    Notes:
        Decoder does not use <GO> token. Instead, it uses <EOS> token as first input (see start_tokens in _build()).
        These schemes are equivalent, but ours use one token less =)

    # TODO: layer norm cell.
    # TODO: different cell types

    """

    def __init__(self, num_units, initial_state, pad_idx, eos_idx, embedding_matrix):
        """
        Args:
            num_units (int): Hidden state size in rnn cell.
            initial_state (LSTMStateTuple): Decoder cell initializer.
            pad_idx (int): Padding token index in vocabulary.
            eos_idx (int): End-of-sequence token index in vocabulary.
            embedding_matrix (tf.Variable): Variable which holds embedding matrix.
        """
        self._num_units = num_units
        self._decoder_initial_state = initial_state
        self._pad_idx = pad_idx
        self._eos_idx = eos_idx
        self._embedding_matrix = embedding_matrix

        self._num_tokens, self._embedding_size = self._embedding_matrix.shape.as_list()

        self._build()

    def _build(self):
        self._inputs = tf.placeholder(tf.int32, shape=[None, None], name='decoder_inputs')
        self._sequence_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self._inputs, self._pad_idx)), axis=1)

        with tf.variable_scope('decoder'):
            res = prepare_inputs_for_decoder(self._inputs, self._sequence_length, self._eos_idx)
            self._decoder_inputs, self._decoder_targets, self._decoder_seq_length = res

            # Build training decoder.
            helper_inputs = tf.nn.embedding_lookup(self._embedding_matrix, ids=self._decoder_inputs)

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=helper_inputs,
                                                                sequence_length=self._decoder_seq_length)
            self._cell = tf.contrib.rnn.LSTMCell(self._num_units)
            self._output_layer = Dense(self._num_tokens, name='softmax_logits')
            self._training_decoder = tf.contrib.seq2seq.BasicDecoder(self._cell, training_helper,
                                                                     initial_state=self._decoder_initial_state,
                                                                     output_layer=self._output_layer)
            decoded = tf.contrib.seq2seq.dynamic_decode(self._training_decoder)
            self._training_output = DecoderOutput(logits=decoded[0].rnn_output, sample_ids=decoded[0].sample_ids,
                                                  sequence_length=decoded[2])

            # Build inference decoder which decodes greedily.
            start_tokens = tf.ones([tf.shape(self._decoder_inputs)[0]], dtype=tf.int32) * self._eos_idx

            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self._embedding_matrix, start_tokens,
                                                                        self._eos_idx)
            self._inference_decoder = tf.contrib.seq2seq.BasicDecoder(self._cell, inference_helper,
                                                                      initial_state=self._decoder_initial_state,
                                                                      output_layer=self._output_layer)
            decoded = tf.contrib.seq2seq.dynamic_decode(self._inference_decoder)
            self._inference_output = DecoderOutput(logits=decoded[0].rnn_output, sample_ids=decoded[0].sample_ids,
                                                   sequence_length=decoded[2])

    @property
    def embedding_matrix(self):
        """tf.Variable which holds current embedding matrix tensor.
        """
        return self._embedding_matrix
