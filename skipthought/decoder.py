from __future__ import unicode_literals

import collections

import tensorflow as tf
from tensorflow.python.layers.core import Dense

from skipthought.utils import prepare_inputs_for_decoder


class DecoderOutput(collections.namedtuple("DecoderOutput", ("logits", "sample_ids", "sequence_length"))):
    pass


class Decoder:
    """Class which implements RNN decoder in SkipThoughtModel.

    Notes:
        1. Decoder does not use <GO> token. Instead, it uses <EOS> token as first input (see start_tokens in _build()).
        These schemes are equivalent, but ours use one token less =)

        2. You only need to pass decoder inputs for training. Targets are computed automatically.

    """

    def __init__(self, num_units, initial_state, pad_idx, eos_idx, embedding_matrix, inference_maxiter=None):
        """
        Args:
            num_units (int): Hidden state size in rnn cell.
            initial_state (LSTMStateTuple): Decoder cell initializer.
            pad_idx (int): Padding token index in vocabulary.
            eos_idx (int): End-of-sequence token index in vocabulary.
            embedding_matrix (tf.Variable): Variable which holds embedding matrix.
            inference_maxiter (int, optional): Maximum inference decoding steps.
                Useful for inference decoder to avoid infinite decoding.
                Default is None, which means decode until decoder is fully done.
        """
        self._num_units = num_units
        self._decoder_initial_state = initial_state
        self._pad_idx = pad_idx
        self._eos_idx = eos_idx
        self._embedding_matrix = embedding_matrix
        self._inference_maxiter = inference_maxiter

        self._num_tokens, self._embedding_size = self._embedding_matrix.shape.as_list()

        self._build()

    def _build(self):
        self._inputs = tf.placeholder(tf.int32, shape=[None, None], name='decoder_inputs')
        sequence_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self._inputs, self._pad_idx)), axis=1)

        with tf.variable_scope('decoder'):
            self._decoder_inputs, self._decoder_targets, self._decoder_seq_length = \
                prepare_inputs_for_decoder(self._inputs, sequence_length, self._eos_idx)

            # TRAINING DECODER ###

            # Build training decoder.
            # Embed input tokens.
            helper_inputs = tf.nn.embedding_lookup(self._embedding_matrix, ids=self._decoder_inputs)

            # Pass embeddings to helper.
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=helper_inputs,
                                                                sequence_length=self._decoder_seq_length)

            # Initialize decoder cell.
            self._cell = tf.contrib.rnn.LSTMCell(self._num_units)

            # Initialize softmax layer (actually, there is no softmax here, the layer computes logits only).
            self._output_layer = Dense(self._num_tokens, name='softmax_logits')

            # Initialize decoder which iterates over correct inputs (teacher-forced regime) and outputs logits.
            self._training_decoder = tf.contrib.seq2seq.BasicDecoder(self._cell, training_helper,
                                                                     initial_state=self._decoder_initial_state,
                                                                     output_layer=self._output_layer)

            # Unroll the decoder.
            decoded = tf.contrib.seq2seq.dynamic_decode(self._training_decoder)

            # Store unrolled logits, sampled tokens and unrolled sequence length.
            self._training_output = DecoderOutput(logits=decoded[0].rnn_output, sample_ids=decoded[0].sample_id,
                                                  sequence_length=decoded[2])

            # INFERENCE DECODER ###

            # Build inference decoder which decodes greedily.
            start_tokens = tf.ones([tf.shape(self._decoder_inputs)[0]], dtype=tf.int32) * self._eos_idx

            # Initialize inference helper passing embedding matrix, column with start tokens and eos token.
            # This helper takes token as input, embeds it with embedding matrix and outputs next token greedily.
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self._embedding_matrix, start_tokens,
                                                                        self._eos_idx)

            self._inference_decoder = tf.contrib.seq2seq.BasicDecoder(self._cell, inference_helper,
                                                                      initial_state=self._decoder_initial_state,
                                                                      output_layer=self._output_layer)
            decoded = tf.contrib.seq2seq.dynamic_decode(self._inference_decoder,
                                                        maximum_iterations=self._inference_maxiter)
            self._inference_output = DecoderOutput(logits=decoded[0].rnn_output, sample_ids=decoded[0].sample_id,
                                                   sequence_length=decoded[2])

    @property
    def inputs(self):
        """Decoder inputs placeholder. Targets are computed automatically within TF.
        """
        return self._inputs

    @property
    def seqlen(self):
        """Int32 Tensor with shape [batch_size,] representing inputs lengths.
        """
        return self._decoder_seq_length

    @property
    def targets(self):
        """Decoder targets which are computed using `inputs` placeholder.
        """
        return self._decoder_targets

    @property
    def embedding_matrix(self):
        """tf.Variable which holds current embedding matrix tensor.
        """
        return self._embedding_matrix

    @property
    def toutput(self):
        return self._training_output

    @property
    def ioutput(self):
        return self._inference_output
