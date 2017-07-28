from __future__ import unicode_literals

import tensorflow as tf

from skipthought.encoder import Encoder
from skipthought.decoder import Decoder
from skipthought.utils import create_embeddings_matrix

import numpy as np


class SkipThoughtModel:
    """Main class which implements skip-thought vectors model.

    Usage:
        sktm = SkipThoughtModel(<your configuration>)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Training loop
        for (inputs_batch, prev_batch, next_batch) in your_dataset:
            batch_loss = sktm.train(sess, inputs_batch, prev_batch, next_batch)

        # Infer embeddings
        embeddings = sktm.encode(sess, inputs)

        # Apply prev/next decoders
        prev_output, next_output = sktm.decode(sess, inputs)

        # Save model
        sktm.save(sess, <save_path>)

        # Restore model
        sktm.restore(sess, <save_path>)


    Notes:
        1. You don't need to append/prepend your data with eos/go tokens. You only have to pad sequences.
            Everything else is computed within TF.

        2. For creating padded batches you may use `data.make_batch()` function.



    # TODO: sampled softmax loss
    # TODO: grad clip
    # TODO: layer norm cell.
    # TODO: different cell types.
    # TODO: freeze embedding matrix.
    # TODO: several layers.

    """
    def __init__(self, num_units, embedding_size, num_tokens, pad_idx, eos_idx, embedding_matrix=None, optimizer=None):
        """
        Args:
            num_units (int): Hidden cell size.
            embedding_size (int): Embedding size. If embedding_matrix is provided, this value must be equal
                to the second dimension of the matrix.
            num_tokens (int): Size of vocabulary (including pad/eos tokens). If embedding_matrix is provided,
                this value must be equal to the first dimension of the matrix.
            pad_idx (int): Vocabulary index of <PAD> token.
            eos_idx (int): Vocabulary index of <EOS> token.
            embedding_matrix (numpy.array, optional): If provided, is used for initializing vocabulary embeddings.
                Defaults to xavier initialization.
            optimizer (tf.train.Optimizer, optional): Custom optimizer to use. Defaults to AdamOptimizer with
                default config.
        """
        self._num_units = num_units
        self._embedding_size = embedding_size
        self._num_tokens = num_tokens
        self._pad_idx = pad_idx
        self._eos_idx = eos_idx
        self._embedding_matrix = embedding_matrix
        self._optimizer = optimizer or tf.train.AdamOptimizer()

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

            self._loss = (self._get_loss(self.prev_decoder) + self._get_loss(self.next_decoder)) / 2.

            self._global_step = tf.Variable(0, name='global_step', trainable=False)
            self._train_op = self._optimizer.minimize(self._loss, global_step=self._global_step)

        self._saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    @classmethod
    def _get_loss(cls, decoder):
        seq_mask = tf.sequence_mask(decoder.seqlen, dtype=tf.float32)
        return tf.contrib.seq2seq.sequence_loss(decoder.toutput.logits, decoder.targets, seq_mask)

    def _decode_from(self, sess, inputs, decoder):
        batch_size = inputs.shape[0]
        return sess.run(decoder.ioutput, feed_dict={
            decoder.inputs: np.zeros((batch_size, 1)),
            self.encoder.inputs: inputs
        })

    @property
    def embedding_matrix(self):
        return self._embedding_matrix

    @property
    def encoder(self):
        return self._encoder

    @property
    def prev_decoder(self):
        return self._prev_decoder

    @property
    def next_decoder(self):
        return self._next_decoder

    @property
    def inputs(self):
        return self.encoder.inputs, self.prev_decoder.inputs, self.next_decoder.inputs

    def encode(self, sess, inputs):
        """Encode batch of inputs and return embedding tensor with shape [batch_size, num_units].
        """
        return self.encoder.encode(sess, inputs)

    def decode(self, sess, inputs):
        """Decode prev and next from inputs and return tuple of decoder outputs.
        Returns:
            (prev_output, next_output) -- where each item is decoder.DecoderOutput
        """
        return self._decode_from(sess, inputs, self.prev_decoder), self._decode_from(sess, inputs, self.next_decoder)

    def train(self, sess, inputs, prev, next):
        return sess.run([self._loss, self._train_op], feed_dict=dict(zip(self.inputs, [inputs, prev, next])))[0]

    def save(self, sess, save_path):
        self._saver.save(sess, save_path, global_step=tf.train.global_step(sess, self._global_step))

    def restore(self, sess, save_path=None):
        save_path = save_path or self._saver.last_checkpoints[-1]
        self._saver.restore(sess, save_path)
