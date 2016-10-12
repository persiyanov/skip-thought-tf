import math
import tensorflow as tf
import numpy as np

from . import data_utils


class SkipthoughtModel:

    SUPPORTED_CELLTYPES = ['lstm', 'gru']

    def __init__(self, cell_type, num_hidden, embedding_size, max_vocab_size,
                 learning_rate, decay_rate, decay_steps, grad_clip, max_length_decoder):
        self.cell_type = cell_type
        self.max_length_decoder = max_length_decoder
        self.grad_clip = grad_clip
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.lr = learning_rate
        self.max_vocab_size = max_vocab_size
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden

        self._check_args()

        self._create_placeholders()
        self._create_network()

    def _check_args(self):
        if self.cell_type not in self.SUPPORTED_CELLTYPES:
            raise ValueError("This cell type is not supported.")

    def _create_placeholders(self):
        with tf.variable_scope('placeholders'):
            self.encoder_input = tf.placeholder(tf.int32, [None, None], name='encoder_input')
            self.encoder_seq_len = tf.placeholder(tf.int32, [None, ], name='encoder_sequence_lengths')

            self.prev_decoder_input = [tf.placeholder(tf.int32, [None, ], name="prev_decoder_input{0}".format(i))
                                       for i in range(self.max_length_decoder)]
            self.prev_decoder_target = [tf.placeholder(tf.int32, [None, ], name="prev_decoder_target{0}".format(i))
                                        for i in range(self.max_length_decoder)]
            self.prev_decoder_weights = [tf.placeholder(tf.float32, shape=[None,], name="prev__decoder_weight{0}".format(i))
                                         for i in range(self.max_length_decoder)]

            self.next_decoder_input = [tf.placeholder(tf.int32, [None, ], name="next_decoder_input{0}".format(i))
                                       for i in range(self.max_length_decoder)]
            self.next_decoder_target = [tf.placeholder(tf.int32, [None, ], name="next_decoder_target{0}".format(i))
                                        for i in range(self.max_length_decoder)]
            self.next_decoder_weights = [tf.placeholder(tf.float32, shape=[None,], name="next__decoder_weight{0}".format(i))
                                         for i in range(self.max_length_decoder)]

    def _create_network(self):
        with tf.variable_scope('embeddings'):
            # Default initializer for embeddings should have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
            self.embedding_matrix = tf.get_variable("embedding_matrix",
                                                    shape=[self.max_vocab_size, self.embedding_size],
                                                    initializer=initializer)
            embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.encoder_input)

        if self.cell_type == 'lstm':
            cell_fn = lambda x: tf.nn.rnn_cell.BasicLSTMCell(x, state_is_tuple=True)
        elif self.cell_type == 'gru':
            cell_fn = lambda x: tf.nn.rnn_cell.GRUCell

        with tf.variable_scope('encoder'):
            cell = cell_fn(self.num_hidden)
            encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, dtype=tf.float32,
                                                              inputs=embedded,
                                                              sequence_length=self.encoder_seq_len)

        loop_function_predict = tf.nn.seq2seq._extract_argmax_and_embed(self.embedding_matrix, update_embedding=False)
        with tf.variable_scope('prev_decoder'):
            embedded_prev = [tf.nn.embedding_lookup(self.embedding_matrix, inp) for inp in self.prev_decoder_input]

            cell = cell_fn(self.num_hidden)
            cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.max_vocab_size)

            prev_decoder_outputs, _ = tf.nn.seq2seq.rnn_decoder(embedded_prev, initial_state=encoder_state,
                                                                cell=cell)

        with tf.variable_scope("prev_decoder", reuse=True):
            prev_decoder_predict_logits, _ = tf.nn.seq2seq.rnn_decoder(embedded_prev, initial_state=encoder_state,
                                                                       cell=cell, loop_function=loop_function_predict)

        with tf.variable_scope("next_decoder"):
            embedded_next = [tf.nn.embedding_lookup(self.embedding_matrix, inp) for inp in self.next_decoder_input]

            cell = cell_fn(self.num_hidden)
            cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.max_vocab_size)

            next_decoder_outputs, _ = tf.nn.seq2seq.rnn_decoder(embedded_next, initial_state=encoder_state,
                                                                cell=cell)

        with tf.variable_scope("next_decoder", reuse=True):
            next_decoder_predict_logits, _ = tf.nn.seq2seq.rnn_decoder(embedded_next, initial_state=encoder_state,
                                                                       cell=cell, loop_function=loop_function_predict)

        self.prev_decoder_outputs = prev_decoder_outputs
        self.prev_decoder_predict_logits = prev_decoder_predict_logits
        self.prev_decoder_predict = [tf.argmax(logit, 1) for logit in self.prev_decoder_predict_logits]

        self.next_decoder_outputs = next_decoder_outputs
        self.next_decoder_predict_logits = next_decoder_predict_logits
        self.next_decoder_predict = [tf.argmax(logit, 1) for logit in self.next_decoder_predict_logits]

        loss_prev = tf.nn.seq2seq.sequence_loss(prev_decoder_outputs, self.prev_decoder_target, self.prev_decoder_weights)
        loss_next = tf.nn.seq2seq.sequence_loss(next_decoder_outputs, self.next_decoder_target, self.next_decoder_weights)
        self.loss = loss_prev + loss_next

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        self.train_op = optimizer.apply_gradients(zip(clipped_grads, tvars), global_step=global_step)

    def _fill_feed_dict_train(self, enc_inp,
                              prev_inp, prev_targ,
                              next_inp, next_targ):
        """Fills feed dictionary.

        Args:
            enc_inp (data_utils.Batch): Encoder input batch.
            prev_inp (data_reader.Batch): Prev decoder input batch.
            prev_targ (data_reader.Batch): Prev decoder target batch.
            next_inp (data_reader.Batch): Next decoder input batch.
            next_targ (data_reader.Batch): Next decoder target batch.
        Returns:
            feed_dict (dict): Feed dictionary.
        """
        assert prev_inp.shape == prev_targ.shape == next_inp.shape == next_targ.shape
        assert prev_inp.shape[1] == self.max_length_decoder
        max_len = self.max_length_decoder

        feed_dict = {self.encoder_input: enc_inp.data, self.encoder_seq_len: enc_inp.seq_lengths}
        feed_dict.update({self.prev_decoder_input[i]: prev_inp.data[:, i] for i in range(max_len)})
        feed_dict.update({self.prev_decoder_target[i]: prev_targ.data[:, i] for i in range(max_len)})
        feed_dict.update({self.prev_decoder_weights[i]: prev_targ.weights[:, i] for i in range(max_len)})

        feed_dict.update({self.next_decoder_input[i]: next_inp.data[:, i] for i in range(max_len)})
        feed_dict.update({self.next_decoder_target[i]: next_targ.data[:, i] for i in range(max_len)})
        feed_dict.update({self.next_decoder_weights[i]: next_targ.weights[:, i] for i in range(max_len)})
        return feed_dict

    def _fill_feed_dict_predict(self, curr):
        feed_dict = {self.encoder_input: curr.data, self.encoder_seq_len: curr.seq_lengths,
                     self.prev_decoder_input[0]: np.array([curr.go_value]),
                     self.next_decoder_input[0]: np.array([curr.go_value])}
        return feed_dict

    def train_step(self, enc_inp, prev_inp, prev_targ, next_inp, next_targ):
        """Returns train_op, loss and feed_dict for performing sess.run(...) on them.

        Args:
            enc_inp (data_utils.Batch): Encoder input with a shape [batch_size, batch_length].
                Batch length can vary from batch to batch.
            prev_inp (data_reader.Batch): Prev decoder input with a shape [batch_size, self.max_decoder_length]
            prev_targ (data_reader.Batch): Prev decoder target with a shape [batch_size, self.max_decoder_length]
            next_inp (data_reader.Batch): Next decoder input with a shape [batch_size, self.max_decoder_length]
            next_targ (data_reader.Batch): Next decoder target with a shape [batch_size, self.max_decoder_length]
        Returns:
            (self.train_op, self.loss, feed_dict)
        """
        feed_dict = self._fill_feed_dict_train(enc_inp, prev_inp, prev_targ, next_inp, next_targ)
        return self.train_op, self.loss, feed_dict

    def predict(self, curr):
        feed_dict = self._fill_feed_dict_predict(curr)
        return self.prev_decoder_predict, self.next_decoder_predict, feed_dict
