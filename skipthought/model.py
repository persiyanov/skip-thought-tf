import math
import tensorflow as tf
import numpy as np
import logging

from . import data_utils


class SkipthoughtModel:

    SUPPORTED_CELLTYPES = ['lstm', 'gru']

    def __init__(self, cell_type, num_hidden, num_layers, embedding_size, max_vocab_size,
                 learning_rate, decay_rate, decay_steps, grad_clip, num_samples, max_length_decoder):
        self.cell_type = cell_type
        self.max_length_decoder = max_length_decoder
        self.grad_clip = grad_clip
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.lr = learning_rate
        self.max_vocab_size = max_vocab_size
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_samples = num_samples
        self._logger = logging.getLogger(__name__)

        self._logger.info("Creating SkipthoughModel.")
        self._check_args()
        self._logger.info("Checked args.")

        if self.cell_type == 'lstm':
            self.cell_fn = lambda x: tf.nn.rnn_cell.BasicLSTMCell(x, state_is_tuple=True)
        elif self.cell_type == 'gru':
            self.cell_fn = tf.nn.rnn_cell.GRUCell

        self._create_placeholders()
        self._logger.info("Created placeholders.")
        self._create_network()
        self._logger.info("Created SkipthoughtModel.")

    def _check_args(self):
        if self.cell_type not in self.SUPPORTED_CELLTYPES:
            raise ValueError("This cell type is not supported.")
        if self.num_samples <= 0 or self.num_samples > self.max_vocab_size:
            raise ValueError("num_samples must be greater than zero and leq than max_vocab_size")

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

    def _create_decoder(self, scope_name, encoder_state, decoder_input):
        with tf.variable_scope(scope_name):
            embedded_prev = [tf.nn.embedding_lookup(self.embedding_matrix, inp) for inp in decoder_input]

            cell = self.cell_fn(self.num_hidden)

            # cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.max_vocab_size)

            w_t = tf.get_variable("proj_w", [self.max_vocab_size, self.num_hidden])
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.max_vocab_size])
            output_projection = (w, b)

            decoder_outputs, _ = tf.nn.seq2seq.rnn_decoder(embedded_prev, initial_state=encoder_state,
                                                           cell=cell)

        loop_function_predict = tf.nn.seq2seq._extract_argmax_and_embed(self.embedding_matrix, output_projection=(w, b), update_embedding=False)
        with tf.variable_scope(scope_name, reuse=True):
            decoder_predict_hiddens, _ = tf.nn.seq2seq.rnn_decoder(embedded_prev, initial_state=encoder_state,
                                                                       cell=cell, loop_function=loop_function_predict)

            decoder_predict_logits = [tf.nn.xw_plus_b(x, w, b) for x in decoder_predict_hiddens]
        return decoder_outputs, decoder_predict_logits, output_projection

    def _create_encoder(self, embedded, cudnn=False):
        with tf.variable_scope('encoder'):
            if cudnn:
                if self.cell_type == 'lstm':
                    pass
                    # lstm_cell = tf.contrib.cudnn_rnn.CudnnLSTM(self.num_layers, self.num_hidden, self.embedding_size)
                    # params_size_t = lstm_cell.params_size()
                    # input_h = tf.Variable(tf.ones([self.num_layers, 64, self.num_hidden]), name='input_h')
                    # input_c = tf.Variable(tf.ones([self.num_layers, 64, self.num_hidden]), name='input_c')
                    # params = tf.Variable(tf.ones([params_size_t]), validate_shape=False, name='params_lstm')
                    # lstm_cell(is_training=True, )
                else:
                    pass
                    # tf.contrib.cudnn_rnn.CudnnGRU(self.num_layers, self.num_hidden, self.embedding_size)
            else:
                cell = self.cell_fn(self.num_hidden)
                if self.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)
                encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, dtype=tf.float32,
                                                                  inputs=embedded,
                                                                  sequence_length=self.encoder_seq_len,
                                                                  swap_memory=True)
                if self.num_layers == 1:
                    encoder_state = encoder_state
                else:
                    assert isinstance(encoder_state, tuple)
                    encoder_state = encoder_state[-1]
        self._logger.info("Encoder done")
        return encoder_state

    def _create_network(self):
        self._logger.info("Create computational graph")
        with tf.variable_scope('embeddings'):
            # Default initializer for embeddings should have variance=1.
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
            self.embedding_matrix = tf.get_variable("embedding_matrix",
                                                    shape=[self.max_vocab_size, self.embedding_size],
                                                    initializer=initializer)
            embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.encoder_input)
        self._logger.info("Embeddings done")

        self.encoder_state = self._create_encoder(embedded, cudnn=False)

        prev_decoder_outputs, prev_decoder_predict_logits, prev_decoder_output_proj = \
            self._create_decoder("prev_decoder", self.encoder_state, self.prev_decoder_input)
        self._logger.info("Prev decoder done")

        next_decoder_outputs, next_decoder_predict_logits, next_decoder_output_proj = \
            self._create_decoder("next_decoder", self.encoder_state, self.next_decoder_input)
        self._logger.info("Next decoder done")

        self.prev_decoder_outputs = prev_decoder_outputs
        self.prev_decoder_predict_logits = prev_decoder_predict_logits
        self.prev_decoder_predict = [tf.argmax(logit, 1) for logit in self.prev_decoder_predict_logits]

        self.next_decoder_outputs = next_decoder_outputs
        self.next_decoder_predict_logits = next_decoder_predict_logits
        self.next_decoder_predict = [tf.argmax(logit, 1) for logit in self.next_decoder_predict_logits]

        def get_sampled_loss(w, b):
            w_t = tf.transpose(w)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                                  self.num_samples, self.max_vocab_size)
            return sampled_loss

        prev_sampled_loss = get_sampled_loss(*prev_decoder_output_proj)
        next_sampled_loss = get_sampled_loss(*next_decoder_output_proj)
        loss_prev = tf.nn.seq2seq.sequence_loss(prev_decoder_outputs,
                                                self.prev_decoder_target,
                                                self.prev_decoder_weights,
                                                softmax_loss_function=prev_sampled_loss)
        loss_next = tf.nn.seq2seq.sequence_loss(next_decoder_outputs,
                                                self.next_decoder_target,
                                                self.next_decoder_weights,
                                                softmax_loss_function=next_sampled_loss)
        self.loss = loss_prev + loss_next

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        self.train_op = optimizer.apply_gradients(zip(clipped_grads, tvars), global_step=global_step)
        self._logger.info("Loss and optimizer done")

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

    def encode(self, curr):
        feed_dict = self._fill_feed_dict_predict(curr)
        return self.encoder_state, feed_dict

    def predict(self, curr):
        feed_dict = self._fill_feed_dict_predict(curr)
        return self.prev_decoder_predict, self.next_decoder_predict, feed_dict
