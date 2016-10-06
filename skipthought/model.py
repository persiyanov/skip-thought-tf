import math
import tensorflow as tf


class SkipthoughtModel(object):
    SUPPORTED_CELLTYPES = ['lstm', 'gru']

    def __init__(self, cell_type, num_hidden, embedding_size, max_vocab_size,
                 learning_rate, decay_rate, grad_clip, max_length_decoder):
        self.cell_type = cell_type
        self.max_length_decoder = max_length_decoder
        self.grad_clip = grad_clip
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.max_vocab_size = max_vocab_size
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden

        self._check_args()

        self._create_placeholders()
        self._create_network()

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

            # self.next_decoder_input = [tf.placeholder(tf.int32, [None, ], name="next_decoder_input{0}".format(i))
            #                            for i in range(self.max_length_decoder)]
            # self.next_decoder_target = [tf.placeholder(tf.int32, [None, ], name="next_decoder_target{0}".format(i))
            #                             for i in range(self.max_length_decoder)]
            # self.next_decoder_weights = [tf.placeholder(tf.float32, shape=[None,], name="next__decoder_weight{0}".format(i))
            #                              for i in range(self.max_length_decoder)]

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

        with tf.variable_scope('prev_decoder'):
            embedded_prev = [tf.nn.embedding_lookup(self.embedding_matrix, inp) for inp in self.prev_decoder_input]

            cell = cell_fn(self.num_hidden)
            cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.max_vocab_size)

            prev_decoder_outputs, _ = tf.nn.seq2seq.rnn_decoder(embedded_prev, initial_state=encoder_state,
                                                                  cell=cell)

        # with tf.variable_scope("next_decoder"):
        #     embedded_next = [tf.nn.embedding_lookup(self.embedding_matrix, inp) for inp in self.next_decoder_input]
        #
        #     cell = cell_fn(self.num_hidden)
        #     cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, self.max_vocab_size)
        #
        #     next_decoder_outputs, _ = tf.nn.seq2seq.rnn_decoder(embedded_next, initial_state=encoder_state,
        #                                                           cell=cell)

        self.prev_decoder_outputs = prev_decoder_outputs
        # self.next_decoder_outputs = next_decoder_outputs


        loss_prev = tf.nn.seq2seq.sequence_loss(prev_decoder_outputs, self.prev_decoder_target, self.prev_decoder_weights)
        # loss_next = tf.nn.seq2seq.sequence_loss(next_decoder_outputs, self.next_decoder_target, self.next_decoder_weights)
        # self.loss = loss_prev + loss_next
        self.loss = loss_prev

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        global_step = tf.Variable(0, trainable=False)
        self.train_op = optimizer.minimize(self.loss,  global_step=global_step)

    def _fill_feed_dict(self, curr, curr_seq_lengths,
                        prev_input, prev_target, prev_weights,
                        next_input, next_target, next_weights):
        assert prev_input.shape == next_input.shape == prev_weights.shape == next_weights.shape
        assert prev_input.shape[1] == self.max_length_decoder
        max_len = self.max_length_decoder

        feed_dict = {self.encoder_input: curr, self.encoder_seq_len: curr_seq_lengths}
        feed_dict.update({self.prev_decoder_input[i]: prev_input[:, i] for i in range(max_len)})
        feed_dict.update({self.prev_decoder_target[i]: prev_target[:, i] for i in range(max_len)})
        feed_dict.update({self.prev_decoder_weights[i]: prev_weights[:, i] for i in range(max_len)})

        # feed_dict.update({self.next_decoder_input[i]: next_input[:, i] for i in range(max_len)})
        # feed_dict.update({self.next_decoder_target[i]: next_target[:, i] for i in range(max_len)})
        # feed_dict.update({self.next_decoder_weights[i]: next_weights[:, i] for i in range(max_len)})
        return feed_dict

    def _check_args(self):
        if self.cell_type not in self.SUPPORTED_CELLTYPES:
            raise ValueError("This cell type is not supported.")

    def train_step(self, curr, prev, next):
        """

        :param curr:
        :param prev:
        :param next:
        :return:
        """
        feed_dict = self._fill_feed_dict(curr, prev, next)
        return self.train_op, self.loss, feed_dict

    # def predict_step(self, curr, prev, next):
    #     feed_dict = self._fill_feed_dict(curr, prev, next)
    #     return self.prev_decoder_outputs, self.next_decoder_outputs, feed_dict