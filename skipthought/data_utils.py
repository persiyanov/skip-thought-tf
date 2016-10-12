import logging
from collections import defaultdict
from skipthought import utils


class Batch:
    def __init__(self, data, pad_value):
        """Class which creates batch from data and could be passed into
        SkipthoughtModel._fill_feed_dict_* methods.

        For encoder batches, `seq_lengths` field is used in order to fill feed_dict.
        For decoder batches, `weights` field is used in order to fill feed_dict.
        (See SkipthoughtModel code)

        Args:
            data (np.array): Encoded and padded batch.
            pad_value (int): Padding value.
        """
        self.data = data
        self.pad_value = pad_value

        self._weights = utils.get_weights_for_sequence_loss(self.data, self.pad_value)
        self._seq_lengths = utils.padded_sequence_lengths(self.data, self.pad_value)

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return self.data.__repr__()

    @property
    def weights(self):
        return self._weights

    @property
    def seq_lengths(self):
        return self._seq_lengths

    @property
    def shape(self):
        return self.data.shape


class Vocab:
    EOS_TOKEN = "<eos>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    START_VOCAB = [EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]

    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.word_freq = defaultdict(int)
        self.total_words = 0
        self.add_words(Vocab.START_VOCAB)

    @property
    def eos_token(self):
        return self.encode_word(Vocab.EOS_TOKEN)

    @property
    def pad_token(self):
        return self.encode_word(Vocab.PAD_TOKEN)

    @property
    def unk_token(self):
        return self.encode_word(Vocab.UNK_TOKEN)

    def cut_by_freq(self, max_vocab_size):
        """Removes all words except `max_vocab_size` most frequent ones.

        Args:
            max_vocab_size (int): Target vocabulary size.
        """
        for token in Vocab.START_VOCAB:
            self.word_freq.pop(token, None)
        self.word_freq = sorted(self.word_freq.items(), key=lambda x: x[1],
                                reverse=True)[:max_vocab_size - len(Vocab.START_VOCAB)]
        self.word_freq = dict(self.word_freq)
        for token in Vocab.START_VOCAB:
            self.word_freq[token] = 1
        self._id_word_mappings_from_word_freq()

    def add_word(self, word, count=1):
        if word not in self.word2index:
            index = len(self.word2index)
            self.word2index[word] = index
            self.index2word[index] = word
        self.word_freq[word] += count

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def encode_word(self, word):
        if word not in self.word2index:
            return self.word2index[Vocab.UNK_TOKEN]
        else:
            return self.word2index[word]

    def encode_words(self, words, with_eos=False, with_go=False):
        encoded = []
        if with_go:
            encoded.append(self.eos_token)
        encoded.extend([self.encode_word(w) for w in words])
        if with_eos:
            encoded.append(self.eos_token)
        return encoded

    def decode_idx(self, index):
        return self.index2word[index]

    def decode_idxs(self, indices):
        return [self.decode_idx(idx) for idx in indices]

    def _id_word_mappings_from_word_freq(self):
        words = self.word_freq.keys()
        self.index2word = dict(enumerate(words))
        self.word2index = {v: k for k, v in self.index2word.items()}

    def __len__(self):
        return len(self.word_freq)

    def __contains__(self, item):
        return item in self.word2index


class TextData:
    def __init__(self, fname, line_process_fn=lambda x: x.strip(),
                 max_vocab_size=100000, verbose=10000):
        """Class for reading text data and generating batches.

        Args:
            fname (str): File with data.
            line_process_fn (callable): Line processing function (str -> str). Use it if you want
                to do lemmatization or remove stopwords or smth. Default lambda x: x.strip()
            max_vocab_size (int): Maximum vocabulary size. Most frequent words are used.
            verbose (int): Verbosity level on reading data.
        """
        self.verbose = verbose
        self._logger = logging.getLogger(__name__)
        self.fname = fname
        self.max_vocab_size = max_vocab_size
        self.line_process_fn = line_process_fn

        self.vocab = None
        self.dataset = None
        self.max_len = None
        self.total_lines = None
        self._build_vocabulary_and_stats()
        self._build_dataset()

    def _build_vocabulary_and_stats(self):
        """Builds vocabulary, calculates maximum length and total number of
        lines in file.
        """
        with open(self.fname) as f:
            self.vocab = Vocab()
            self.total_lines = 0
            self.max_len = 0
            for line in f:
                tokens = self._tok_line(line)
                self.vocab.add_words(tokens)

                self.total_lines += 1
                self.max_len = max(self.max_len, len(tokens))
                if self.total_lines % self.verbose == 0:
                    self._logger.warning("Read\t{0} lines.".format(
                        self.total_lines))
        self.vocab.cut_by_freq(self.max_vocab_size)

    def _build_dataset(self):
        """Reads lines from file and encodes words.
        """
        with open(self.fname) as f:
            self.dataset = []
            for line in f:
                line = line.strip()
                self.dataset.append(line)

    def _tok_line(self, line):
        """Tokenizes raw line.

        Args:
            line (str): Raw line.
        Returns:
            tokens (list of str): List of tokens.
        """
        return self.line_process_fn(line).split()

    def encode_line(self, line, with_eos=False, with_go=False):
        """Encodes raw line to list of word indices. Applies ``line_process_fn`` before encoding.

        Args:
            line (str): Raw lines.
            with_eos (bool): Whether to append eos_token at the end or not.
            with_go (bool): Whether to append go_token in the beginning of line or not.
        Returns:
             encoded (list of ints): Encoded line.
        """
        tokens = self._tok_line(line)
        encoded = self.vocab.encode_words(tokens, with_eos, with_go)
        return encoded

    def encode_lines(self, lines, with_eos=False, with_go=False):
        """Encodes raw lines to list of word indices. Applies ``line_process_fn`` for each line.

        Args:
            lines (list of str): List of raw lines.
            with_eos (bool): Whether to append eos_token at the end of each line or not.
            with_go (bool): Whether to append go_token in the beginning of each line or not.
        Returns:
             encoded (list of list of ints): List of encoded lines.
        """
        encoded = [self.encode_line(line, with_eos, with_go) for line in lines]
        return encoded

    def decode_line(self, encoded_line):
        return self.vocab.decode_idxs(encoded_line)

    def make_batch(self, encoded_lines, max_len=None):
        """Makes `Batch` instance based on `encoded_lines`.

        Args:
            encoded_lines (list of list of int): List of encoded lines. Encoded lines
            can be obtained via ``encode_lines`` or ``encode_line`` methods.
            max_len (int): If not None, lines will be padded up to max_len with vocab.pad_token.
                Otherwise, lines will be padded using maximum length of line in ``encoded_lines``.
        Returns:
            batch (Batch): Batch instance.
        """
        if not max_len:
            max_len = max(map(len, encoded_lines))
        padded_lines = utils.pad_sequences(encoded_lines, max_len, self.vocab.pad_token)
        batch = Batch(padded_lines, self.vocab.pad_token)
        return batch
