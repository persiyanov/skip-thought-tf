import logging
from collections import defaultdict
from skipthought import utils


class Batch:
    def __init__(self, data, pad_value=None):
        self.data = data
        self.pad_value = pad_value

        self._weights = None
        self._seq_lengths = None
        if self.pad_value is not None:
            self._weights = utils.get_weights_for_sequence_loss(self.data, self.pad_value)
            self._seq_lengths = utils.padded_sequence_lengths(self.data, self.pad_value)

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return self.data.__repr__()

    @property
    def weights(self):
        assert self._weights is not None
        return self._weights

    @property
    def seq_lengths(self):
        assert self._weights is not None
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

    def encode_words(self, words, with_eos=False):
        encoded = []
        encoded.extend([self.encode_word(w) for w in words])
        if with_eos:
            encoded.append(self.encode_word(Vocab.EOS_TOKEN))
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


class SequenceDataReader:
    def __init__(self, fname,
                 line_process_fn=lambda x: x.lower().strip(),
                 max_vocab_size=100000, verbose=10000):
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)
        self.fname = fname
        self.max_vocab_size = max_vocab_size
        self._line_process_fn = line_process_fn
        self._build_vocabulary_and_stats()
        self._build_dataset()

    def _build_vocabulary_and_stats(self):
        """
        Fills vocabulary, calculates maximum length and total number of
        lines in file.
        """
        with open(self.fname) as f:
            self.vocab = Vocab()
            self.total_lines = 0
            for line in f:
                line = self._line_process_fn(line)
                words = line.split()
                self.vocab.add_words(words)

                self.total_lines += 1
                if self.total_lines % self._verbose == 0:
                    self._logger.warning("Read\t{0} lines.".format(
                        self.total_lines))
        self.vocab.cut_by_freq(self.max_vocab_size)

    def _build_dataset(self):
        """
        Read lines from file and encodes words.
        """
        with open(self.fname) as f:
            self.max_len = 0
            self.dataset = []
            for line in f:
                line = line.strip()
                encoded = self._encode(line.strip())
                self.dataset.append(encoded)
                self.max_len = max(self.max_len, len(encoded))

    def _encode(self, line):
        """Encodes processed line to list of word indices.

        Args:
            line (str): Raw line.
        Returns:
             encoded_words (list of ints): List of encoded words + encoded EOS_TOKEN at the end.
        """
        words = self._line_process_fn(line).split()
        return self.vocab.encode_words(words, with_eos=True)

    def get_data(self):
        return self.dataset

    def get_sequence_lengths(self):
        return utils.sequence_lengths(self.dataset)

