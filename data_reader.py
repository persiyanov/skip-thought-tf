from collections import defaultdict

import utils
import numpy as np
import logging


class Vocab(object):
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


class SequenceDataReader(object):
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
        """
        Encodes processed line to list of word indices.
        :param line: string
        :return: list of integers
        """
        words = self._line_process_fn(line).split()
        return self.vocab.encode_words(words, with_eos=True)

    def get_data(self):
        return self.dataset

    def get_sequence_lengths(self):
        return utils.sequence_lengths(self.dataset)


class IntDataReader(object):
    def __init__(self, fname):
        self.fname = fname
        self._build_dataset()

    def _build_dataset(self):
        with open(self.fname) as f:
            self.dataset = np.array([int(l.strip()) for l in f],
                                    dtype=np.int32)

    def get_data(self, one_hot=False):
        if one_hot:
            return utils.one_hot(self.dataset)
        else:
            return self.dataset
