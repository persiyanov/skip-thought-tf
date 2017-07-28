from __future__ import unicode_literals
from __future__ import division

import pickle

from collections import defaultdict
from skipthought import utils


class Vocab:
    """Class which allows you conveniently build the vocabulary from your data.

     Usage:
        vcb = Vocab()

        with open('yourfile') as fin:
            lines = fin.readlines()

        # Parse tokens from lines and add them to vocabulary. You can use custom line processing function here. See doc.
        vcb.update_from_lines(lines)

        # Shrink vocabulary by removing rare tokens.
        vcb.shrink(min_count=10)

    """
    EOS_TOKEN = "<eos>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    _SPEC_TOKENS = {EOS_TOKEN, PAD_TOKEN, UNK_TOKEN}

    def __init__(self):
        self._token2index = {}
        self._index2token = {}
        self._token_freq = defaultdict(int)
        self.add_tokens(self._SPEC_TOKENS)

    @property
    def eos_idx(self):
        return self.get_index(self.EOS_TOKEN)

    @property
    def pad_idx(self):
        return self.get_index(self.PAD_TOKEN)

    @property
    def n_tokens(self):
        return len(self._token2index)

    @property
    def tokens(self):
        return self._token2index.keys()

    def add_tokens(self, tokens):
        map(self.add_token, tokens)

    def add_token(self, token):
        if token not in self:
            self._token2index[token] = self.n_tokens
            self._index2token[self.n_tokens] = token
        self._token_freq[token] += 1

    def update_from_lines(self, lines, processing_fn=lambda x: x.strip()):
        """Parse token from lines and add them to vocabulary.
        If you want to use some processing on your data, e.g. lemmatization/stemming, use `processing_fn` argument.

        Args:
            lines (list of strings): Strings to parse tokens from.
            processing_fn (callable, string -> string, optional): Custom string processing function.
                Defaults to strip().
        """
        for line in lines:
            self.add_tokens(processing_fn(line).split())

    def get_index(self, token):
        """Get token index in vocabulary. If token is not in vocabulary, index of <UNK> token is returned.
        """
        return self._token2index.get(token, self._token2index[self.UNK_TOKEN])

    def get_token(self, idx):
        """Get token by its index in vocabulary.
        """
        return self._index2token[idx]

    def __contains__(self, item):
        return self._token2index.__contains__(item)

    def shrink(self, min_count=1):
        """Remove rare tokens from vocabulary.

        Args:
            min_count (int): The threshold for removing rare tokens. Tokens which frequency is less than `min_count`
                are removed from vocabulary.
        """
        for token in self.tokens:
            if token not in self._SPEC_TOKENS and self._token_freq[token] < min_count:
                self._token_freq.pop(token)
                token_idx = self._token2index.pop(token)
                self._index2token.pop(token_idx)

    def save(self, save_to):
        """Dump vocabulary to `save_to` for future usage.
        """
        d = {
            't2i': self._token2index,
            'i2t': self._index2token,
            'tfreq': self._token_freq
        }
        with open(save_to, 'wb') as fout:
            pickle.dump(d, fout)

    def load(self, load_from):
        """Load previously saved vocabulary.
        """
        with open(load_from, 'rb') as fin:
            d = pickle.load(fin)
        self._token2index, self._index2token, self._token_freq = d['t2i'], d['i2t'], d['tfreq']

    @staticmethod
    def from_file(load_from):
        vocab = Vocab()
        vocab.load(load_from)
        return vocab


def make_batch(vocab, lines, processing_fn=lambda x: x.strip()):
    """Make np.array batch using `vocab` from list of strings.

    Args:
        vocab (data.Vocab): Vocab instance.
        lines (list of strings): Data to create batch from.
        processing_fn (callable, string -> string, optional): Custom line processing function. Defaults to strip().
    """
    lines_token_idxs = []

    max_len = 0
    for line in lines:
        line_tokens = processing_fn(line).split()
        token_idxs = [vocab.get_index(tok) for tok in line_tokens]
        max_len = max(max_len, len(token_idxs))
        lines_token_idxs.append(token_idxs)

    return utils.pad_sequences(lines_token_idxs, max_len, vocab.pad_idx)


def iterate_batches(batch_size, vocab, lines, processing_fn=lambda x: x.strip()):
    total_batches = len(lines)//batch_size

    for n_batch in range(total_batches):
        batch_lines = lines[n_batch*batch_size:(n_batch+1)*batch_size]
        yield make_batch(vocab, batch_lines, processing_fn)
