import numpy as np


def one_hot(labels, num_of_classes=None):
    """
    Create one-hot representation of class labels stored in ``labels``.
    Labels must be zero-based.

    Args:
        labels (list or numpy.array): Array-like with labels.
        num_of_classes (int): Total number of classes.
        
    Returns:
        y (numpy.array): Array with one-hot encoded labels.
    """
    if not num_of_classes:
        num_of_classes = max(labels) + 1
    y = np.zeros((len(labels), num_of_classes), dtype=np.int32)
    y[np.arange(len(labels)), labels] = 1
    return y


def sequence_lengths(data):
    """
    Return array with length of sequences. It is useful for TensorFlow RNN
    models.

    Args:
        data (list of list of int): List with encoded lines with different
            lengths.

    Returns:
        res (numpy.array): 1D array with sequence lengths.
    """
    lengths = list(map(lambda x: len(x), data))
    return np.array(lengths)


def padded_sequence_lengths(data, pad_value):
    """
    Return array with length of sequences. It is useful for TensorFlow RNN
    models.

    Args:
        data (numpy.array): Array with sequence word indices.
        pad_value (float): Value used for padding sequences.

    Returns:
        res (numpy.array): 1D array with sequence lengths.
    """
    res = np.sum((data != pad_value).astype(np.int32), axis=1)
    return res


def pad_sequences(data, max_length, pad_value):
    """
    Pad sequence of indices with pad values to the length of max_length.

    Args:
        data (lists of lists of int): List of encoded lines.
        max_length (int): Padded sequence length.
        pad_value (float): Padding value.

    Returns:
        object (numpy.array): Padded array of indices.
    """
    data = [indices + [pad_value] * (max_length - len(indices)) for indices
            in data]
    return np.array(data)


def get_weights_for_sequence_loss(data, pad_value, dtype=np.float32):
    """
    Obtain weights for TensorFlow sequence loss.

    Args:
        data (numpy.array): Array with padded sentences.
        pad_value (float): Padding value.
        dtype (numpy.dtype): Weights type.

    Returns:
        object (numpy.array): Array shaped like `data`.
    """
    mask = (data != pad_value).astype(dtype)
    return mask


def data_iterator(orig_X, orig_y=None, batch_size=64,
                  labels_one_hot=False, shuffle=False):
    """Creates iterator which yields batch_size chunks.
    Code was taken from cs224d course.

    Args:
        orig_X (numpy.array): Array with samples.
        orig_y (numpy.array): Array with labels.
        batch_size (int): Size of batch.
        labels_one_hot (bool): Whether to do one-hot encoding of labels or not.
        shuffle (bool): Whether to shuffle data or not.

    Yields:
        (x, y) or x: Pair where x is a `numpy.array` with shape [
        `batch_size`, `orig_X.shape[1]`], y is a `numpy.array` with shape [
        `batch_size`, `orig_y.shape[1]`]
    """
    # Optionally shuffle the data before training
    if shuffle:
        indices = np.random.permutation(len(orig_X))
        data_X = orig_X[indices]
        data_y = orig_y[indices] if np.any(orig_y) else None
    else:
        data_X = orig_X
        data_y = orig_y
    ###
    total_processed_examples = 0
    total_steps = int(np.ceil(len(data_X) / float(batch_size)))
    for step in range(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        x = data_X[batch_start:batch_start + batch_size]
        # Convert our target from the class index to a one hot vector
        if np.any(data_y):
            y = data_y[batch_start:batch_start + batch_size]
            if labels_one_hot:
                y = one_hot(y, num_of_classes=max(data_y) + 1)
            yield x, y
        else:
            yield x
        ###
        total_processed_examples += len(x)
    # Sanity check to make sure we iterated over all the dataset as intended
    assert total_processed_examples == len(data_X), \
        'Expected {} and processed {}'.format(len(data_X),
                                              total_processed_examples)


def seq2seq_data_iterator(orig_X, orig_y, batch_size=64, shuffle=False,
                          pad_value=0):
    """Creates iterator for sequence-to-sequence data.

    Args:
        orig_X (list of lists of ints): List with encoded origin lines. Lines
            can be with different lengths.
        orig_y (list of lists of ints): List with encoded target lines.
            Lines can be with different lengths.
        batch_size (int): Size of batch.
        shuffle (bool): Whether to shuffle data or not.
        pad_value (int): Padding value.

    Yields:
        res (tuple of numpy.array): Padded batch for origin lines and batch
            for target lines. They are padded with maximum line length in
            batch.
    """
    if shuffle:
        indices = np.random.permutation(len(orig_X))
        data_X = [orig_X[i] for i in indices]
        data_y = [orig_y[i] for i in indices]
    else:
        data_X = orig_X
        data_y = orig_y

    total_processed_examples = 0
    total_steps = int(np.ceil(len(data_X)) / float(batch_size))
    for step in range(total_steps+1):
        batch_start = step * batch_size
        x = data_X[batch_start:batch_start + batch_size]
        x_seq_lengths = sequence_lengths(x)
        x = pad_sequences(x, max(x_seq_lengths), pad_value)

        y = data_y[batch_start:batch_start + batch_size]
        y_seq_lengths = sequence_lengths(y)
        y = pad_sequences(y, max(y_seq_lengths), pad_value)
        yield x, y

        total_processed_examples += len(x)
    # Sanity check to make sure we iterated over all the dataset as intended
    assert total_processed_examples == len(data_X), \
        'Expected {} and processed {}'.format(len(data_X),
                                              total_processed_examples)


def seq2seq_triples_data_iterator(curr_data, prev_data, next_data, max_len,
                                  batch_size=64, shuffle=False, pad_value=0):
    """Creates iterator for (current sentence, prev sentence, next sentence)
    data. Is is useful for training skip-thought vectors.

    Args:
        curr_data (list of lists of ints): List with encoded current
            sentences. Lines can be with different lengths.
        prev_data (list of lists of ints): List with encoded  previous
            lines. Lines can be with different lengths.
        next_data (list of lists of ints): List with encoded next lines.
            Lines can be with different lengths.
        max_len (int): Maximum length for padding previous and next sentences.
        batch_size (int): Size of batch.
        shuffle (bool): Whether to shuffle data or not.
        pad_value (int): Padding value.

    """
    if shuffle:
        indices = np.random.permutation(len(curr_data))
        curr_data = [curr_data[i] for i in indices]
        prev_data = [prev_data[i] for i in indices]
        next_data = [next_data[i] for i in indices]

    total_processed_examples = 0
    total_steps = int(np.ceil(len(curr_data)) / float(batch_size))
    for step in range(total_steps+1):
        batch_start = step * batch_size
        curr = curr_data[batch_start:batch_start + batch_size]
        x_seq_lengths = sequence_lengths(curr)
        curr = pad_sequences(curr, max(x_seq_lengths), pad_value)

        prev = prev_data[batch_start:batch_start + batch_size]
        prev = pad_sequences(prev, max_len, pad_value)

        next = next_data[batch_start:batch_start + batch_size]
        next = pad_sequences(next, max_len, pad_value)
        yield curr.astype(np.int32),\
              prev.astype(np.int32),\
              next.astype(np.int32)

        total_processed_examples += len(curr)
    # Sanity check to make sure we iterated over all the dataset as intended
    assert total_processed_examples == len(curr_data), \
        'Expected {} and processed {}'.format(len(curr_data),
                                              total_processed_examples)