import numpy as np


def sequence_lengths(data, pad_value):
    """
    Return array with length of sequences. It is useful for TensorFlow RNN
    models.

    Args:
        data (numpy.array): Array with sequence word indices aligned with pad_value.
        pad_value (int): Value used for padding sequences.

    Returns:
        res (numpy.array): 1D array with sequence lengths.
    """
    assert data.ndim == 2, data
    res = np.sum((data != pad_value).astype(np.int32), axis=1)
    return res


def pad_sequences(data, max_length, pad_value):
    """
    Pad sequence of indices with pad values to the length of max_length.

    Args:
        data (lists of lists of int): List of encoded lines.
        max_length (int): Padded sequence length.
        pad_value (int): Padding value.

    Returns:
        object (numpy.array): Padded array of indices.
    """
    data = [indices + [pad_value] * (max_length - len(indices)) for indices
            in data]
    return np.array(data)


def seq_loss_weights(data, pad_value, dtype=np.float32):
    """
    Obtain weights for TensorFlow sequence loss.

    Args:
        data (numpy.array): Array with padded sentences.
        pad_value (int): Padding value.
        dtype (numpy.dtype): Weights type.

    Returns:
        object (numpy.array): Array shaped like `data`.
    """
    mask = (data != pad_value).astype(dtype)
    return mask