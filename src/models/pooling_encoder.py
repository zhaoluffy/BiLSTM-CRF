import numpy as np
import tensorflow as tf


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 of
    End-To-End Memory Networks (https://arxiv.org/abs/1503.08895).

    Args:
      sentence_size: length of the sentence
      embedding_size: dimensionality of the embeddings

    Returns:
      A numpy array of shape [sentence_size, embedding_size] containing
      the fixed position encodings for each sentence position.
    """
    encoding = np.ones((sentence_size, embedding_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for k in range(1, le):
        for j in range(1, ls):
            encoding[j-1, k-1] = (1.0 - j/float(ls)) - (
                                                           k / float(le)) * (1. - 2. * j/float(ls))
    return encoding


def _create_position_embedding(embedding_dim, num_positions, lengths, maxlen):
    """Creates position embeddings.

    Args:
      embedding_dim: Dimensionality of the embeddings. An integer.
      num_positions: The number of positions to be embedded. For example,
        if you have inputs of length up to 100, this should be 100. An integer.
      lengths: The lengths of the inputs to create position embeddings for.
        An int32 tensor of shape `[batch_size]`.
      maxlen: The maximum length of the input sequence to create position
        embeddings for. An int32 tensor.

    Returns:
      A tensor of shape `[batch_size, maxlen, embedding_dim]` that contains
      embeddings for each position. All elements past `lengths` are zero.
    """
    # Create constant position encodings
    position_encodings = tf.constant(
        position_encoding(num_positions, embedding_dim),
        name="position_encoding")

    # Slice to size of current sequence
    pe_slice = position_encodings[:maxlen, :]
    # Replicate encodings for each element in the batch
    batch_size = tf.shape(lengths)[0]
    pe_batch = tf.tile([pe_slice], [batch_size, 1, 1])

    # Mask out positions that are padded
    positions_mask = tf.sequence_mask(
        lengths=lengths, maxlen=maxlen, dtype=tf.float32)
    positions_embed = pe_batch * tf.expand_dims(positions_mask, 2)

    return positions_embed
