# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vocabulary related functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow import gfile
from collections import OrderedDict


UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


def create_vocabulary_mapping(filename):
    """Creates a mapping for a vocabulary file.

    Args:
      filename: Path to a vocabulary file containg one word per line.
        Each word is mapped to its line number.
      default_value: UNK tokens will be mapped to this id.
        If None, UNK tokens will be mapped to [vocab_size]

      Returns:
        A tuple (vocab_to_id_table, id_to_vocab_table,
        word_to_count_table, vocab_size). The vocab size does not include
        the UNK token.
    """
    if not gfile.Exists(filename):
        raise ValueError("File does not exist: {}".format(filename))

    # Load vocabulary into memory
    with gfile.GFile(filename) as file:
        vocab = list(line.strip("\n") for line in file)
    vocab_size = len(vocab)

    has_counts = len(vocab[0].split("\t")) == 2
    if has_counts:
        vocab, counts = zip(*[_.split("\t") for _ in vocab])
        counts = [float(_) for _ in counts]
        vocab = list(vocab)
    else:
        counts = [-1. for _ in vocab]

    print("Creating vocabulary lookup table of size %d", vocab_size)

    vocab_idx = range(vocab_size)

    vocab_to_id_mapping = OrderedDict()
    id_to_vocab_mapping = OrderedDict()
    word_to_count_mapping = OrderedDict()
    for word, count, idx in zip(vocab, counts, vocab_idx):
        vocab_to_id_mapping[word] = idx
        id_to_vocab_mapping[idx] = word
        word_to_count_mapping[word] = count

    return vocab_to_id_mapping, id_to_vocab_mapping, word_to_count_mapping, vocab_size


def lookup_vocab(vocab, keys, unk_tok=UNK):
    types = (list, tuple, np.ndarray)
    if type(keys) in types and \
            type(keys[0]) in types and \
            type(keys[0][0]) in types:
        return [[[vocab[id] if id in vocab else vocab[unk_tok] for id in idx] for idx in idxs] for idxs in keys]
    elif type(keys) in types and type(keys[0]) in types:
        return [[vocab[idx] if idx in vocab else vocab[unk_tok] for idx in idxs] for idxs in keys]
    elif type(keys) in types:
        return [vocab[idx] if idx in vocab else vocab[unk_tok] for idx in keys]
    else:
        raise NotImplementedError


if __name__ == "__main__":
    _, id_to_vocab_mapping, _, size = create_vocabulary_mapping("../../data/coNLL/vocab_tags.eng")
    keys = [[[1,2,3],[1,2,3]], [[2,4,5], [3,4,5]]]
    map = lookup_vocab(id_to_vocab_mapping, keys)
    print(map)
