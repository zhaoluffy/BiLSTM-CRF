# coding=utf-8
from src.utils.data_utils import pad_sequences, iob_iobes, iob2
from src.utils.vocab_utils import lookup_vocab


class DataManager(object):

    def __init__(self,
                 data,
                 words_vocab_to_id,
                 chars_vocab_to_id,
                 tags_vocab_to_id,
                 batch_size,
                 pad_id=0,
                 use_char=False,
                 tag_scheme="iob"):
        self.data = data
        self.words_vocab_to_id = words_vocab_to_id
        self.chars_vocab_to_id = chars_vocab_to_id
        self.tags_vocab_to_id = tags_vocab_to_id
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.use_char = use_char
        self.tag_scheme = tag_scheme

    def minibatches(self):
        """
        得到最小批数据
        参数：
            data            -I      数据
            minibatch_size  -I      最小批次大小
        返回值：
            x_batch         -O      一批句子数据
            y_batch         -O      一批对应标签数据
        """
        x_batch, y_batch = [], []
        for (x, y) in self.data:
            if len(x_batch) == self.batch_size:
                word_ids, sequence_lengths, char_ids, word_lengths, labels_ids = \
                    self.pad_batch(x_batch, y_batch)
                yield word_ids, sequence_lengths, char_ids, word_lengths, labels_ids
                x_batch, y_batch = [], []

            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x]
            if iob2(y) and self.tag_scheme == "iobes":
                y = iob_iobes(y)
            y_batch += [y]

        if len(x_batch) != 0:
            word_ids, sequence_lengths, char_ids, word_lengths, labels_ids = \
                self.pad_batch(x_batch, y_batch)
            yield word_ids, sequence_lengths, char_ids, word_lengths, labels_ids

    @property
    def data_size(self):
        return len(self.data)

    def pad_batch(self, words, labels=None):
        if self.use_char:
            chars, words = zip(*words)
            char_ids = lookup_vocab(self.chars_vocab_to_id, chars)
            word_ids = lookup_vocab(self.words_vocab_to_id, words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids,
                                                   pad_tok=0,
                                                   nlevels=2)
        else:
            word_ids = lookup_vocab(self.words_vocab_to_id, words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids = None
            word_lengths = None

        label_ids = None
        if labels is not None:
            label_ids = lookup_vocab(self.tags_vocab_to_id, labels)
            label_ids, _ = pad_sequences(label_ids, 0)

        return word_ids, sequence_lengths, char_ids, word_lengths, label_ids




