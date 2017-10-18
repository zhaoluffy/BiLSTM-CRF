# coding=utf-8
"""Main script to run training and evaluation of models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import tensorflow as tf

from src.data.data_provider import CoNLLDataset
from src.utils.data_utils import get_vocabs, \
    get_glove_vocab, write_vocab, export_trimmed_glove_vectors, get_char_vocab
from src.utils.vocab_utils import create_vocabulary_mapping
from src.utils.vocab_utils import NUM, UNK


tf.flags.DEFINE_string("root_path", "data", """data root path.""")

tf.flags.DEFINE_string("train_file", "coNLL/eng.train", """trian file.""")

tf.flags.DEFINE_string("dev_file", "coNLL/eng.testa", """dev_file.""")

tf.flags.DEFINE_string("test_file", "coNLL/eng.testb", """test_file.""")

tf.flags.DEFINE_string("glove_file", "E:/models/glove.6B.100d.txt", """glove_file.""")
# tf.flags.DEFINE_string("glove_file", "/data/guozhao/models/glove.6B/glove.6B.100d.txt",  """glove_file.""")

tf.flags.DEFINE_string("trimmed_file", "coNLL/trimmed.glove.100", """trimmed_file.""")

tf.flags.DEFINE_string("vocab_words_file", "coNLL/vocab_words.eng", """vocab_words_file.""")

tf.flags.DEFINE_string("vocab_chars_file", "coNLL/vocab_chars.eng", """vocab_chars_file.""")

tf.flags.DEFINE_string("vocab_tags_file", "coNLL/vocab_tags.eng", """vocab_tags_file.""")

tf.flags.DEFINE_string("vocab_iobes_tags_file", "coNLL/vocab_iobes_tags.eng", """vocab_tags_file.""")

tf.flags.DEFINE_string("dim_word", 100, """dim_word.""")

FLAGS = tf.flags.FLAGS


def main(_argv):
    """
    构建数据，将从数据集中构建好的词典保存
    参数：
        _argv      -I      参数配置
    返回值：无
    """

    train_file = os.path.join(FLAGS.root_path, FLAGS.train_file)
    dev_file = os.path.join(FLAGS.root_path, FLAGS.dev_file)
    test_file = os.path.join(FLAGS.root_path, FLAGS.test_file)
    glove_file = os.path.join(FLAGS.root_path, FLAGS.glove_file)
    trimmed_file = os.path.join(FLAGS.root_path, FLAGS.trimmed_file)
    vocab_words_file = os.path.join(FLAGS.root_path, FLAGS.vocab_words_file)
    vocab_chars_file = os.path.join(FLAGS.root_path, FLAGS.vocab_chars_file)
    vocab_tags_file = os.path.join(FLAGS.root_path, FLAGS.vocab_tags_file)
    vocab_iobes_tags_file = os.path.join(FLAGS.root_path, FLAGS.vocab_iobes_tags_file)
    # Generators
    dev = CoNLLDataset(dev_file, lowercase=True)
    test = CoNLLDataset(test_file, lowercase=True)
    train = CoNLLDataset(train_file, lowercase=True)

    # Build Word and Tag vocab
    vocab_words, vocab_tags, iobes_vocab_tags = get_vocabs([train, dev, test], mode="iobes")
    vocab_glove = get_glove_vocab(glove_file)

    vocab = vocab_words & vocab_glove
    vocab.add(NUM)
    vocab.add(UNK)

    # Save vocab
    write_vocab(vocab, vocab_words_file)
    write_vocab(vocab_tags, vocab_tags_file)
    write_vocab(iobes_vocab_tags, vocab_iobes_tags_file)

    # Trim GloVe Vectors
    vocab, _, _, _ = create_vocabulary_mapping(vocab_words_file)
    export_trimmed_glove_vectors(vocab, glove_file,
                                 trimmed_file, FLAGS.dim_word)

    # Build and save char vocab
    train = CoNLLDataset(train_file)
    vocab_chars = get_char_vocab(train)
    vocab_chars.add(UNK)
    write_vocab(vocab_chars, vocab_chars_file)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
