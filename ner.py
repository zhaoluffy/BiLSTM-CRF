"""Main script to run training and evaluation of models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile

import yaml

import tensorflow as tf
from tensorflow import gfile

from src.configurable import _maybe_load_yaml
from src.configurable import _deep_merge_dict
import src.utils.misc_utils as misc_utils
from train import train


tf.flags.DEFINE_string("config_paths", "./configs/ner_conv_character.yml",
                       """Path to a YAML configuration files defining FLAG
                       values. Multiple files can be separated by commas. Files are merged recursively. Setting a key 
                       in these files is equivalent to setting the FLAG value with the same name.""")
tf.flags.DEFINE_string("model", "", """Name of the model class.""")
tf.flags.DEFINE_string("model_params", "{}", """YAML configuration string for the model parameters.""")

tf.flags.DEFINE_string("train_file", "data/coNLL/eng.train", """train file.""")
tf.flags.DEFINE_string("eval_file", "data/coNLL/eng.testa", """evaluate file.""")
tf.flags.DEFINE_string("test_file", "data/coNLL/eng.testb", """test file.""")

tf.flags.DEFINE_string("vocab_words_file", "data/coNLL/vocab_words.eng", """vocab_words_file.""")
tf.flags.DEFINE_string("vocab_chars_file", "data/coNLL/vocab_chars.eng", """vocab_chars_file.""")
tf.flags.DEFINE_string("vocab_tags_file", "data/coNLL/vocab_iobes_tags.eng", """vocab_tags_file.""")
tf.flags.DEFINE_string("trimmed_file", "data/coNLL/trimmed.glove.100.npz",  """trimmed_file.""")
tf.flags.DEFINE_string("tag_scheme", "iobes", """tag scheme.""")

tf.flags.DEFINE_integer("batch_size", 20, """Batch size used for training and evaluation.""")
tf.flags.DEFINE_string("model_dir", "./trained_model",
                       """The directory to write model checkpoints and summaries.""")
tf.flags.DEFINE_string("cur_ckpt_dir", "ckpt",
                       """The directory to write best model checkpoints and summaries to. """)
tf.flags.DEFINE_string("best_ckpt_dir", "best_ckpt",
                       """The directory to write best model checkpoints and summaries to. """)
tf.flags.DEFINE_string("log_file", "train.log", """log file.""")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", 50, """Maximum number of training epoch to run.  If None, train forever.""")
tf.flags.DEFINE_integer("display_every_n_steps", 10, "Display on validation data every N steps.")
tf.flags.DEFINE_integer("nepoch_no_imprv", 10, "Display on validation data every N steps.")

# RunConfig Flags
tf.flags.DEFINE_float("allow_soft_placement", True,
                      """Fraction of GPU memory used by the process on each GPU uniformly on the same machine.""")
tf.flags.DEFINE_boolean("log_device_placement", False, """Log the op placement to devices""")


FLAGS = tf.flags.FLAGS


def main(_argv):
    """The entrypoint for the script"""
    misc_utils.clean(FLAGS)
    misc_utils.make_path(FLAGS)
    log_file = os.path.join(FLAGS.model_dir, FLAGS.log_file)
    logger = misc_utils.get_logger(log_file)

    # Parse YAML FLAGS
    FLAGS.model_params = _maybe_load_yaml(FLAGS.model_params)

    # Load flags from config file
    final_config = {}
    if FLAGS.config_paths:
        for config_path in FLAGS.config_paths.split(","):
            config_path = config_path.strip()
            if not config_path:
                continue
            config_path = os.path.abspath(config_path)
            logger.info("Loading config from %s", config_path)
            with gfile.GFile(config_path.strip()) as config_file:
                config_flags = yaml.load(config_file)
                final_config = _deep_merge_dict(final_config, config_flags)

    logger.info("Final Config:\n%s", yaml.dump(final_config))

    # Merge flags with config values
    for flag_key, flag_value in final_config.items():
        if hasattr(FLAGS, flag_key) and isinstance(getattr(FLAGS, flag_key), dict):
            merged_value = _deep_merge_dict(flag_value, getattr(FLAGS, flag_key))
            setattr(FLAGS, flag_key, merged_value)
        elif hasattr(FLAGS, flag_key):
            setattr(FLAGS, flag_key, flag_value)
        else:
            logger.warning("Ignoring config flag: %s", flag_key)

    if not FLAGS.model_dir:
        FLAGS.output_dir = tempfile.mkdtemp()

    train(FLAGS, logger)


if __name__ == "__main__":
    tf.app.run()
