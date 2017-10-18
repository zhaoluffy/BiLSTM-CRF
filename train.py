"""Main script to run training and evaluation of models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from pydoc import locate

import numpy as np
import tensorflow as tf

import src.models.model_helper as model_helper
import src.utils.misc_utils as misc_utils
from src.data.data_manager import DataManager
from src.data.data_provider import CoNLLDataset
from src.utils import data_utils, vocab_utils
from src.utils.misc_utils import Progbar
from src.utils import evaluate_utils


def train(params, logger):
    cur_ckpt_dir = os.path.join(params.model_dir, params.cur_ckpt_dir)
    cur_checkpoint_path = os.path.join(cur_ckpt_dir, "cur_ner_model.trained_ckpt")
    # best_ckpt_dir = os.path.join(params.model_dir, params.best_ckpt_dir)
    best_checkpoint_path = os.path.join(cur_ckpt_dir, "ner_model.trained_ckpt")

    # get pre trained embeddings
    embeddings = data_utils.get_trimmed_glove_vectors(params.trimmed_file)
    print(embeddings.shape)

    # load vocabs
    words_vocab_to_id, words_id_to_vocab, _, words_vocab_size = \
        vocab_utils.create_vocabulary_mapping(params.vocab_words_file)
    chars_vocab_to_id, chars_id_to_vocab, _, chars_vocab_size = \
        vocab_utils.create_vocabulary_mapping(params.vocab_chars_file)
    tags_vocab_to_id, tags_id_to_vocab, _, tags_vocab_size = \
        vocab_utils.create_vocabulary_mapping(params.vocab_tags_file)

    params.model_params["count_vocab_source"] = words_vocab_size
    params.model_params["count_vocab_char"] = chars_vocab_size
    params.model_params["count_vocab_target"] = tags_vocab_size

    # create dataset
    eval_data = DataManager(CoNLLDataset(params.eval_file,
                                         lowercase=True,
                                         use_char=params.model_params["use_char"]),
                            words_vocab_to_id,
                            chars_vocab_to_id,
                            tags_vocab_to_id,
                            params.batch_size,
                            use_char=params.model_params["use_char"],
                            tag_scheme=params.tag_scheme
                            )
    test_data = DataManager(CoNLLDataset(params.test_file,
                                         lowercase=True,
                                         use_char=params.model_params["use_char"]),
                            words_vocab_to_id,
                            chars_vocab_to_id,
                            tags_vocab_to_id,
                            params.batch_size,
                            use_char=params.model_params["use_char"],
                            tag_scheme=params.tag_scheme
                            )
    train_data = DataManager(CoNLLDataset(params.train_file,
                                          lowercase=True,
                                          use_char=params.model_params["use_char"]),
                             words_vocab_to_id,
                             chars_vocab_to_id,
                             tags_vocab_to_id,
                             params.batch_size,
                             use_char=params.model_params["use_char"],
                             tag_scheme=params.tag_scheme
                             )
    len_train_data = train_data.data_size
    nbatches = (len_train_data + params.batch_size - 1) // params.batch_size

    # create model
    model_creator = locate(params.model)
    train_model = model_helper.create_model(model_creator, params.model_params,
                                            mode=tf.contrib.learn.ModeKeys.TRAIN)
    eval_model = model_helper.create_model(model_creator, params.model_params,
                                           mode=tf.contrib.learn.ModeKeys.EVAL)
    infer_model = model_helper.create_model(model_creator, params.model_params,
                                            mode=tf.contrib.learn.ModeKeys.INFER)
    # get config proto
    config_proto = misc_utils.get_config_proto(params.log_device_placement,
                                               params.allow_soft_placement)

    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)

    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            train_sess, train_model.model, cur_ckpt_dir, name="train", loaded_vector=embeddings)

    # start run epoch
    # losses = []
    best_score = 0
    nepoch_no_imprv = 0
    for epoch in range(params.num_epochs):
        prog = Progbar(target=nbatches)
        for i, batched_data in enumerate(train_data.minibatches()):
            # batched_data = train_data.pad_batch(words, labels)
            _, loss, global_step = loaded_train_model.train(train_sess, batched_data)
            prog.update(i + 1, [("train loss", loss)])
            # save current check point
        loaded_train_model.saver.save(train_sess, cur_checkpoint_path)

        logger.info(" epoch {} finished.".format(epoch+1))
        logger.info(" start internal evaluate")
        eval_loss, test_loss = run_internal_evaluate(
            eval_sess, eval_model, cur_ckpt_dir, eval_data, test_data)
        logger.info([" eval_loss:{:04.3f}".format(eval_loss),
                    " test_loss:{:04.3f} ".format(test_loss)])

        logger.info(" start external evaluate")
        eval_lines, test_lines = run_external_evaluate(
            infer_sess, infer_model, cur_ckpt_dir,
            words_id_to_vocab, tags_id_to_vocab, eval_data, test_data)

        for line in eval_lines:
            logger.info(line)
        if test_lines is not None:
            for line in test_lines:
                logger.info(line)
        eval_f1 = float(eval_lines[1].strip().split()[-1])

        if eval_f1 > best_score:
            nepoch_no_imprv = 0
            loaded_train_model.saver.save(train_sess, best_checkpoint_path, global_step=epoch + 1)
            best_score = eval_f1
            logger.info("- new best score!")
        else:
            nepoch_no_imprv += 1
            if nepoch_no_imprv >= params.nepoch_no_imprv:
                logger.info(
                    "- early stopping {} epochs without improvement".format(
                        nepoch_no_imprv))
                break


def run_internal_evaluate(eval_sess, eval_model, model_dir, eval_data, test_data=None):
    with eval_model.graph.as_default():
        loaded_eval_model, global_step = model_helper.create_or_load_model(
            eval_sess, eval_model.model, model_dir, name="eval")
    eval_loss = _internal_evaluate(eval_sess, loaded_eval_model, global_step, eval_data)
    test_loss = 0
    if test_data is not None:
        test_loss = _internal_evaluate(eval_sess, loaded_eval_model, global_step, test_data)
    return eval_loss, test_loss


def _internal_evaluate(sess, model, global_step, iterator):
    """Computing perplexity."""
    losses = []
    for i, batched_data in enumerate(iterator.minibatches()):
        # batched_data = iterator.pad_batch(words, labels)
        loss = model.eval(sess, batched_data)
        losses.append(loss)
    return np.mean(losses)


def run_external_evaluate(
        infer_sess, infer_model, model_dir,
        words_id_to_vocab, tags_id_to_vocab, eval_data, test_data=None):

    with infer_model.graph.as_default():
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            infer_sess, infer_model.model, model_dir, name="infer")
        eval_lines = _external_evaluate(
            infer_sess, loaded_infer_model, global_step,
            words_id_to_vocab, tags_id_to_vocab, eval_data)

        test_line = None
        if test_data is not None:
            test_line = _external_evaluate(
                infer_sess, loaded_infer_model, global_step,
                words_id_to_vocab, tags_id_to_vocab, test_data)

    return eval_lines, test_line


def _external_evaluate(
        sess, model, global_step,
        words_id_to_vocab, tags_id_to_vocab, iterator):

    ner_results = evaluate_utils.external_evaluate(
        sess, model, global_step, words_id_to_vocab, tags_id_to_vocab, iterator)
    eval_lines = evaluate_utils.test_ner(ner_results, global_step)
    return eval_lines
