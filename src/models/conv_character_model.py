"""Base class for models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from src import losses as seq2seq_losses
from src.models.model_base import ModelBase
from tensorflow.contrib.crf import viterbi_decode
from src.models import model_helper
from src.models import pooling_encoder
from pydoc import locate
from src.utils.conv_encoder_utils import parse_list_or_default, linear_mapping_weightnorm, conv_encoder_stack


class ConvCharacterModel(ModelBase):
    """Base class for seq2seq models with embeddings
    """

    def __init__(self, params, mode, name="birnn_model"):
        super(ConvCharacterModel, self).__init__(params, mode, name)
        self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])

        # placeholders
        self.source_placeholder = tf.placeholder(tf.int32,
                                                 shape=[None, None],
                                                 name="source_placeholder")
        self.seq_len_palceholder = tf.placeholder(tf.int32,
                                                  shape=[None],
                                                  name="seq_len_palceholder")
        self.src_char_palceholder = tf.placeholder(tf.int32,
                                                   shape=[None, None, None],
                                                   name="src_char_palceholder")
        self.word_len_placeholder = tf.placeholder(tf.int32,
                                                   shape=[None, None],
                                                   name="word_len_placeholder")
        self.target_palceholder = tf.placeholder(tf.int32,
                                                 shape=[None, None],
                                                 name="target_palceholder")

        self.loss, self.logits, self.train_op, self.transition_param = \
            self.build_graph()

        # Saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    @staticmethod
    def default_params():
        params = ModelBase.default_params()
        params.update({
            "dropout_keep_prob": 0.5,
            "embedding.word_dim": 300,
            "embedding.char_dim": 100,
            "embedding.init_scale": 0.04,
            "optimizer.clip_embed_gradients": 0.1,
            "count_vocab_source": 0,
            "count_vocab_char": 0,
            "count_vocab_target": 0,
            "use_char": False,
            "cnn.layers": 4,
            "cnn.nhids": "100,100,100,100",
            "cnn.kwidths": "3,3,3,3",
            "cnn.nhid_default": 100,
            "cnn.kwidth_default": 3,
            "embedding_dropout_keep_prob": 0.9,
            "nhid_dropout_keep_prob": 0.9,
            "position_embeddings.enable": True,
            "position_embeddings.combiner_fn": "tensorflow.add",
            "position_embeddings.num_positions": 50,
            "encoder.num_units": 100,
            "keep_checkpoint_max": 5
        })
        return params

    @property
    def word_embedding(self):
        return self._word_embeddings

    def embedding(self):
        with tf.variable_scope("words"):
            self._word_embeddings = tf.get_variable(name="_word_embeddings",
                                                    shape=[self.params["count_vocab_source"],
                                                           self.params["embedding.word_dim"]],
                                                    trainable=False
                                                    )
            word_embeddings = tf.nn.embedding_lookup(self._word_embeddings, self.source_placeholder,
                                                     name="word_embeddings")
        with tf.variable_scope("chars"):
            if self.params["use_char"]:
                # get embeddings matrix
                _char_embeddings = tf.get_variable(name="_char_embeddings",
                                                   shape=[self.params["count_vocab_char"],
                                                          self.params["embedding.char_dim"]],
                                                   dtype=tf.float32
                                                   )
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.src_char_palceholder,
                                                         name="char_embeddings")
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.params["embedding.char_dim"]])
                word_lengths = tf.reshape(self.word_len_placeholder, shape=[-1])
                # bi lstm on chars
                # need 2 instances of cells since tf 1.1
                if self.params["position_embeddings.enable"]:
                    positions_embed = pooling_encoder._create_position_embedding(
                        embedding_dim=char_embeddings.get_shape().as_list()[-1],
                        num_positions=self.params["position_embeddings.num_positions"],
                        lengths=word_lengths,
                        maxlen=tf.shape(char_embeddings)[1])
                    char_embeddings = self._combiner_fn(char_embeddings, positions_embed)

                embed_size = char_embeddings.get_shape().as_list()[-1]
                with tf.variable_scope("encoder_cnn"):
                    next_layer = char_embeddings
                    if self.params["cnn.layers"] > 0:
                        nhids_list = parse_list_or_default(self.params["cnn.nhids"], self.params["cnn.layers"],
                                                           self.params["cnn.nhid_default"])
                        kwidths_list = parse_list_or_default(self.params["cnn.kwidths"], self.params["cnn.layers"],
                                                             self.params["cnn.kwidth_default"])

                        # mapping emb dim to hid dim
                        next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0],
                                                               dropout=self.params["embedding_dropout_keep_prob"],
                                                               var_scope_name="linear_mapping_before_cnn")
                        next_layer = conv_encoder_stack(next_layer, nhids_list, kwidths_list,
                                                        {'src': self.params["embedding_dropout_keep_prob"],
                                                         'hid': self.params["nhid_dropout_keep_prob"]}, mode=self.mode)

                        next_layer = linear_mapping_weightnorm(next_layer, embed_size,
                                                               var_scope_name="linear_mapping_after_cnn")
                    ## The encoder stack will receive gradients *twice* for each attention pass: dot product and weighted sum.
                    ##cnn = nn.GradMultiply(cnn, 1 / (2 * nattn))
                    cnn_c_output = (next_layer + char_embeddings) * tf.sqrt(0.5)

                output = tf.reduce_mean(cnn_c_output, 1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[-1, s[1], self.params["embedding.char_dim"]])

                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            word_embeddings = tf.nn.dropout(word_embeddings, self.params["dropout_keep_prob"])

        return word_embeddings

    def encode(self, word_embeddings):
        num_unit = self.params["encoder.num_units"]
        with tf.variable_scope("words_encoder"):
            output = model_helper.bidirectional_LSTM(
                word_embeddings, num_unit, self.seq_len_palceholder, output_sequence=True)
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                output = tf.nn.dropout(output, self.params["dropout_keep_prob"])

        with tf.variable_scope("dense"):
            W = tf.get_variable("W", shape=[2 * num_unit, self.params["count_vocab_target"]],
                                dtype=tf.float32)

            b = tf.get_variable("b", shape=[self.params["count_vocab_target"]],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * num_unit])
            logits = tf.matmul(output, W) + b
            logits = tf.reshape(logits, [-1, ntime_steps, self.params["count_vocab_target"]])

        return logits

    def compute_loss(self, logits):
        """Computes the loss for this model.

        Returns a tuple `(losses, loss)`, where `losses` are the per-batch
        losses and loss is a single scalar tensor to minimize.
        """
        loss, transition_param = seq2seq_losses.crf_loss(
            logits=logits,
            targets=self.target_palceholder,
            sequence_length=self.seq_len_palceholder)

        return loss, transition_param

    def build_graph(self):
        # Pre-process features and labels
        with tf.variable_scope(self.name):
            word_embedded = self.embedding()
            logits = self.encode(word_embedded)

            loss, transition_param = self.compute_loss(logits)

            train_op = None
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                train_op = self._build_train_op(loss)

        return loss, logits, train_op, transition_param

    def create_feed_dict(self, batch):
        """
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        sources, seq_lens, src_chars, word_lens, targets = batch
        feed_dict = {
            self.source_placeholder: sources,
            self.seq_len_palceholder: seq_lens,
            self.target_palceholder: targets
        }
        if self.params["use_char"]:
            feed_dict[self.src_char_palceholder] = src_chars
            feed_dict[self.word_len_placeholder] = word_lens

        return feed_dict

    def train(self, sess, batch):
        """
        :param sess: session to run the batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        feed_dict = self.create_feed_dict(batch)
        return sess.run([self.train_op,
                         self.loss,
                         self.global_step
                         ],
                        feed_dict=feed_dict)

    def eval(self, sess, batch):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        feed_dict = self.create_feed_dict(batch)
        return sess.run([self.loss], feed_dict=feed_dict)

    def infer(self, sess, batch):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        feed_dict = self.create_feed_dict(batch)
        return sess.run([self.logits,
                         self.seq_len_palceholder,
                         self.transition_param],
                        feed_dict=feed_dict)

    def decode(self, sess, batch):
        """Creates the dictionary of predictions that is returned by the model.
        """
        logits, sequence_lengths, transition_param = self.infer(sess, batch=batch)
        predicted_ids = []
        for logit, length in zip(logits, sequence_lengths):
            logit = logit[:length]
            predicted_id, _ = viterbi_decode(logit, transition_param)
            predicted_ids.append(predicted_id)

        return predicted_ids
